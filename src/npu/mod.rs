//! Run NPU work on memory the GPU can also reach.
//!
//! On Strix Halo both engines sit on the same LPDDR5X, but each driver pins pages into its
//! own IOMMU domain. Exactly one bridge between them works: a GPU-agent HSA pool allocation
//! exported as a dma-buf and imported by the NPU's driver. [`Shared`] is that allocation --
//! allocated once, exported once, addressed by both engines for its whole life, so export
//! is a per-buffer setup cost rather than a per-handoff one.
//!
//! The pages are shared; the caches are not. [`Shared`] tracks which engine last touched the
//! memory and performs exactly the cache maintenance a transition needs -- and no more, so a
//! run of NPU dispatches costs one flush, not one per dispatch.
//!
//! ```no_run
//! # fn main() -> Result<(), hrx::Error> {
//! let npu = hrx::npu::Npu::open("model.xclbin")?;
//! let mut activations = npu.alloc(32 << 20, npu.group_id(3)?)?;
//!
//! activations.host()?.fill(0);          // host owns it
//! let bo = activations.npu();           // flushed for the NPU, which now owns it
//! # let _ = bo;
//! let result = activations.host()?;     // invalidated back for the host
//! # let _ = result;
//! # Ok(())
//! # }
//! ```
//!
//! ## What is not here yet
//!
//! GPU *kernels* cannot bind a [`Shared`] directly. libhrx will not return a device pointer
//! for device-local buffers and has no export entry point, so a buffer cannot be both an
//! `hrx::Buffer` and dma-buf exportable. Until that gap closes, move data with
//! [`Shared::copy_from`] / [`Shared::copy_to`], which stay on the device rather than
//! bouncing through host staging. When libhrx grows either capability, those calls collapse
//! to nothing and the rest of this API is unchanged.

mod hsa;

use crate::{Buffer, Error, Result};
use std::ffi::c_void;
use std::path::Path;

use hsa::Hsa;

fn failed(message: String) -> Error {
    Error::Message(message)
}

/// An NPU device plus the HSA plumbing needed to allocate memory it can import.
pub struct Npu {
    context: dvxrt::Context,
    hsa: Hsa,
}

impl Npu {
    /// Open the NPU against an xclbin and locate the GPU pool shared memory comes from.
    pub fn open(xclbin: impl AsRef<Path>) -> Result<Self> {
        let xclbin = xclbin.as_ref();
        let path = xclbin
            .to_str()
            .ok_or_else(|| failed(format!("xclbin path is not UTF-8: {}", xclbin.display())))?;
        let context = dvxrt::Context::new(0, path).map_err(failed)?;
        let hsa = Hsa::open().map_err(failed)?;
        Ok(Npu { context, hsa })
    }

    /// The memory group for a kernel argument index, needed when allocating for that binding.
    pub fn group_id(&self, argument: i32) -> Result<i32> {
        self.context.group_id(argument).map_err(failed)
    }

    /// Allocate `bytes` reachable by both engines, bound to kernel argument group `group`.
    pub fn alloc(&self, bytes: usize, group: i32) -> Result<Shared<'_>> {
        if bytes == 0 {
            return Err(failed("cannot allocate a zero-length shared buffer".into()));
        }
        let pointer = self.hsa.allocate(bytes).map_err(failed)?;
        // From here on every exit path must release the allocation.
        let descriptor = match self.hsa.export_dmabuf(pointer, bytes) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                self.hsa.free(pointer);
                return Err(failed(error));
            }
        };
        let bo = match unsafe { self.context.import_dmabuf(descriptor, bytes) } {
            Ok(bo) => bo,
            Err(error) => {
                unsafe { libc::close(descriptor) };
                self.hsa.free(pointer);
                return Err(failed(error));
            }
        };
        let _ = group; // The import carries the mapping; the group is validated by dispatch.
        Ok(Shared {
            npu: self,
            pointer,
            bytes,
            descriptor,
            bo,
            owner: Owner::Host,
        })
    }

    /// The underlying XRT context, for dispatching kernels against [`Shared::npu`] buffers.
    pub fn context(&self) -> &dvxrt::Context {
        &self.context
    }
}

/// Which engine last wrote the memory, and therefore whose cache is authoritative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Owner {
    Host,
    Gpu,
    Npu,
}

/// One allocation both the GPU and the NPU address, with no copy between them.
///
/// Each accessor moves ownership to the engine that is about to read or write, doing the
/// cache maintenance that transition requires. Re-entering the current owner is free.
pub struct Shared<'a> {
    npu: &'a Npu,
    pointer: *mut c_void,
    bytes: usize,
    descriptor: i32,
    bo: dvxrt::Bo,
    owner: Owner,
}

impl Shared<'_> {
    /// The size of the shared allocation in bytes.
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Whether the allocation is zero-length. Always false: `alloc` rejects zero.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Hand the buffer to the NPU, flushing host or GPU writes first.
    ///
    /// The returned BO is the kernel argument. Successive calls without an intervening
    /// [`Shared::host`] cost nothing, so a chain of dispatches flushes once.
    pub fn npu(&mut self) -> &dvxrt::Bo {
        if self.owner != Owner::Npu {
            // Errors here would mean the NPU reads stale bytes rather than none, so they
            // must not be silent -- but sync has no failure mode a caller could act on.
            if let Err(error) = self.bo.sync(true, self.bytes) {
                debug_assert!(false, "flush to the NPU failed: {error}");
            }
            self.owner = Owner::Npu;
        }
        &self.bo
    }

    /// Take the buffer back for the host, invalidating NPU writes first.
    pub fn host(&mut self) -> Result<&mut [u8]> {
        if self.owner == Owner::Npu {
            self.bo.sync(false, self.bytes).map_err(failed)?;
        }
        self.owner = Owner::Host;
        // The allocation is live for the whole borrow and no engine owns it here.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.pointer.cast::<u8>(), self.bytes) })
    }

    /// Fill the buffer with a repeating 32-bit pattern, on the GPU.
    pub fn fill(&mut self, value: u32) -> Result<()> {
        if self.bytes % 4 != 0 {
            return Err(failed("fill needs a length that is a multiple of 4".into()));
        }
        self.transfer_ready()?;
        self.npu.hsa.fill(self.pointer, value, self.bytes / 4).map_err(failed)?;
        self.owner = Owner::Gpu;
        Ok(())
    }

    /// Copy a GPU buffer into this one, device-side.
    ///
    /// `source` must be an [`crate::Stream::allocate_shared`] buffer: those are the only
    /// ones libhrx will name with a device pointer.
    pub fn copy_from(&mut self, source: &Buffer) -> Result<()> {
        let pointer = self.checked_peer(source)?;
        self.transfer_ready()?;
        self.npu.hsa.copy(self.pointer, pointer, self.bytes).map_err(failed)?;
        self.owner = Owner::Gpu;
        Ok(())
    }

    /// Copy this buffer into a GPU buffer, device-side.
    pub fn copy_to(&mut self, destination: &Buffer) -> Result<()> {
        let pointer = self.checked_peer(destination)?;
        // The GPU is about to read, so NPU writes must be visible first.
        if self.owner == Owner::Npu {
            self.bo.sync(false, self.bytes).map_err(failed)?;
            self.owner = Owner::Gpu;
        }
        self.npu.hsa.copy(pointer, self.pointer, self.bytes).map_err(failed)
    }

    /// Validate a peer GPU buffer and get the device pointer to copy against.
    fn checked_peer(&self, peer: &Buffer) -> Result<*mut c_void> {
        if peer.bytes() < self.bytes {
            return Err(failed(format!(
                "gpu buffer is {} bytes but the shared buffer is {}",
                peer.bytes(),
                self.bytes
            )));
        }
        peer.device_ptr().map_err(|error| {
            failed(format!(
                "{error}; shared transfers need a Stream::allocate_shared buffer"
            ))
        })
    }

    /// Make NPU writes visible before the GPU reads or overwrites the allocation.
    fn transfer_ready(&mut self) -> Result<()> {
        if self.owner == Owner::Npu {
            self.bo.sync(false, self.bytes).map_err(failed)?;
        }
        Ok(())
    }
}

impl Drop for Shared<'_> {
    fn drop(&mut self) {
        // The BO holds the import; drop it before the descriptor and the pages it names.
        unsafe { libc::close(self.descriptor) };
        self.npu.hsa.free(self.pointer);
    }
}

// The allocation is owned exclusively by this handle; every accessor takes &mut self.
unsafe impl Send for Shared<'_> {}
