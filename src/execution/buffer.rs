use super::{Access, RuntimeOwner};
use crate::{Error, Result};
use std::{
    ops::{Deref, DerefMut, Range},
    sync::{Arc, Mutex},
};

/// Explicit storage placement. Shared storage requires a resident NPU program
/// to create its device mapping once, before execution.
#[derive(Clone)]
pub enum MemoryPlacement {
    /// GPU pool memory for GPU-only work.
    GpuLocal,
    /// Coherent host-local memory mapped for guarded CPU access and GPU copies.
    /// Does not require an NPU or its runtime.
    HostVisible,
    /// NPU host-only memory, bound to the program's host memory group.
    #[cfg(feature = "npu")]
    NpuLocal(crate::npu::NpuProgram),
    /// One exportable GPU allocation, imported into the NPU without copying.
    #[cfg(feature = "npu")]
    Shared(crate::npu::NpuProgram),
}
#[derive(Default)]
pub(super) struct HostState {
    pub readers: usize,
    pub writer: bool,
    pub external: bool,
    pub poison: Option<String>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Engine {
    Host,
    Gpu,
    Npu,
}
pub(super) struct Visibility {
    pub writer: Engine,
    pub visible: [bool; 3],
}
impl Visibility {
    pub fn new() -> Self {
        Self {
            writer: Engine::Host,
            visible: [true, false, false],
        }
    }
    #[cfg(any(test, feature = "npu"))]
    pub fn sync_to_device(&self, consumer: Engine) -> bool {
        self.writer == Engine::Host || consumer == Engine::Npu
    }
    pub fn wrote(&mut self, engine: Engine) {
        self.writer = engine;
        self.visible = [false; 3];
        self.visible[engine as usize] = true;
    }
}
pub(super) struct Storage {
    pub accounted: bool,
    // Imported mappings drop before descriptor and backing GPU allocation.
    #[cfg(feature = "npu")]
    pub bo: Option<crate::npu::raw::Bo>,
    #[cfg(feature = "npu")]
    pub descriptor: Option<std::os::fd::OwnedFd>,
    pub gpu: Option<crate::gpu::Buffer>,
    pub pointer: *mut u8,
    pub bytes: usize,
    #[cfg(feature = "npu")]
    pub npu_device: Option<i32>,
    #[cfg(feature = "npu")]
    pub group: Option<i32>,
    pub shared: bool,
    pub host: Mutex<HostState>,
    pub visibility: Mutex<Visibility>,
    pub runtime: Arc<RuntimeOwner>,
    #[cfg(test)]
    pub _test_memory: Option<Arc<std::cell::UnsafeCell<[u8; 256]>>>,
}
// All host dereferences require a host lease; device access is reserved under
// the scheduler lock. Native handles never escape this tracked allocation.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}
/// A cloneable, tracked allocation. Every alias shares lifetime and access state.
#[derive(Clone)]
pub struct Buffer {
    pub(super) storage: Arc<Storage>,
}
/// An owned checked view. Keeping a view keeps its root allocation alive.
#[derive(Clone)]
pub struct BufferView {
    pub(super) buffer: Buffer,
    pub(super) range: Range<usize>,
}
impl Buffer {
    /// Allocation size in bytes.
    pub fn len(&self) -> usize {
        self.storage.bytes
    }
    /// Whether the allocation is empty. Allocations reject zero lengths.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow all bytes as an owned device binding.
    pub fn view(&self) -> BufferView {
        BufferView {
            buffer: self.clone(),
            range: 0..self.len(),
        }
    }
    /// Create a checked subregion.
    pub fn slice(&self, range: Range<usize>) -> Result<BufferView> {
        self.view().slice(range)
    }
    /// Wait for device writers and map initialized bytes for host reading.
    /// Live conflicting host guards return `Busy` instead of waiting.
    pub fn map_read(&self) -> Result<ReadGuard<'_>> {
        self.acquire(false, true)?;
        Ok(ReadGuard { buffer: self })
    }
    /// Map for reading without waiting for device work.
    pub fn try_map_read(&self) -> Result<ReadGuard<'_>> {
        self.acquire(false, false)?;
        Ok(ReadGuard { buffer: self })
    }
    /// Wait for device access and map initialized bytes for host mutation.
    pub fn map_write(&self) -> Result<WriteGuard<'_>> {
        self.acquire(true, true)?;
        Ok(WriteGuard { buffer: self })
    }
    /// Map for mutation without waiting for device work.
    pub fn try_map_write(&self) -> Result<WriteGuard<'_>> {
        self.acquire(true, false)?;
        Ok(WriteGuard { buffer: self })
    }
    fn acquire(&self, write: bool, wait: bool) -> Result<()> {
        self.acquire_with(write, wait, || self.storage.make_visible(Engine::Host))
    }
    // Fault-injection seam for testing lease rollback when cache maintenance fails.
    pub(super) fn acquire_with(
        &self,
        write: bool,
        wait: bool,
        make_visible: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if self.storage.pointer.is_null() {
            return Err(Error::Unsupported(
                "GPU-local memory is not host mapped; use an explicit graph copy to shared storage"
                    .into(),
            ));
        }
        let core = &self.storage.runtime.core;
        let view = self.view();
        let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            {
                let host = self.storage.host.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(error) = &host.poison {
                    return Err(Error::DeviceLost(error.clone()));
                }
                if host.external || host.writer || (write && host.readers != 0) {
                    return Err(Error::Busy("conflicting host mapping".into()));
                }
            }
            if !scheduler.conflicts(
                &view,
                if write {
                    Access::ReadWrite
                } else {
                    Access::Read
                },
            ) {
                break;
            }
            if !wait {
                return Err(Error::Busy("allocation is in device use".into()));
            }
            scheduler = core
                .host_changed
                .wait(scheduler)
                .unwrap_or_else(|e| e.into_inner());
        }
        let mut host = self.storage.host.lock().unwrap_or_else(|e| e.into_inner());
        if write {
            host.writer = true;
        } else {
            host.readers += 1;
        }
        // Reserve the lease while submissions are excluded, then let unrelated
        // allocations proceed during potentially expensive cache maintenance.
        drop(host);
        drop(scheduler);
        if let Err(error) = make_visible() {
            let mut host = self.storage.host.lock().unwrap_or_else(|e| e.into_inner());
            if write {
                host.writer = false;
            } else {
                host.readers -= 1;
            }
            drop(host);
            core.host_changed.notify_all();
            return Err(error);
        }
        Ok(())
    }
}
impl BufferView {
    /// Size of this view.
    pub fn len(&self) -> usize {
        self.range.len()
    }
    /// Whether this view is empty.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
    /// Byte offset within the root allocation.
    pub fn offset(&self) -> usize {
        self.range.start
    }
    /// Create a subregion relative to this view.
    pub fn slice(&self, range: Range<usize>) -> Result<Self> {
        if range.start > range.end || range.end > self.len() {
            return Err(Error::Message("buffer view is out of bounds".into()));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            range: self.range.start + range.start..self.range.start + range.end,
        })
    }
    pub(super) fn overlaps(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer.storage, &other.buffer.storage)
            && self.range.start < other.range.end
            && other.range.start < self.range.end
    }
    pub(super) fn conflicts(&self, other: &Self) -> bool {
        // Cache maintenance is allocation-wide in the initial shared-memory ABI.
        Arc::ptr_eq(&self.buffer.storage, &other.buffer.storage)
            && (self.buffer.storage.shared || self.overlaps(other))
    }
    pub(super) fn gpu(&self) -> Result<crate::gpu::View<'_>> {
        self.buffer
            .storage
            .gpu
            .as_ref()
            .ok_or_else(|| Error::Unsupported("NPU-local storage cannot bind to the GPU".into()))?
            .try_slice(self.offset(), self.len())
    }
}
impl Storage {
    pub fn host_conflict(&self, access: Access) -> Result<()> {
        let host = self.host.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = &host.poison {
            return Err(Error::DeviceLost(error.clone()));
        }
        if host.external || host.writer || (access.writes() && host.readers != 0) {
            return Err(Error::Busy(
                "device binding conflicts with a live host mapping".into(),
            ));
        }
        Ok(())
    }
    pub fn make_visible(&self, engine: Engine) -> Result<()> {
        let mut visibility = self.visibility.lock().unwrap_or_else(|e| e.into_inner());
        if visibility.visible[engine as usize] {
            return Ok(());
        }
        #[cfg(feature = "npu")]
        if let Some(bo) = &self.bo {
            // Only cross-engine transitions require maintenance. Producer work
            // has already completed before this method can be reached. Host-dirty
            // lines must be flushed even when the consumer is the GPU.
            bo.sync(visibility.sync_to_device(engine), self.bytes)
                .map_err(Error::Message)?;
            self.runtime
                .core
                .counters
                .cache_maintenance_bytes
                .fetch_add(self.bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
        visibility.visible[engine as usize] = true;
        Ok(())
    }
}
/// A host read lease. The slice cannot outlive the guard.
///
/// ```compile_fail
/// # use hrx::execution::Buffer;
/// fn escape(buffer: &Buffer) -> &[u8] {
///     let guard = buffer.map_read().unwrap();
///     &guard
/// }
/// ```
pub struct ReadGuard<'a> {
    buffer: &'a Buffer,
}
impl Deref for ReadGuard<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.storage.pointer, self.buffer.len()) }
    }
}
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.buffer
            .storage
            .host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .readers -= 1;
        self.buffer.storage.runtime.core.host_changed.notify_all();
    }
}
/// An exclusive host write lease. Releasing it marks host writes authoritative.
pub struct WriteGuard<'a> {
    buffer: &'a Buffer,
}
impl Deref for WriteGuard<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.storage.pointer, self.buffer.len()) }
    }
}
impl DerefMut for WriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.storage.pointer, self.buffer.len()) }
    }
}
impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        // Publish dirty state before releasing the host lease.
        self.buffer
            .storage
            .visibility
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .wrote(Engine::Host);
        self.buffer
            .storage
            .host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .writer = false;
        self.buffer.storage.runtime.core.host_changed.notify_all();
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        #[cfg(feature = "npu")]
        if self
            .visibility
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .writer
            == Engine::Host
            && let Some(bo) = &self.bo
            && bo.sync(true, self.bytes).is_err()
        {
            // Do not release pages while failed cache maintenance could leave
            // CPU writes able to reach a later allocation of those pages.
            std::mem::forget((self.bo.take(), self.descriptor.take(), self.gpu.take()));
            return;
        }

        if self.accounted {
            self.runtime
                .core
                .counters
                .live_bytes
                .fetch_sub(self.bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
