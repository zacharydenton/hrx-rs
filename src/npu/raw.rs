// SPDX-License-Identifier: Apache-2.0
// Derived from dinov3-xdna2 crates/xrt; see native/npu/NOTICE.
//! Safe Rust wrapper over the C XRT shim (`xrt-shim/shim.cpp`).
//!
//! Soundness: each [`Bo`] holds shared ownership of its native wrapper and XRT
//! context. In-flight runs retain BO leases, synchronously abort on timeout, and
//! track overlapping ranges so safe host operations cannot reuse active memory.
//! Raw instruction dispatch remains `unsafe` because XRT cannot validate the DMA
//! semantics encoded inside an instruction stream.
#![allow(missing_docs, unsafe_op_in_unsafe_fn)]
// FFI shim over XRT: deliberate width casts (handles/sizes), raw-pointer idioms, internal-crate doc rules.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::ptr_as_ptr,
    clippy::borrow_as_ptr,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]

use std::ffi::CString;
use std::ops::Range;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[repr(C)]
struct DvCtx {
    _private: [u8; 0],
}
#[repr(C)]
struct DvBo {
    _private: [u8; 0],
}

macro_rules! native_api {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {
        struct Api { $($name: unsafe extern "C" fn($($ty),*) $(-> $ret)?,)* }
        static API: std::sync::OnceLock<Api> = std::sync::OnceLock::new();
        fn initialize() -> Result<(), String> {
            static LOCK: Mutex<()> = Mutex::new(());
            let _lock = LOCK.lock().map_err(|_| "XRT loader poisoned")?;
            if API.get().is_some() { return Ok(()); }
            let library = super::load_shim().map_err(|e| e.to_string())?;
            unsafe {
                let api = Api { $($name: *library.get(concat!(stringify!($name), "\0").as_bytes()).map_err(|e| e.to_string())?,)* };
                // XRT process globals and abandoned native runs may retain code.
                std::mem::forget(library);
                let _ = API.set(api);
            }
            Ok(())
        }
        $(unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
            unsafe { (API.get().expect("initialized XRT context").$name)($($arg),*) }
        })*
    };
}
native_api! {
    fn dv_last_error() -> *const c_char;
    fn dv_ctx_create(dev_idx: c_int, path: *const c_char) -> *mut DvCtx;
    fn dv_ctx_free(c: *mut DvCtx);
    fn dv_kernel_group_id(c: *mut DvCtx, argidx: c_int) -> c_int;
    fn dv_bo_alloc(c: *mut DvCtx, nbytes: usize, kind: c_int, group_id: c_int) -> *mut DvBo;
    fn dv_bo_import(c: *mut DvCtx, userptr: *mut c_void, nbytes: usize, kind: c_int, group_id: c_int) -> *mut DvBo;
    fn dv_bo_import_dmabuf(c: *mut DvCtx, fd: c_int, nbytes: usize) -> *mut DvBo;
    fn dv_bo_suballoc(parent: *mut DvBo, size: usize, offset: usize) -> *mut DvBo;
    fn dv_bo_write(b: *mut DvBo, src: *const c_void, nbytes: usize) -> c_int;
    fn dv_bo_read(b: *mut DvBo, dst: *mut c_void, nbytes: usize) -> c_int;
    fn dv_bo_free(b: *mut DvBo);
    fn dv_bo_address(b: *mut DvBo) -> u64;
    fn dv_bo_map(b: *mut DvBo) -> *mut c_void;
    fn dv_bo_sync(b: *mut DvBo, to_device: c_int, nbytes: usize) -> c_int;
    fn dv_run_start(
        c: *mut DvCtx,
        insts: *mut DvBo,
        insts_nbytes: u32,
        a: *mut DvBo,
        b: *mut DvBo,
        cbo: *mut DvBo,
    ) -> *mut c_void;
    fn dv_run_prepare(c: *mut DvCtx, insts: *mut DvBo, words: u32, args: *const *mut DvBo, count: usize) -> *mut c_void;
    fn dv_prepared_execute(handle: *mut c_void, timeout_ms: u32) -> c_int;
    fn dv_prepared_free(handle: *mut c_void) -> c_int;
    fn dv_run_start_args(c: *mut DvCtx, insts: *mut DvBo, words: u32, args: *const *mut DvBo, count: usize) -> *mut c_void;
    fn dv_run_wait(handle: *mut c_void, timeout_ms: u32) -> c_int;
    fn dv_run_cancel(handle: *mut c_void) -> c_int;
}

/// An in-flight async dispatch (CPU/NPU overlap). Submit with `Context::run_start`, do host work, then
/// `wait()` to block on completion (returns the ERT state; consumes the handle and frees the underlying run).
pub struct RunHandle {
    ptr: *mut c_void,
    // XRT stores non-owning buffer handles in a run. These leases keep every BO
    // and its context alive until completion or a synchronous abort.
    resources: Option<RunResources>,
}

// XRT run operations have no creator-thread affinity (the official queue API
// explicitly starts and waits on a run from a worker thread). Rust ownership keeps
// wait/cancel exclusive, and every referenced context/BO is retained in `resources`.
unsafe impl Send for RunHandle {}

struct RunResources {
    // Keep the dispatch context alive even for cross-context runs whose BOs all
    // originate from other contexts.
    _ctx: Arc<CtxInner>,
    _leases: Vec<RunLease>,
}

const RUN_RETAINED: i32 = -2;

impl RunHandle {
    /// Wait for completion. XRT timeouts are synchronously aborted by the shim
    /// before this method releases any BO lease.
    pub fn wait(mut self, timeout_ms: u32) -> Result<i32, String> {
        let state = unsafe { dv_run_wait(self.ptr, timeout_ms) };
        if state != RUN_RETAINED {
            self.ptr = std::ptr::null_mut();
        }
        if state < 0 {
            Err(ffi_error(format!("XRT run wait failed (state={state})")))
        } else {
            Ok(state)
        }
    }
}

impl Drop for RunHandle {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let state = unsafe { dv_run_cancel(self.ptr) };
        self.ptr = std::ptr::null_mut();
        if state == RUN_RETAINED {
            eprintln!(
                "dvxrt: XRT could not abort an abandoned run; intentionally leaking its context and BO leases"
            );
            if let Some(resources) = self.resources.take() {
                std::mem::forget(resources);
            }
        }
    }
}

/// ERT command state for a completed dispatch.
pub const COMPLETED: i32 = 4;

#[derive(Clone, Copy)]
pub enum BoKind {
    HostOnly = 0,
    Cacheable = 1,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Access {
    Read,
    Write,
}

struct ActiveAccess {
    id: u64,
    range: Range<usize>,
    access: Access,
}

struct AllocationState {
    next_id: AtomicU64,
    active: Mutex<Vec<ActiveAccess>>,
}

impl AllocationState {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            active: Mutex::new(Vec::with_capacity(4)),
        }
    }

    fn acquire(
        self: &Arc<Self>,
        range: Range<usize>,
        access: Access,
    ) -> Result<AccessLease, String> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| "BO access tracker is poisoned".to_string())?;
        let conflicts = active.iter().any(|other| {
            ranges_overlap(&range, &other.range)
                && (access == Access::Write || other.access == Access::Write)
        });
        if conflicts {
            return Err(format!(
                "BO range {}..{} is still in use by an overlapping device/host access",
                range.start, range.end
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        active.push(ActiveAccess { id, range, access });
        drop(active);
        Ok(AccessLease {
            allocation: Arc::clone(self),
            id,
        })
    }
}

struct AccessLease {
    allocation: Arc<AllocationState>,
    id: u64,
}

impl Drop for AccessLease {
    fn drop(&mut self) {
        if let Ok(mut active) = self.allocation.active.lock() {
            active.retain(|entry| entry.id != self.id);
        }
    }
}

struct RunLease {
    // Drop the access token before releasing the final BO wrapper.
    _access: AccessLease,
    _bo: Bo,
}

fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn checked_range(capacity: usize, offset: usize, len: usize) -> Result<Range<usize>, String> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("BO range offset={offset} len={len} overflows usize"))?;
    if end > capacity {
        return Err(format!(
            "BO range {offset}..{end} exceeds allocation size {capacity}"
        ));
    }
    Ok(offset..end)
}

/// Owns the raw `dv_ctx`. Freed only when the last `Arc` (the `Context` and all
/// `Bo`s it created) is dropped. XRT kernel handles are thread-safe; BO access
/// conflicts are additionally serialized by `AllocationState`.
struct CtxInner {
    ptr: *mut DvCtx,
}
impl Drop for CtxInner {
    fn drop(&mut self) {
        unsafe { dv_ctx_free(self.ptr) }
    }
}
unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

/// A device + xclbin + hw_context + `MLIR_AIE` kernel.
pub struct Context {
    inner: Arc<CtxInner>,
}

impl Context {
    /// Open a context for a trusted device image.
    /// # Safety
    /// Loading may start tile code. The image must be valid for this device and
    /// must not access host memory outside correctly bound dispatch arguments.
    pub unsafe fn new(dev_idx: i32, xclbin_path: &str) -> Result<Self, String> {
        initialize()?;
        let c = CString::new(xclbin_path).map_err(|e| e.to_string())?;
        let ptr = unsafe { dv_ctx_create(dev_idx, c.as_ptr()) };
        if ptr.is_null() {
            return Err(ffi_error(format!("dv_ctx_create failed for {xclbin_path}")));
        }
        Ok(Context {
            inner: Arc::new(CtxInner { ptr }),
        })
    }

    /// The memory bank for kernel argument `argidx` (used when allocating BOs).
    pub fn group_id(&self, argidx: i32) -> Result<i32, String> {
        let group = unsafe { dv_kernel_group_id(self.inner.ptr, argidx) };
        if group < 0 {
            Err(ffi_error(format!("XRT kernel group_id({argidx}) failed")))
        } else {
            Ok(group)
        }
    }

    /// Import host pages the caller owns as a BO, with no copy.
    ///
    /// The same pages can be imported into the GPU runtime at the same time, which is what
    /// makes a GPU/NPU handoff a cache flush rather than a copy through system memory.
    ///
    /// # Safety
    ///
    /// `userptr` must contain initialized bytes, be page-aligned, cover at least `nbytes`, and stay allocated and
    /// unmoved until the last clone of the returned `Bo` is dropped. The host must not
    /// touch those bytes while an NPU run using them is in flight.
    pub unsafe fn import_bo(
        &self,
        userptr: *mut c_void,
        nbytes: usize,
        kind: BoKind,
        group_id: i32,
    ) -> Result<Bo, String> {
        if nbytes == 0 {
            return Err("cannot import a zero-length XRT BO".into());
        }
        let ptr = unsafe { dv_bo_import(self.inner.ptr, userptr, nbytes, kind as c_int, group_id) };
        if ptr.is_null() {
            return Err(ffi_error(format!("dv_bo_import({nbytes}) failed")));
        }
        Ok(Bo {
            inner: Arc::new(BoInner {
                ptr,
                ctx: self.inner.clone(),
                allocation: Arc::new(AllocationState::new()),
                range: 0..nbytes,
                _parent: None,
            }),
        })
    }

    /// Import a dma-buf exported by another driver as a BO, with no copy.
    ///
    /// # Safety
    ///
    /// `fd` must name initialized bytes in a live dma-buf for at least `nbytes`, and the memory behind
    /// it must outlive the last clone of the returned `Bo`.
    pub unsafe fn import_dmabuf(&self, fd: i32, nbytes: usize) -> Result<Bo, String> {
        unsafe { self.import_dmabuf_region(fd, 0, nbytes) }
    }

    /// Import only the initialized, owned region of a potentially pooled dma-buf.
    /// # Safety
    /// The region must contain initialized bytes and its backing allocation must
    /// outlive every returned BO clone. All external access must be synchronized.
    pub(crate) unsafe fn import_dmabuf_region(
        &self,
        fd: i32,
        offset: usize,
        nbytes: usize,
    ) -> Result<Bo, String> {
        let extent = offset
            .checked_add(nbytes)
            .filter(|_| nbytes != 0)
            .ok_or("invalid dma-buf region")?;
        let ptr = unsafe { dv_bo_import_dmabuf(self.inner.ptr, fd, extent) };
        if ptr.is_null() {
            return Err(ffi_error(format!(
                "dv_bo_import_dmabuf(fd={fd}, {extent}) failed"
            )));
        }
        // The root can cover unrelated pooled bytes. It never escapes this
        // method: only the initialized region is exposed through the safe BO API.
        let root = Bo {
            inner: Arc::new(BoInner {
                ptr,
                ctx: self.inner.clone(),
                allocation: Arc::new(AllocationState::new()),
                range: 0..extent,
                _parent: None,
            }),
        };
        if offset == 0 {
            Ok(root)
        } else {
            root.sub(nbytes, offset)
        }
    }

    pub fn alloc_bo(&self, nbytes: usize, kind: BoKind, group_id: i32) -> Result<Bo, String> {
        if nbytes == 0 {
            return Err("cannot allocate a zero-length XRT BO".into());
        }
        let ptr = unsafe { dv_bo_alloc(self.inner.ptr, nbytes, kind as c_int, group_id) };
        if ptr.is_null() {
            return Err(ffi_error(format!("dv_bo_alloc({nbytes}) failed")));
        }
        Ok(Bo {
            inner: Arc::new(BoInner {
                ptr,
                ctx: self.inner.clone(),
                allocation: Arc::new(AllocationState::new()),
                range: 0..nbytes,
                _parent: None,
            }),
        })
    }

    /// Dispatch (opcode 3, insts, nbytes, A, B, C), wait, and synchronously
    /// abort on timeout before releasing any BO.
    ///
    /// # Safety
    ///
    /// The opaque instruction stream must be compiled for this xclbin and its
    /// DMA descriptors must stay within `a`, `b`, and `c`. XRT exposes no way to
    /// validate those semantic extents.
    pub unsafe fn run(
        &self,
        insts: &Bo,
        insts_nbytes: u32,
        a: &Bo,
        b: &Bo,
        c: &Bo,
        timeout_ms: u32,
    ) -> Result<i32, String> {
        self.run_start(insts, insts_nbytes, a, b, c)?
            .wait(timeout_ms)
    }

    /// ASYNC dispatch (CPU/NPU overlap): submit the kernel and return immediately with a `RunHandle`. The
    /// NPU runs while the caller does host work; call `handle.wait(timeout)` to block on completion. Inputs
    /// must be synced TO_DEVICE before this, and the output synced FROM_DEVICE after wait() — same as `run`.
    ///
    /// # Safety
    ///
    /// The opaque instruction stream must be compiled for this xclbin and its
    /// DMA descriptors must stay within `a`, `b`, and `c`. XRT exposes no way to
    /// validate those semantic extents.
    pub unsafe fn run_start(
        &self,
        insts: &Bo,
        insts_nbytes: u32,
        a: &Bo,
        b: &Bo,
        c: &Bo,
    ) -> Result<RunHandle, String> {
        if ![insts, a, b, c]
            .iter()
            .all(|bo| Arc::ptr_eq(&bo.inner.ctx, &self.inner))
        {
            return Err("BO used with a different XRT context".into());
        }
        self.run_start_inner(insts, insts_nbytes, a, b, c)
    }

    /// Like `run` but does NOT assert the BOs belong to this context — for the on-chip attention chain where a
    /// device BO is produced by one context (qproj/kvproj) and consumed by another (flash). XRT BOs are
    /// device-global; the hw_context selects the kernel, not the BO's memory (proven by run_chain_dk.py:
    /// a BO filled by the kvproj kernel is read by the flash kernel on a second context). The caller must
    /// ensure the BO's group_id/bank is compatible with this context's argument (HostOnly works across both).
    ///
    /// # Safety
    ///
    /// Every cross-context BO must refer to the same physical device and a bank
    /// compatible with this kernel argument. XRT does not validate that pairing.
    pub unsafe fn run_shared(
        &self,
        insts: &Bo,
        insts_nbytes: u32,
        a: &Bo,
        b: &Bo,
        c: &Bo,
        timeout_ms: u32,
    ) -> Result<i32, String> {
        self.run_start_inner(insts, insts_nbytes, a, b, c)?
            .wait(timeout_ms)
    }

    /// ASYNC + cross-context: `run_start` without the same-context BO assertion (= `run_shared` made async).
    /// The Stage-C foundation for overlapping the on-chip DK flash (q_bo/kv_bo are produced by qproj/kvproj on
    /// other contexts) behind host work. BO sharing is a BO property (HostOnly, device-global), not a dispatch
    /// property, so dv_run_start drives cross-context BOs the same as dv_run does for run_shared.
    ///
    /// # Safety
    ///
    /// Every cross-context BO must refer to the same physical device and a bank
    /// compatible with this kernel argument. XRT does not validate that pairing.
    pub unsafe fn run_start_shared(
        &self,
        insts: &Bo,
        insts_nbytes: u32,
        a: &Bo,
        b: &Bo,
        c: &Bo,
    ) -> Result<RunHandle, String> {
        self.run_start_inner(insts, insts_nbytes, a, b, c)
    }

    /// Dispatch the MLIR_AIE ABI with an arbitrary number of buffer arguments.
    /// # Safety
    /// The instructions, program and every argument must obey their DMA contract.
    pub unsafe fn run_args(&self, insts: &Bo, arguments: &[Bo]) -> Result<RunHandle, String> {
        if !insts.len().is_multiple_of(4)
            || insts.len() / 4 > u32::MAX as usize
            || arguments.len() > 64
        {
            return Err("invalid MLIR_AIE instruction length or argument count".into());
        }
        let mut pointers = [std::ptr::null_mut(); 64];
        let mut leases = Vec::with_capacity(arguments.len() + 1);
        for bo in std::iter::once(insts).chain(arguments.iter()) {
            leases.push(RunLease {
                _access: bo
                    .inner
                    .allocation
                    .acquire(bo.inner.range.clone(), Access::Write)?,
                _bo: bo.clone(),
            });
        }
        for (pointer, bo) in pointers.iter_mut().zip(arguments) {
            *pointer = bo.inner.ptr;
        }
        let ptr = unsafe {
            dv_run_start_args(
                self.inner.ptr,
                insts.inner.ptr,
                (insts.len() / 4) as u32,
                pointers.as_ptr(),
                arguments.len(),
            )
        };
        if ptr.is_null() {
            return Err(ffi_error("XRT submission failed"));
        }
        Ok(RunHandle {
            ptr,
            resources: Some(RunResources {
                _ctx: self.inner.clone(),
                _leases: leases,
            }),
        })
    }

    fn run_start_inner(
        &self,
        insts: &Bo,
        insts_nbytes: u32,
        a: &Bo,
        b: &Bo,
        c: &Bo,
    ) -> Result<RunHandle, String> {
        checked_range(insts.len(), 0, insts_nbytes as usize)?;

        let mut leases = Vec::with_capacity(4);
        for (bo, access) in [
            (insts, Access::Read),
            (a, Access::Read),
            (b, Access::Read),
            (c, Access::Write),
        ] {
            leases.push(RunLease {
                _access: bo
                    .inner
                    .allocation
                    .acquire(bo.inner.range.clone(), access)?,
                _bo: bo.clone(),
            });
        }

        let ptr = unsafe {
            dv_run_start(
                self.inner.ptr,
                insts.inner.ptr,
                insts_nbytes,
                a.inner.ptr,
                b.inner.ptr,
                c.inner.ptr,
            )
        };
        if ptr.is_null() {
            return Err(ffi_error("XRT run submission failed"));
        }
        Ok(RunHandle {
            ptr,
            resources: Some(RunResources {
                _ctx: self.inner.clone(),
                _leases: leases,
            }),
        })
    }
}

struct BoInner {
    ptr: *mut DvBo,
    ctx: Arc<CtxInner>,
    allocation: Arc<AllocationState>,
    // Byte range within the root allocation. Sub-BOs share `allocation`.
    range: Range<usize>,
    _parent: Option<Arc<BoInner>>,
}

unsafe impl Send for BoInner {}
unsafe impl Sync for BoInner {}

impl Drop for BoInner {
    fn drop(&mut self) {
        unsafe { dv_bo_free(self.ptr) }
        // `ctx` drops after the native BO wrapper.
    }
}

/// A cloneable device-buffer lease. The context and native BO wrapper remain
/// alive until the last clone and every in-flight run have been released.
#[derive(Clone)]
pub struct Bo {
    inner: Arc<BoInner>,
}

impl Bo {
    pub(crate) fn address(&self) -> Result<u64, String> {
        let address = unsafe { dv_bo_address(self.inner.ptr) };
        if address == 0 {
            Err("NPU buffer address unavailable".into())
        } else {
            Ok(address)
        }
    }

    /// Write host bytes into the BO and sync to device.
    pub fn write(&self, src: &[u8]) -> Result<(), String> {
        checked_range(self.len(), 0, src.len())?;
        let _access = self
            .inner
            .allocation
            .acquire(self.inner.range.clone(), Access::Write)?;
        let status =
            unsafe { dv_bo_write(self.inner.ptr, src.as_ptr().cast::<c_void>(), src.len()) };
        ffi_status("dv_bo_write", status)
    }

    /// Sync from device and read into the host buffer.
    pub fn read(&self, dst: &mut [u8]) -> Result<(), String> {
        checked_range(self.len(), 0, dst.len())?;
        let _access = self
            .inner
            .allocation
            .acquire(self.inner.range.clone(), Access::Write)?;
        let status =
            unsafe { dv_bo_read(self.inner.ptr, dst.as_mut_ptr().cast::<c_void>(), dst.len()) };
        ffi_status("dv_bo_read", status)
    }

    /// Zero-copy (unified memory): host pointer into the BO's shared RAM. Write/read
    /// it directly, then `sync` (cache flush) — no host<->device copy. The pointer is
    /// valid for the BO's lifetime; caller must respect `nbytes` and sync direction.
    pub fn map(&self) -> Result<*mut u8, String> {
        let ptr = unsafe { dv_bo_map(self.inner.ptr).cast::<u8>() };
        if ptr.is_null() {
            Err(ffi_error("dv_bo_map failed"))
        } else {
            Ok(ptr)
        }
    }
    /// Sub-buffer VIEW: an offset+size window into this BO (no copy). For offset-dispatch — pack A once into a
    /// big resident BO, then dispatch the M-baked inst per chunk via `parent.sub(chunk_bytes, c*chunk_bytes)`.
    /// The returned Bo shares this BO's context; keep the parent alive while sub-BOs are in use.
    pub fn sub(&self, size: usize, offset: usize) -> Result<Bo, String> {
        if size == 0 {
            return Err("cannot create a zero-length XRT sub-BO".into());
        }
        let local = checked_range(self.len(), offset, size)?;
        let absolute = self
            .inner
            .range
            .start
            .checked_add(local.start)
            .and_then(|start| start.checked_add(size).map(|end| start..end))
            .ok_or("sub-BO absolute range overflows usize")?;
        let ptr = unsafe { dv_bo_suballoc(self.inner.ptr, size, offset) };
        if ptr.is_null() {
            return Err(ffi_error(format!(
                "dv_bo_suballoc(size={size}, off={offset}) failed"
            )));
        }
        Ok(Bo {
            inner: Arc::new(BoInner {
                ptr,
                ctx: self.inner.ctx.clone(),
                allocation: self.inner.allocation.clone(),
                range: absolute,
                _parent: Some(self.inner.clone()),
            }),
        })
    }

    pub fn sync(&self, to_device: bool, nbytes: usize) -> Result<(), String> {
        checked_range(self.len(), 0, nbytes)?;
        let _access = self
            .inner
            .allocation
            .acquire(self.inner.range.clone(), Access::Write)?;
        let status = unsafe { dv_bo_sync(self.inner.ptr, i32::from(to_device), nbytes) };
        ffi_status("dv_bo_sync", status)
    }

    pub fn len(&self) -> usize {
        self.inner.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn ffi_error(message: impl std::fmt::Display) -> String {
    let detail = unsafe { std::ffi::CStr::from_ptr(dv_last_error()) }.to_string_lossy();
    format!("{message}: {detail}")
}

fn ffi_status(operation: &str, status: i32) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(ffi_error(format!("{operation} failed (status={status})")))
    }
}

/// Prepared native run; only the tracked scheduler can execute it.
pub(crate) struct PreparedRun {
    pointer: *mut c_void,
    resources: Option<(Arc<CtxInner>, Bo, Vec<Bo>)>,
}
// Access is exclusive through the backend mutex; native XRT runs have no
// creator-thread affinity. Bound arguments remain retained for its lifetime.
unsafe impl Send for PreparedRun {}
impl PreparedRun {
    pub(crate) unsafe fn new(
        context: &Context,
        instructions: &Bo,
        arguments: Vec<Bo>,
    ) -> Result<Self, String> {
        if arguments.len() > 64
            || !instructions.len().is_multiple_of(4)
            || instructions.len() / 4 > u32::MAX as usize
        {
            return Err("invalid prepared NPU ABI".into());
        }
        let mut pointers = [std::ptr::null_mut(); 64];
        for (pointer, bo) in pointers.iter_mut().zip(&arguments) {
            *pointer = bo.inner.ptr;
        }
        let pointer = unsafe {
            dv_run_prepare(
                context.inner.ptr,
                instructions.inner.ptr,
                (instructions.len() / 4) as u32,
                pointers.as_ptr(),
                arguments.len(),
            )
        };
        if pointer.is_null() {
            return Err(ffi_error("preparing XRT run failed"));
        }
        Ok(Self {
            pointer,
            resources: Some((context.inner.clone(), instructions.clone(), arguments)),
        })
    }
    pub(crate) fn execute(&mut self) -> crate::Result<()> {
        let status = unsafe { dv_prepared_execute(self.pointer, 60_000) };
        if status == COMPLETED {
            Ok(())
        } else {
            Err(crate::Error::Backend {
                backend: "XRT",
                operation: "prepared dispatch",
                code: status,
                message: unsafe { std::ffi::CStr::from_ptr(dv_last_error()) }
                    .to_string_lossy()
                    .into_owned()
                    .into_boxed_str(),
            })
        }
    }
}
impl Drop for PreparedRun {
    fn drop(&mut self) {
        if unsafe { dv_prepared_free(self.pointer) } == RUN_RETAINED
            && let Some(resources) = self.resources.take()
        {
            std::mem::forget(resources);
        }
    }
}
