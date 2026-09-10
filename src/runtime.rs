use std::ffi::{CStr, CString, c_void};
use std::path::Path;

/// Native export metadata used to validate dispatch shapes and arguments.
pub use crate::sys::ExportInfo;
use crate::{Error, Result, TARGET_FAMILY, Target, sys};

/// Turns a status into a `Result`, taking ownership of its message. A null status is success.
pub(crate) fn check(status: sys::Status, what: impl std::fmt::Display) -> Result<()> {
    if sys::is_ok(status) {
        return Ok(());
    }
    let code = unsafe { sys::hrx_status_code(status) };
    let mut message: *mut std::ffi::c_char = std::ptr::null_mut();
    let mut length: usize = 0;
    let text = unsafe {
        let to_string = sys::hrx_status_to_string(status, &mut message, &mut length);
        let text = if sys::is_ok(to_string) && !message.is_null() {
            let owned = CStr::from_ptr(message).to_string_lossy().into_owned();
            sys::hrx_status_free_message(message);
            owned
        } else {
            sys::hrx_status_ignore(to_string);
            "unknown error".to_string()
        };
        sys::hrx_status_ignore(status);
        text
    };
    Err(Error::Runtime {
        context: what.to_string(),
        code,
        message: text,
    })
}

/// Shared ownership of a stream and its borrowed device registry entry.
/// Buffers, kernels and events retain this handle through their native uses.
struct Inner {
    device: sys::Device,
    target: Target,
    stream: sys::Stream,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // hrx_stream_create returns an owned reference. hrx_gpu_device_get
        // borrows the global registry entry without retaining it; do not release it.
        unsafe { sys::hrx_stream_release(self.stream) };
    }
}

// Safety: native handle reference counts are atomic. Stream mutation requires
// exclusive Stream access. Shared events query or wait on their semaphore;
// they do not mutate the stream.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// Initialize the process-wide runtime once, serializing access to native global
/// state. ALREADY_EXISTS permits reuse across model libraries; failures are retryable.
pub(crate) fn initialize_runtime() -> Result<()> {
    sys::load()?;
    static READY: std::sync::Mutex<Option<()>> = std::sync::Mutex::new(None);
    crate::cached_init(&READY, || {
        let _lock = sys::runtime_lock()?;
        unsafe {
            let status = sys::hrx_gpu_initialize(0);
            if sys::hrx_status_code(status) == sys::STATUS_ALREADY_EXISTS {
                sys::hrx_status_ignore(status);
            } else {
                check(status, "hrx_gpu_initialize")?;
            }
        }
        Ok(())
    })
}

pub(crate) fn open_device(index: i32) -> Result<(sys::Device, Target)> {
    initialize_runtime()?;
    unsafe {
        let mut count = 0;
        check(
            sys::hrx_gpu_device_count(&mut count),
            "hrx_gpu_device_count",
        )?;
        if index < 0 || index >= count {
            return Err(Error::Message(
                "no GPU device (check GPU access and the README provisioning instructions)".into(),
            ));
        }
        let mut device = std::ptr::null_mut();
        check(
            sys::hrx_gpu_device_get(index, &mut device),
            "hrx_gpu_device_get",
        )?;
        let mut architecture = [0u8; 64];
        check(
            sys::hrx_device_get_property(
                device,
                sys::DEVICE_PROPERTY_ARCHITECTURE,
                architecture.as_mut_ptr().cast(),
                architecture.len(),
            ),
            "device architecture",
        )?;
        let key = std::str::from_utf8(architecture.split(|b| *b == 0).next().unwrap())
            .map_err(|_| Error::Message("invalid device architecture".into()))?;
        Ok((device, Target::from_device_architecture(key)?))
    }
}

impl Stream {
    fn new(index: i32) -> Result<Self> {
        let (device, target) = open_device(index)?;
        unsafe {
            let mut stream = std::ptr::null_mut();
            check(
                sys::hrx_stream_create(device, 0, &mut stream),
                "hrx_stream_create",
            )?;
            Ok(Self {
                inner: std::sync::Arc::new(Inner {
                    device,
                    target,
                    stream,
                }),
                _not_sync: std::marker::PhantomData,
                staging: Vec::new(),
                staging_pool: Vec::new(),
                scratch: std::collections::BTreeMap::new(),
                scratch_bytes: 0,
                scratch_limit: 256 * 1024 * 1024,
            })
        }
    }

    fn owns(&self, buffer: &Buffer) -> Result<()> {
        owns(&self.inner, buffer)
    }

    fn synchronize_native(&self) -> Result<()> {
        unsafe {
            check(
                sys::hrx_stream_synchronize(self.inner.stream),
                "hrx_stream_synchronize",
            )
        }
    }

    unsafe fn loaded_export(&self, executable: sys::Executable, symbol: &str) -> Result<Kernel> {
        let kernel = (|| {
            let symbol_c =
                CString::new(symbol).map_err(|_| Error::Message("export contains a NUL".into()))?;
            let mut ordinal = 0;
            let mut info = sys::ExportInfo::default();
            unsafe {
                check(
                    sys::hrx_executable_lookup_export_by_name(
                        executable,
                        symbol_c.as_ptr(),
                        &mut ordinal,
                    ),
                    "looking up export",
                )?;
                check(
                    sys::hrx_executable_export_info(executable, ordinal, &mut info),
                    "export metadata",
                )?;
            }
            Ok(Kernel {
                executable: std::sync::Arc::new(Executable {
                    raw: executable,
                    device: self.inner.clone(),
                }),
                ordinal,
                info,
                symbol: symbol.into(),
            })
        })();
        if kernel.is_err() {
            unsafe { sys::hrx_executable_release(executable) }
        }
        kernel
    }
}

/// An owned device allocation. Use [`Buffer::binding`] or [`Buffer::try_slice`]
/// to borrow a binding; this API does not expose host pointers.
/// A handle is bound to its device. Any stream on that device may use it;
/// ordering conflicting access is the caller's, with events.
pub struct Buffer {
    raw: sys::Buffer,
    bytes: usize,
    /// Keeps the device alive: releasing a buffer after its device is gone would be a use-after-free.
    _device: std::sync::Arc<Inner>,
}

// Safety: moving/releasing an allocation uses native atomic reference counts.
// Every operation checks its owning Inner before touching native buffer state.
// Streams are !Sync, and a buffer used from two streams needs events between
// conflicting accesses; unordered use yields stale bytes, never invalid memory.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A binding into a device allocation, borrowed from it.
///
/// The borrow keeps the buffer alive and identifies the stream allowed to use
/// this view. Any stream on the same device may use it, ordered by the caller.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    raw: sys::BufferRef,
    owner: &'a Buffer,
}

impl<'a> View<'a> {
    fn new(raw: sys::BufferRef, owner: &'a Buffer) -> Self {
        Self { raw, owner }
    }
    /// Borrow a region relative to this view, rejecting overflow and overruns.
    /// An empty region at the end of the view is valid.
    pub fn slice(self, offset: usize, length: usize) -> Result<Self> {
        checked_span(offset, length, self.len())?;
        Ok(Self::new(
            sys::BufferRef {
                offset: self.raw.offset + offset,
                length,
                ..self.raw
            },
            self.owner,
        ))
    }
    /// Byte offset from the start of the owning allocation.
    #[must_use]
    pub fn offset(self) -> usize {
        self.raw.offset
    }
    /// The buffer handle from which this view was borrowed.
    #[must_use]
    pub fn owner(self) -> &'a Buffer {
        self.owner
    }
    /// The bytes this view covers.
    #[must_use]
    pub fn len(self) -> usize {
        self.raw.length
    }
    #[must_use]
    /// Whether this view covers no bytes.
    pub fn is_empty(self) -> bool {
        self.raw.length == 0
    }
}

impl Buffer {
    /// The actual allocation size, including rounding of empty allocations.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Borrow a span, rejecting overflow and allocation overruns.
    pub fn try_slice(&self, offset: usize, length: usize) -> Result<View<'_>> {
        self.binding().slice(offset, length)
    }

    /// The device address backing this buffer.
    ///
    /// Needed to hand the allocation to another driver -- exporting it as a dma-buf for the
    /// NPU, say -- rather than copying its contents out.
    pub fn device_ptr(&self) -> Result<*mut std::ffi::c_void> {
        let mut pointer = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_buffer_get_device_ptr(self.raw, &mut pointer),
                "hrx_buffer_get_device_ptr",
            )?;
        }
        Ok(pointer)
    }

    /// The whole allocation.
    pub(crate) fn allocation_address(&self) -> Result<u64> {
        type Address = unsafe extern "C" fn(sys::Buffer, *mut u64) -> sys::Status;
        let address: Address = unsafe { sys::interop_symbol(b"hrx_buffer_allocation_address\0") }?;
        let mut value = 0;
        unsafe {
            check(
                address(self.raw, &mut value),
                "querying GPU allocation address",
            )?;
        }
        Ok(value)
    }

    #[cfg(feature = "npu")]
    pub(crate) fn export_dmabuf(&self) -> Result<(std::os::fd::OwnedFd, u64)> {
        use std::os::fd::FromRawFd;
        type Export = unsafe extern "C" fn(sys::Buffer, *mut i32, *mut u64) -> sys::Status;
        let export: Export = unsafe { sys::interop_symbol(b"hrx_buffer_export_dmabuf\0") }?;
        let mut descriptor = -1;
        let mut offset = 0;
        unsafe {
            check(
                export(self.raw, &mut descriptor, &mut offset),
                "exporting GPU allocation",
            )?;
        }
        if descriptor < 0 {
            return Err(Error::Message(
                "native export returned invalid descriptor".into(),
            ));
        }
        Ok((
            unsafe { std::os::fd::OwnedFd::from_raw_fd(descriptor) },
            offset,
        ))
    }

    /// Borrow the whole allocation as a device binding.
    pub fn binding(&self) -> View<'_> {
        View::new(
            sys::BufferRef {
                buffer: self.raw,
                offset: 0,
                length: self.bytes,
            },
            self,
        )
    }

    /// Borrow a span within the allocation.
    ///
    /// # Panics
    /// Panics if offset arithmetic overflows or the span exceeds the allocation.
    /// Use [`Buffer::try_slice`] to return an error instead.
    pub fn slice(&self, offset: usize, length: usize) -> View<'_> {
        self.try_slice(offset, length)
            .expect("slice past the allocation or overflow")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { sys::hrx_buffer_release(self.raw) }
    }
}

/// A loaded executable export with immutable dispatch metadata.
/// Kernels can be dispatched on any stream on their device; buffer handles belong
/// to individual streams.
pub struct Kernel {
    /// Shared by every clone, so cloning is two atomics rather than a call
    /// across the FFI boundary. A cache that hands a kernel out per dispatch
    /// clones it on every launch, so `Kernel: Clone` is only worth advertising
    /// over `Arc<Kernel>` if it is no more expensive.
    executable: std::sync::Arc<Executable>,
    ordinal: u32,
    info: sys::ExportInfo,
    symbol: std::sync::Arc<str>,
}

/// The native executable, released once the last kernel naming it is dropped.
struct Executable {
    raw: sys::Executable,
    /// As for a buffer: the executable names its device.
    device: std::sync::Arc<Inner>,
}

// Safety: the same assertion `Kernel` carries, moved to the field that actually
// holds the native handle. Executable metadata is immutable and the native
// reference count is atomic, so sharing one across threads is sound; dispatch
// still requires serialized access to a command stream.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        unsafe { sys::hrx_executable_release(self.raw) }
    }
}

// Safety: executable metadata is immutable and native reference counts are
// atomic. Dispatch requires serialized access to a command stream.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Native export metadata, including argument counts and workgroup dimensions.
    pub fn info(&self) -> &ExportInfo {
        &self.info
    }
    /// The loaded export name.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Cloning shares the native executable, so a kernel can be cached and handed
/// out by value instead of behind an `Arc`. Export metadata is immutable, and
/// the borrowed `ExportInfo::name` stays valid while any clone is alive.
impl Clone for Kernel {
    fn clone(&self) -> Self {
        Self {
            executable: self.executable.clone(),
            ordinal: self.ordinal,
            info: self.info,
            symbol: self.symbol.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn compiled_workgroup_dimensions_are_enforced() {
        let info = super::ExportInfo {
            workgroup_size: [64, 2, 1],
            ..Default::default()
        };
        assert!(super::validate_export_launch(&info, [1; 3], [64, 2, 1]).is_ok());
        assert!(super::validate_export_launch(&info, [1; 3], [128, 1, 1]).is_err());
        assert!(super::validate_export_launch(&info, [0, 1, 1], [64, 2, 1]).is_err());
        let dynamic = super::ExportInfo {
            workgroup_size: [0, 2, 1],
            ..Default::default()
        };
        assert!(super::validate_export_launch(&dynamic, [1; 3], [32, 2, 1]).is_ok());
        assert!(super::validate_export_launch(&dynamic, [1; 3], [32, 1, 1]).is_err());
    }
    /// The bounds arithmetic, without a device: `offset + length` must not wrap into a pass.
    #[test]
    fn a_slice_past_the_allocation_is_refused_even_when_the_sum_wraps() {
        let checked = |offset, length, bytes| super::checked_span(offset, length, bytes).is_ok();
        assert!(checked(0, 8, 8));
        assert!(checked(4, 4, 8));
        assert!(!checked(4, 5, 8));
        // the wrapping case: 8 bytes must not accept an offset near the top of the address space
        assert!(!checked(usize::MAX, 2, 8));
        assert!(!checked(usize::MAX - 1, 4, 8));
    }
}

/// Buffers are bound to their device, not to the stream that allocated them.
///
/// The allocator is asked for `queue_affinity: u64::MAX` and executables are
/// already device-scoped, so a stricter check here would be conservatism rather
/// than a constraint. Ordering conflicting access across streams is the caller's
/// job, and [`Stream::record_event`] / [`Stream::wait_event`] are the tools for
/// it: a buffer written by one stream and read by another with no event between
/// them yields whichever bytes the device happened to hold.
fn owns(inner: &std::sync::Arc<Inner>, buffer: &Buffer) -> Result<()> {
    if inner.device == buffer._device.device {
        Ok(())
    } else {
        Err(Error::Message("buffer belongs to another device".into()))
    }
}

// Keep common dispatches on the stack without imposing a new binding limit.
fn raw_bindings<'a>(
    views: &[View<'_>],
    stack: &'a mut [std::mem::MaybeUninit<sys::BufferRef>; 32],
) -> std::borrow::Cow<'a, [sys::BufferRef]> {
    if views.len() <= stack.len() {
        for (out, view) in stack.iter_mut().zip(views) {
            out.write(view.raw);
        }
        // Only the prefix written above is exposed, and BufferRef is Copy.
        std::borrow::Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(stack.as_ptr().cast(), views.len())
        })
    } else {
        std::borrow::Cow::Owned(views.iter().map(|v| v.raw).collect())
    }
}
pub(crate) fn checked_span(offset: usize, length: usize, capacity: usize) -> Result<()> {
    if offset.checked_add(length).is_none_or(|end| end > capacity) {
        Err(Error::Message(format!(
            "span {offset}+{length} exceeds {capacity} bytes"
        )))
    } else {
        Ok(())
    }
}
pub(crate) fn validate_launch(grid: [u32; 3], block: [u32; 3]) -> Result<()> {
    if grid.contains(&0)
        || block.contains(&0)
        || block
            .iter()
            .try_fold(1u32, |n, b| n.checked_mul(*b))
            .is_none_or(|n| n > 1024)
    {
        Err(Error::Message("invalid GPU launch dimensions".into()))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_export_launch(
    info: &sys::ExportInfo,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<()> {
    validate_launch(grid, block)?;
    if info
        .workgroup_size
        .iter()
        .zip(block)
        .any(|(&expected, actual)| expected != 0 && expected != actual)
    {
        return Err(Error::Message(format!(
            "workgroup size {block:?} disagrees with compiled size {:?}",
            info.workgroup_size
        )));
    }
    Ok(())
}

/// A borrowed device registry entry. Opening a stream never acquires ownership
/// of the native device, and dropping a model never shuts down HRX globally.
#[derive(Clone, Debug)]
pub struct Device {
    index: i32,
    target: Target,
}
impl Device {
    /// Whether this native library exposes shared-allocation interop ABI 1.
    /// This checks the native API, not whether a particular NPU driver can import.
    pub fn supports_shared_interop(&self) -> Result<bool> {
        type Abi = unsafe extern "C" fn() -> u32;
        match unsafe { sys::interop_symbol::<Abi>(b"hrx_interop_abi_version\0") } {
            Ok(abi) => Ok(unsafe { abi() } == 1),
            Err(Error::Unsupported(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Validate a device index without creating a native stream.
    pub fn open(index: i32) -> Result<Self> {
        let (_, target) = open_device(index)?;
        Ok(Self { index, target })
    }
    /// Architecture reported by the selected device.
    pub fn target(&self) -> &Target {
        &self.target
    }
    /// Create an independent ordered stream on this device.
    pub fn stream(&self) -> Result<Stream> {
        Stream::new(self.index)
    }
}

/// An ordered command stream. Mutating operations require exclusive access.
/// Prepared kernels and allocations can be reused without a compiler/cache lookup.
pub struct Stream {
    inner: std::sync::Arc<Inner>,
    // Native stream fields are unsynchronized. Stream is Send, but not Sync.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
    // Native command buffers retain HAL storage, not the hrx_buffer wrapper.
    // Releasing a mapped wrapper unmaps it (buffer.c::hrx_buffer_release).
    // Upload staging must therefore retain its wrapper through completion.
    // Readbacks are unmapped until wait; ordinary/scratch buffers are never
    // mapped by this API, so native retention suffices for their early release.
    staging: Vec<Buffer>,
    staging_pool: Vec<Buffer>,
    scratch: std::collections::BTreeMap<usize, Vec<Buffer>>,
    scratch_bytes: usize,
    scratch_limit: usize,
}
impl Stream {
    /// Architecture reported by this stream's device.
    pub fn target(&self) -> &Target {
        &self.inner.target
    }

    /// Submit preceding work and record a single immutable completion event.
    /// Recording does not wait on the host and does not reclaim upload staging.
    pub fn record_event(&mut self) -> Result<Event> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_event_create(self.inner.device, sys::EVENT_FLAG_DISABLE_TIMING, &mut raw),
                "create event",
            )?;
            let event = Event {
                raw,
                inner: self.inner.clone(),
            };
            check(
                sys::hrx_event_record(raw, self.inner.stream),
                "record event",
            )?;
            Ok(event)
        }
    }

    /// Queue a device-side dependency before subsequent work on this stream.
    /// The event must come from the same device. This call does not wait on the host.
    pub fn wait_event(&mut self, event: &Event) -> Result<()> {
        if self.inner.device != event.inner.device {
            return Err(Error::Message("event belongs to another device".into()));
        }
        unsafe {
            check(
                sys::hrx_stream_wait_event(self.inner.stream, event.raw),
                "wait event",
            )
        }
    }
    /// Open an ordered stream on device zero.
    pub fn open() -> Result<Self> {
        Device::open(0)?.stream()
    }
    /// Identifies the device this stream runs on, for callers that keep
    /// per-device state. Kernels are device-scoped, so a cache of loaded
    /// executables is only valid for the device that loaded them.
    pub fn device_id(&self) -> usize {
        self.inner.device as usize
    }
    /// Identifies this stream, for callers that keep per-stream state.
    ///
    /// Buffers are device-scoped, so nothing here rejects one used on a sibling
    /// stream — correct, because events can order that. What events cannot fix
    /// is *reuse*: a pool handing a buffer out again relies on the queue that
    /// used it last running in order, which holds within a stream and not
    /// across them. A caller pooling allocations needs to say which stream a
    /// block came from, and this is how. Unique among live streams; a value may
    /// repeat once its stream is dropped.
    pub fn id(&self) -> usize {
        std::sync::Arc::as_ptr(&self.inner) as usize
    }
    /// Allocate storage owned by this stream; zero bytes is rounded to one.
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        let mut buffer = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_allocate_buffer(
                    sys::hrx_device_allocator(self.inner.device),
                    sys::BufferParams {
                        memory_type: sys::MEMORY_TYPE_DEVICE_LOCAL,
                        access: sys::MEMORY_ACCESS_ALL,
                        usage: sys::BUFFER_USAGE_DEFAULT,
                        queue_affinity: u64::MAX,
                    },
                    bytes.max(1),
                    &mut buffer,
                ),
                "hrx_buffer_allocate",
            )?;
        }
        Ok(Buffer {
            raw: buffer,
            bytes: bytes.max(1),
            _device: self.inner.clone(),
        })
    }
    /// Import host memory the caller owns as a device-visible buffer, without copying.
    ///
    /// On an APU the GPU and the NPU address the same physical pages, so importing one
    /// host allocation into both runtimes is the basis of zero-copy handoff between them:
    /// the same pages can simultaneously back an XRT BO driving the NPU.
    ///
    /// # Safety
    ///
    /// `pointer` must be page-aligned, cover at least `bytes`, and stay allocated and
    /// unmoved for the whole life of the returned buffer -- the buffer borrows the memory
    /// and never frees it. The host must not read or write those bytes while device work
    /// touching them is in flight.
    pub unsafe fn import_host(
        &self,
        pointer: *mut std::ffi::c_void,
        bytes: usize,
    ) -> Result<Buffer> {
        let mut buffer = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_import_buffer(
                    sys::hrx_device_allocator(self.inner.device),
                    sys::BufferParams {
                        // Host-resident pages the device reads in place, rather than a
                        // device-local allocation the runtime would have to copy into.
                        memory_type: sys::MEMORY_TYPE_HOST_LOCAL
                            | sys::MEMORY_TYPE_HOST_COHERENT
                            | sys::MEMORY_TYPE_DEVICE_VISIBLE,
                        access: sys::MEMORY_ACCESS_ALL,
                        usage: sys::BUFFER_USAGE_DEFAULT,
                        queue_affinity: u64::MAX,
                    },
                    pointer,
                    bytes,
                    &mut buffer,
                ),
                "hrx_allocator_import_buffer",
            )?;
        }
        Ok(Buffer {
            raw: buffer,
            bytes,
            _device: self.inner.clone(),
        })
    }

    /// Submit and wait for all work, then reclaim completed upload staging.
    pub fn synchronize(&mut self) -> Result<()> {
        self.synchronize_native()?;
        self.reclaim_staging();
        Ok(())
    }
    /// Drain pending work, then upload bytes at the start of a view and wait for completion.
    /// The input must fit within the view. Completed staging is reclaimed.
    pub fn upload_blocking(&mut self, dst: View<'_>, bytes: &[u8]) -> Result<()> {
        self.owns(dst.owner)?;
        checked_span(0, bytes.len(), dst.len())?;
        self.synchronize()?;
        if bytes.is_empty() {
            return Ok(());
        }
        unsafe {
            check(
                sys::hrx_synchronous_h2d(
                    self.inner.device,
                    bytes.as_ptr().cast(),
                    dst.raw.buffer,
                    dst.raw.offset,
                    bytes.len(),
                ),
                "hrx_synchronous_h2d",
            )
        }
    }
    /// Drain all pending work before a synchronous read and reclaim staging.
    pub fn read_blocking(&mut self, src: View<'_>, bytes: &mut [u8]) -> Result<()> {
        self.owns(src.owner)?;
        checked_span(0, bytes.len(), src.len())?;
        self.synchronize()?;
        if bytes.is_empty() {
            return Ok(());
        }
        unsafe {
            check(
                sys::hrx_synchronous_d2h(
                    self.inner.device,
                    src.raw.buffer,
                    src.raw.offset,
                    bytes.as_mut_ptr().cast(),
                    bytes.len(),
                ),
                "hrx_synchronous_d2h",
            )
        }
    }
    /// Queue a byte-pattern fill over a nonempty view owned by this stream.
    pub fn fill(&self, dst: View<'_>, value: u8) -> Result<()> {
        self.owns(dst.owner)?;
        if dst.is_empty() {
            return Err(Error::Message("empty stream fill".into()));
        }
        unsafe {
            check(
                sys::hrx_stream_fill_buffer(
                    self.inner.stream,
                    dst.raw.buffer,
                    dst.raw.offset,
                    dst.len(),
                    &value as *const u8 as *const c_void,
                    1,
                ),
                "hrx_stream_fill_buffer",
            )
        }
    }
    /// Queue a copy between equal, nonempty views owned by this stream.
    pub fn copy(&self, dst: View<'_>, src: View<'_>) -> Result<()> {
        self.owns(dst.owner)?;
        self.owns(src.owner)?;
        if dst.len() != src.len() || dst.is_empty() {
            return Err(Error::Message(
                "stream copy requires equal nonempty spans".into(),
            ));
        }
        unsafe {
            check(
                sys::hrx_stream_copy_buffer(
                    self.inner.stream,
                    src.raw.buffer,
                    src.raw.offset,
                    dst.raw.buffer,
                    dst.raw.offset,
                    src.len(),
                ),
                "hrx_stream_copy_buffer",
            )
        }
    }
    /// Copy the input into runtime-owned staging, then enqueue an actual GPU
    /// buffer copy. The caller's slice is no longer referenced when this returns.
    /// Bytes are written at the start of the view and must fit within it.
    /// Staging pressure may submit or wait for prior work to bound memory use.
    pub fn upload(&mut self, dst: View<'_>, bytes: &[u8]) -> Result<()> {
        self.owns(dst.owner)?;
        checked_span(0, bytes.len(), dst.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        // Submit/query only under pressure, allowing uploads to batch.
        let next_capacity = self
            .staging_pool
            .iter()
            .filter(|b| b.bytes >= bytes.len())
            .map(|b| b.bytes)
            .min()
            .unwrap_or(bytes.len());
        if !self.staging.is_empty()
            && (self.staging.len() >= 8
                || self
                    .staging
                    .iter()
                    .map(|b| b.bytes)
                    .sum::<usize>()
                    .saturating_add(next_capacity)
                    > STAGING_LIMIT)
            && !self.submit()?.is_complete()?
        {
            self.synchronize()?;
        }
        let staging = if let Some(i) = self
            .staging_pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bytes >= bytes.len())
            .min_by_key(|(_, b)| b.bytes)
            .map(|(i, _)| i)
        {
            self.staging_pool.swap_remove(i)
        } else {
            self.allocate_host(bytes.len())?
        };
        let raw = staging.raw;
        unsafe {
            let mut pointer = std::ptr::null_mut();
            check(
                sys::hrx_buffer_get_device_ptr(raw, &mut pointer),
                "map staging",
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.cast::<u8>(), bytes.len());
            // Retain before enqueue: even a partially recorded command on failure
            // must not outlive the wrapper or its mapping.
            self.staging.push(staging);
            check(
                sys::hrx_stream_copy_buffer(
                    self.inner.stream,
                    raw,
                    0,
                    dst.raw.buffer,
                    dst.raw.offset,
                    bytes.len(),
                ),
                "enqueue upload",
            )
        }
    }
    /// Allocate a host-local, device-visible buffer.
    ///
    /// Unlike [`Stream::allocate`], the runtime hands out a device pointer for these, so
    /// the allocation can be exported to another driver -- which is what lets one buffer
    /// serve both the GPU and the NPU. Device-local memory is faster for GPU-only work.
    pub fn allocate_shared(&self, bytes: usize) -> Result<Buffer> {
        self.allocate_host(bytes)
    }

    fn allocate_host(&self, bytes: usize) -> Result<Buffer> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_allocate_buffer(
                    sys::hrx_device_allocator(self.inner.device),
                    sys::BufferParams {
                        memory_type: sys::MEMORY_TYPE_HOST_LOCAL
                            | sys::MEMORY_TYPE_DEVICE_VISIBLE
                            | sys::MEMORY_TYPE_HOST_COHERENT,
                        access: sys::MEMORY_ACCESS_ALL,
                        usage: sys::BUFFER_USAGE_DEFAULT | sys::BUFFER_USAGE_MAPPING_SCOPED,
                        queue_affinity: u64::MAX,
                    },
                    bytes.max(1),
                    &mut raw,
                ),
                "allocate staging",
            )?;
        }
        Ok(Buffer {
            raw,
            bytes,
            _device: self.inner.clone(),
        })
    }
    fn reclaim_staging(&mut self) {
        let mut cached = self.staging_pool.iter().map(|b| b.bytes).sum::<usize>();
        for buffer in self.staging.drain(..) {
            if self.staging_pool.len() < 8 && buffer.bytes <= STAGING_LIMIT.saturating_sub(cached) {
                cached += buffer.bytes;
                self.staging_pool.push(buffer);
            }
        }
    }
    /// Submit pending commands before querying completion; native query alone
    /// does not include the unsubmitted command buffer.
    pub fn submit(&mut self) -> Result<Submission<'_>> {
        unsafe {
            check(sys::hrx_stream_flush(self.inner.stream), "submit")?;
        }
        Ok(Submission { stream: self })
    }
    /// Load a native code object and select one export by name.
    ///
    /// # Safety
    /// The code object must be trusted native machine code.
    pub unsafe fn load(&self, path: &Path, symbol: &str) -> Result<Kernel> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::Message(format!("{} contains a NUL", path.display())))?;
        let c_family = TARGET_FAMILY;
        let c_key = self.inner.target.as_c_str();
        unsafe {
            let mut executable = std::ptr::null_mut();
            check(
                sys::hrx_executable_load_file(
                    self.inner.device,
                    c_path.as_ptr(),
                    c_family.as_ptr(),
                    c_key.as_ptr(),
                    &mut executable,
                ),
                format_args!("loading {}", path.display()),
            )?;
            self.loaded_export(executable, symbol)
        }
    }
    /// Load a compiled artifact directly, without a filesystem round trip.
    ///
    /// # Safety
    /// The artifact must be trusted native code, as for [`Stream::load`].
    #[cfg(feature = "loom")]
    pub unsafe fn load_artifact(&self, artifact: &crate::loom::Artifact) -> Result<Kernel> {
        if artifact.target() != self.inner.target.as_str() {
            return Err(Error::Message(
                "artifact target does not match this runtime".into(),
            ));
        }
        let bytes = artifact.bytes();
        let mut executable = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_executable_load_data(
                    self.inner.device,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    TARGET_FAMILY.as_ptr(),
                    self.inner.target.as_c_str().as_ptr(),
                    &mut executable,
                ),
                "loading compiled artifact",
            )?;
            self.loaded_export(executable, artifact.symbol())
        }
    }

    /// Queue a kernel invocation with explicitly packed constants and borrowed bindings.
    ///
    /// # Safety
    /// Kernel, dimensions, constants and binding spans must agree. GPU addressing
    /// is not sandboxed by a binding's length.
    pub unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'_>],
    ) -> Result<()> {
        if kernel.executable.device.device != self.inner.device {
            return Err(Error::Message("kernel belongs to another device".into()));
        }
        for view in bindings {
            owns(&self.inner, view.owner)?;
        }
        let mut binding_storage = [std::mem::MaybeUninit::uninit(); 32];
        let raw_bindings = raw_bindings(bindings, &mut binding_storage);
        validate_export_launch(&kernel.info, grid, block)?;
        if kernel.info.binding_count as usize != bindings.len()
            || kernel.info.constant_byte_length as usize != constants.len
        {
            return Err(Error::Message(
                "kernel binding or constant byte count mismatch".into(),
            ));
        }
        let config = sys::DispatchConfig {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: sys::SUBGROUP_SIZE_FROM_EXECUTABLE,
        };
        unsafe {
            check(
                sys::hrx_stream_dispatch(
                    self.inner.stream,
                    kernel.executable.raw,
                    kernel.ordinal,
                    &config,
                    constants.bytes.as_ptr().cast(),
                    constants.len,
                    raw_bindings.as_ptr(),
                    bindings.len(),
                    0,
                ),
                "dispatch",
            )
        }
    }

    /// Reuse a cached allocation or allocate new scratch storage. Reuse follows
    /// the stream's command order and requires no host wait.
    /// Return it explicitly with [`Stream::recycle`] after recording its last use.
    /// Dropping a buffer releases it rather than returning it to this pool.
    pub fn scratch(&mut self, bytes: usize) -> Result<Buffer> {
        let choice = self
            .scratch
            .range(bytes.max(1)..=bytes.saturating_mul(2).max(1))
            .next()
            .map(|(&size, _)| size);
        if let Some(size) = choice {
            let buffers = self.scratch.get_mut(&size).unwrap();
            let buffer = buffers.pop().unwrap();
            if buffers.is_empty() {
                self.scratch.remove(&size);
            }
            self.scratch_bytes -= buffer.bytes;
            Ok(buffer)
        } else {
            self.allocate(bytes)
        }
    }
    /// Return an allocation to this stream's bounded scratch pool after recording
    /// its final use. A later scratch request may reuse its storage.
    /// Returns an error for a buffer another stream allocated: a scratch pool is
    /// one stream's private free list, even though the buffer itself is usable
    /// from any stream on the device.
    /// Recycling needs mutable stream access; there is no automatic return on drop.
    pub fn recycle(&mut self, buffer: Buffer) -> Result<()> {
        if !std::sync::Arc::ptr_eq(&buffer._device, &self.inner) {
            return Err(Error::Message("scratch belongs to another stream".into()));
        }
        if buffer.bytes <= self.scratch_limit.saturating_sub(self.scratch_bytes) {
            self.scratch_bytes += buffer.bytes;
            self.scratch.entry(buffer.bytes).or_default().push(buffer);
        }
        Ok(())
    }
    /// Set the scratch byte budget, evicting the largest size classes as needed.
    pub fn set_scratch_limit(&mut self, bytes: usize) {
        self.scratch_limit = bytes;
        while self.scratch_bytes > bytes {
            // Evict the largest size class first, preserving smaller reusable buffers.
            if let Some((_, buffers)) = self.scratch.pop_last() {
                self.scratch_bytes -= buffers.iter().map(|b| b.bytes).sum::<usize>();
            } else {
                break;
            }
        }
    }
}

/// An immutable stream completion point. Native queues retain its semaphore
/// after a wait is enqueued, so dropping the event never cancels a dependency.
/// This is a synchronization primitive; the pinned runtime has no GPU timestamps.
pub struct Event {
    raw: sys::Event,
    inner: std::sync::Arc<Inner>,
}
// The recorded event is immutable and retains its native device and semaphore.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
impl Event {
    /// Query completion of the work preceding this event.
    pub fn is_complete(&self) -> Result<bool> {
        let mut complete = false;
        unsafe {
            check(sys::hrx_event_query(self.raw, &mut complete), "query event")?;
        }
        Ok(complete)
    }
    /// Wait on the host for the work preceding this event.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(sys::hrx_event_synchronize(self.raw), "synchronize event") }
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::hrx_event_release(self.raw) }
    }
}

/// A borrowed submission fence. Waiting also releases completed staging. Drop
/// does not wait; the stream continues to own everything needed by queued work.
#[must_use]
pub struct Submission<'a> {
    stream: &'a mut Stream,
}
impl Submission<'_> {
    /// Query submitted work; successful completion reclaims upload staging.
    pub fn is_complete(&mut self) -> Result<bool> {
        let mut done = false;
        unsafe {
            check(
                sys::hrx_stream_query(self.stream.inner.stream, &mut done),
                "query submission",
            )?;
        }
        if done {
            self.stream.reclaim_staging();
        }
        Ok(done)
    }
    /// Wait for this submission and reclaim completed upload staging.
    pub fn wait(self) -> Result<()> {
        self.stream.synchronize()
    }
}

/// Explicit scalar widths, packed in declaration order for HRX binding dispatch.
/// Unlike direct kernargs this format has no implicit alignment or pointer slots.
/// Export metadata gives only the total byte count. Mixed scalar widths must
/// come from the kernel's argument contract.
#[derive(Clone, Debug)]
pub struct Constants {
    bytes: [u8; 256],
    len: usize,
}
impl Default for Constants {
    fn default() -> Self {
        Self {
            bytes: [0; 256],
            len: 0,
        }
    }
}
impl Constants {
    /// Pack an entry point whose scalar arguments are all unsigned Loom indices.
    ///
    /// Loom lowers a homogeneous index list to 32- or 64-bit slots depending on
    /// its range analysis. This adapter is only for that model contract; mixed
    /// scalar types must use `push` with their explicit native widths.
    ///
    /// Neither [`ExportInfo`] nor the compiler's resource report carries per-slot
    /// scalar types — only `constant_byte_length` and `parameter_count` — so no
    /// API here can build a mixed-width constant block by construction. A width
    /// inferred by dividing the byte count is wrong for a mixed `u32`/`u64`
    /// signature, and wrong quietly. Take each width from the source that
    /// declared it.
    pub fn indices(kernel: &Kernel, values: &[u32]) -> Result<Self> {
        let size = kernel.info.constant_byte_length as usize;
        let mut constants = Self::new();
        if values.is_empty() {
            if size != 0 {
                return Err(Error::Message("missing index constants".into()));
            }
            return Ok(constants);
        }
        if size == values.len() * 4 {
            for &value in values {
                constants.push(value)?;
            }
        } else if size == values.len() * 8 {
            for &value in values {
                constants.push(u64::from(value))?;
            }
        } else {
            return Err(Error::Message(
                "export is not a homogeneous 32/64-bit index list".into(),
            ));
        }
        Ok(constants)
    }
    /// Create an empty constant block.
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use = "constant packing can fail when capacity is exceeded"]
    /// Append a scalar at its explicit width; fails if the 256-byte capacity is exceeded.
    pub fn push<T: Scalar>(&mut self, value: T) -> Result<()> {
        let bytes = value.bytes();
        checked_span(self.len, bytes.as_ref().len(), self.bytes.len())?;
        self.bytes[self.len..self.len + bytes.as_ref().len()].copy_from_slice(bytes.as_ref());
        self.len += bytes.as_ref().len();
        Ok(())
    }
    /// Construct an explicitly contracted scalar block from packed bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut result = Self::new();
        checked_span(0, bytes.len(), result.bytes.len())?;
        result.bytes[..bytes.len()].copy_from_slice(bytes);
        result.len = bytes.len();
        Ok(result)
    }
    /// The packed constant bytes in declaration order.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}
mod sealed {
    pub trait Sealed {}
}
/// A supported fixed-width scalar for little-endian binding constants.
pub trait Scalar: sealed::Sealed {
    /// The scalar's fixed-size byte representation.
    type Bytes: AsRef<[u8]>;
    /// Encode this value in little-endian order.
    fn bytes(self) -> Self::Bytes;
}
macro_rules! scalars {
    ($($ty:ty => $n:literal),*) => {
        $(
            impl sealed::Sealed for $ty {}
            impl Scalar for $ty {
                type Bytes = [u8; $n];
                fn bytes(self) -> Self::Bytes { self.to_le_bytes() }
            }
        )*
    };
}
scalars!(u32 => 4, i32 => 4, f32 => 4, u64 => 8, i64 => 8, f64 => 8);
impl Drop for Stream {
    fn drop(&mut self) {
        if !self.staging.is_empty() && self.synchronize_native().is_err() {
            // A failed wait provides no proof that mapped staging is idle.
            std::mem::forget(std::mem::take(&mut self.staging));
        }
    }
}

/// A recorded operation, used to declare what later operations depend on.
///
/// Copy and cheap: it is an index into its own graph, not a native handle, so it
/// never borrows the graph and can be held across recording calls. A node from
/// another graph is rejected rather than silently indexing the wrong recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    graph: u64,
    index: u32,
}

static NEXT_GRAPH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A recording of GPU work as a dependency graph, replayed as a unit.
///
/// Every operation states what it comes after. Nothing is implicit: `&[]` records
/// work that may run as soon as the graph starts, and `&[a, b]` records work that
/// waits for both. The runtime schedules the result — `graph_analysis.c` does a
/// topological sort, partitions it, and detects independent workstreams, running
/// up to eight concurrently — so declaring only the edges that exist is what lets
/// it overlap anything.
///
/// Dependencies affect barriers and partitioning; there is no fixed cost per
/// edge. The pinned scheduler considers additional workstreams only after the
/// first 16 recordable nodes of a partition, and an empty join node ends that
/// partition. Declaring branches permits overlap but does not guarantee it.
///
/// A dependency can only name an already-recorded node, so a recording is
/// acyclic by construction and every edge points forward. That also keeps
/// instantiation on the runtime's linear fast path. Naming the same node twice
/// in one list is collapsed rather than rejected, so `after` may be assembled
/// from overlapping stage outputs.
///
/// Addresses, constants and shapes are fixed at record time. Native capture and
/// graph-exec update are unimplemented in the pinned revision rather than merely
/// unwrapped: `hrx_graph_exec_update` is a 17-byte stub and
/// `hrx_stream_capture_status` is 3 bytes, so changing a recording means
/// recording a new one.
pub struct Graph<'a> {
    raw: sys::Graph,
    // Brands this graph's nodes so another graph's cannot be resolved here.
    id: u64,
    nodes: Vec<sys::GraphNode>,
    inner: std::sync::Arc<Inner>,
    _resources: std::marker::PhantomData<(&'a Buffer, &'a Kernel)>,
}
impl Stream {
    /// Begin recording while borrowing this stream and the recorded resources.
    /// `finish` instantiates an owned executable and ends those borrows.
    ///
    /// ```compile_fail
    /// fn escape() -> hrx::Result<hrx::Graph<'static>> {
    ///     let stream = hrx::Stream::open()?;
    ///     stream.graph()
    /// }
    /// ```
    pub fn graph(&self) -> Result<Graph<'_>> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_create(self.inner.device, 0, &mut raw),
                "create graph",
            )?;
        }
        Ok(Graph {
            raw,
            id: NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            nodes: Vec::new(),
            inner: self.inner.clone(),
            _resources: std::marker::PhantomData,
        })
    }
    /// Replay an instantiated graph on its original stream.
    ///
    /// A native submission failure may leave partial work in flight. Such a
    /// graph cannot be replayed again and its native resources are retained on
    /// drop because completion is unknown. Rejecting a foreign stream does not
    /// invalidate the graph.
    pub fn launch(&mut self, graph: &mut GraphExec) -> Result<()> {
        if !std::sync::Arc::ptr_eq(&graph.inner, &self.inner) {
            return Err(Error::Message("graph belongs to another stream".into()));
        }
        if matches!(graph.completion, GraphCompletion::Failed) {
            return Err(Error::Message(
                "graph has an untracked failed launch".into(),
            ));
        }
        // A native launch can fail after submitting only part of its work. In
        // that case the stream timeline does not cover everything submitted:
        // neither an older fence nor a later successful launch makes release safe.
        graph.completion = GraphCompletion::Failed;
        unsafe {
            check(
                sys::hrx_graph_exec_launch(graph.raw, self.inner.stream),
                "launch graph",
            )?;
            let mut point = sys::TimelinePoint::default();
            check(
                sys::hrx_stream_get_timeline_position(self.inner.stream, &mut point),
                "snapshot graph completion",
            )?;
            graph.completion = GraphCompletion::Submitted(point);
            Ok(())
        }
    }
}
impl<'a> Graph<'a> {
    /// Translate caller-facing nodes into native handles, rejecting foreign ones.
    /// Small fan-in stays on the stack, as dispatch bindings do.
    fn resolve<'s>(
        &self,
        after: &[Node],
        stack: &'s mut [std::mem::MaybeUninit<sys::GraphNode>; 16],
    ) -> Result<std::borrow::Cow<'s, [sys::GraphNode]>> {
        let handle = |node: &Node| -> Result<sys::GraphNode> {
            if node.graph != self.id || node.index as usize >= self.nodes.len() {
                return Err(Error::Message(
                    "dependency node belongs to another graph".into(),
                ));
            }
            Ok(self.nodes[node.index as usize])
        };
        // The native sort counts in-degree per entry but clears it once, so a
        // repeated dependency would strand a node. Collapsing here keeps the
        // caller free to assemble `after` from overlapping stage outputs.
        if after.len() <= stack.len() {
            // Small fan-in is the common case and never allocates: dedupe runs
            // against indices already on the stack, beside the handles themselves.
            let mut seen = [0u32; 16];
            let mut count = 0;
            for node in after {
                let raw = handle(node)?;
                if seen[..count].contains(&node.index) {
                    continue;
                }
                seen[count] = node.index;
                stack[count].write(raw);
                count += 1;
            }
            // Only the prefix written above is exposed, and a node handle is Copy.
            Ok(std::borrow::Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(stack.as_ptr().cast(), count)
            }))
        } else {
            let mut unique: Vec<sys::GraphNode> = Vec::with_capacity(after.len());
            for node in after {
                let raw = handle(node)?;
                if !unique.contains(&raw) {
                    unique.push(raw);
                }
            }
            // Native in-degree is 16-bit; only a heap-sized fan-in can reach it.
            if unique.len() > u16::MAX as usize {
                return Err(Error::Message("too many graph dependencies".into()));
            }
            Ok(std::borrow::Cow::Owned(unique))
        }
    }
    fn record(&mut self, raw: sys::GraphNode) -> Node {
        let index = self.nodes.len() as u32;
        self.nodes.push(raw);
        Node {
            graph: self.id,
            index,
        }
    }
    /// Record a byte-pattern fill over a nonempty span on this device.
    pub fn fill(&mut self, after: &[Node], dst: View<'a>, pattern: u8) -> Result<Node> {
        owns(&self.inner, dst.owner)?;
        if dst.is_empty() {
            return Err(Error::Message("empty graph fill".into()));
        }
        let attrs = sys::GraphFill {
            dst: dst.raw,
            pattern: pattern.into(),
            pattern_size: 1,
        };
        let mut storage = [std::mem::MaybeUninit::uninit(); 16];
        let deps = self.resolve(after, &mut storage)?;
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_fill_buffer_node(
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &attrs,
                    &mut next,
                ),
                "record fill",
            )?;
        }
        Ok(self.record(next))
    }
    /// Record a copy between equal, nonempty spans on this device.
    pub fn copy(&mut self, after: &[Node], dst: View<'a>, src: View<'a>) -> Result<Node> {
        owns(&self.inner, dst.owner)?;
        owns(&self.inner, src.owner)?;
        if dst.len() != src.len() || dst.is_empty() {
            return Err(Error::Message(
                "graph copy requires equal nonempty spans".into(),
            ));
        }
        let attrs = sys::GraphCopy {
            src: src.raw,
            dst: dst.raw,
        };
        let mut storage = [std::mem::MaybeUninit::uninit(); 16];
        let deps = self.resolve(after, &mut storage)?;
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_copy_buffer_node(
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &attrs,
                    &mut next,
                ),
                "record copy",
            )?;
        }
        Ok(self.record(next))
    }
    /// Record a kernel invocation.
    ///
    /// # Safety
    /// As [`Stream::dispatch`]. Constants, addresses and grid are fixed for every
    /// replay. Nodes that access overlapping spans, with at least one write,
    /// must be ordered through `after`. Shared read-only weights need no edge.
    pub unsafe fn dispatch(
        &mut self,
        after: &[Node],
        kernel: &'a Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'a>],
    ) -> Result<Node> {
        if kernel.executable.device.device != self.inner.device {
            return Err(Error::Message("kernel belongs to another device".into()));
        }
        for view in bindings {
            owns(&self.inner, view.owner)?;
        }
        let mut binding_storage = [std::mem::MaybeUninit::uninit(); 32];
        let raw_bindings = raw_bindings(bindings, &mut binding_storage);
        validate_export_launch(&kernel.info, grid, block)?;
        if kernel.info.binding_count as usize != bindings.len()
            || kernel.info.constant_byte_length as usize != constants.len
        {
            return Err(Error::Message(
                "graph binding or constant byte count mismatch".into(),
            ));
        }
        // graph.c copies constants and binding descriptors into its arena, but
        // those descriptors only borrow HAL resources until instantiation. Thus
        // buffers/kernels borrow for recording, while constants borrow for this call.
        let attrs = sys::GraphKernel {
            executable: kernel.executable.raw,
            ordinal: kernel.ordinal,
            config: sys::DispatchConfig {
                workgroup_count: grid,
                workgroup_size: block,
                subgroup_size: sys::SUBGROUP_SIZE_FROM_EXECUTABLE,
            },
            constants: constants.bytes.as_ptr().cast(),
            constants_size: constants.len,
            bindings: raw_bindings.as_ptr(),
            binding_count: bindings.len(),
            flags: 0,
        };
        let mut storage = [std::mem::MaybeUninit::uninit(); 16];
        let deps = self.resolve(after, &mut storage)?;
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_kernel_node(
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &attrs,
                    &mut next,
                ),
                "record kernel",
            )?;
        }
        Ok(self.record(next))
    }
    /// Record a node that does no work and exists only to collect dependencies.
    ///
    /// The pinned runtime gives this node its own partition and queue barrier.
    /// For one consumer, pass the producer nodes directly to that operation.
    /// A join can express a shared dependency, but is not necessarily cheaper.
    pub fn join(&mut self, after: &[Node]) -> Result<Node> {
        let mut storage = [std::mem::MaybeUninit::uninit(); 16];
        let deps = self.resolve(after, &mut storage)?;
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_empty_node(self.raw, deps.as_ptr(), deps.len(), &mut next),
                "record join",
            )?;
        }
        Ok(self.record(next))
    }
    /// Instantiate the recording, retaining native resources independently of its
    /// borrows. A cyclic graph is rejected here.
    pub fn finish(self) -> Result<GraphExec> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_instantiate(self.raw, 0, &mut raw),
                "instantiate graph",
            )?;
        }
        Ok(GraphExec {
            raw,
            inner: self.inner.clone(),
            completion: GraphCompletion::Idle,
        })
    }
}
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::hrx_graph_release(self.raw);
        }
    }
}
/// An instantiated graph. Native instantiation retains HAL allocations and
/// executables; the Arc keeps their device and originating stream alive.
/// Recording borrows its inputs until finish; replay no longer borrows them.
/// Drop waits for its last replay, without submitting or waiting for later
/// stream work. A graph that has never been launched is released immediately.
pub struct GraphExec {
    raw: sys::GraphExec,
    inner: std::sync::Arc<Inner>,
    completion: GraphCompletion,
}

enum GraphCompletion {
    Idle,
    // The semaphore is borrowed from the stream retained by GraphExec::inner.
    Submitted(sys::TimelinePoint),
    Failed,
}
// Send rests on native behaviour, not on anything the compiler checks. What is
// asserted: an instantiated graph owns its recorded HAL resources and semaphore
// state, so moving the handle between threads and launching from the receiving
// one is sound. Rust contributes only exclusivity — launch takes `&mut` — and
// `Arc::ptr_eq` in `Stream::launch`, which rejects a foreign stream but says
// nothing about threads. `independent_streams_move_between_threads` exercises
// this and is evidence, not proof; a native revision that made replay
// thread-affine would invalidate the impl without failing to compile.
// Drop waits only on a captured semaphore/value pair, never on mutable stream
// state: the original Stream may be recording on another thread by then.
unsafe impl Send for GraphExec {}
impl Drop for GraphExec {
    fn drop(&mut self) {
        // Unlike a buffer, whose storage the command buffer retains, releasing
        // an executable graph that is still replaying frees native structures
        // the device is reading: the observed failure is an AMDGPU memory
        // access fault, not an error a caller could handle. Wait for the last
        // replay's immutable completion point, without flushing or reading the
        // stream's current position. A failed launch or wait is no proof the
        // replay is idle, so the native graph leaks rather than freeing early.
        let drained = match self.completion {
            GraphCompletion::Idle => true,
            GraphCompletion::Submitted(point) => check(
                unsafe { sys::hrx_semaphore_wait(point.semaphore, point.value, u64::MAX) },
                "wait for graph completion",
            )
            .is_ok(),
            GraphCompletion::Failed => false,
        };
        if drained {
            unsafe { sys::hrx_graph_exec_release(self.raw) };
        }
    }
}

/// Owned host-visible destination of a queued download. Dropping it before
/// completion is safe: the native command buffer retains its unmapped storage.
#[must_use]
pub struct Readback {
    buffer: Buffer,
}
impl Stream {
    /// Queue a download into owned, initially unmapped host-visible storage.
    pub fn read(&mut self, source: View<'_>) -> Result<Readback> {
        self.owns(source.owner)?;
        let buffer = self.allocate_host(source.len())?;
        let raw = buffer.raw;
        unsafe {
            if !source.is_empty() {
                check(
                    sys::hrx_stream_copy_buffer(
                        self.inner.stream,
                        source.raw.buffer,
                        source.raw.offset,
                        raw,
                        0,
                        source.len(),
                    ),
                    "enqueue readback",
                )?;
            }
            Ok(Readback { buffer })
        }
    }
}
impl Readback {
    /// Wait on the originating stream and return initialized download bytes.
    pub fn wait(self, stream: &mut Stream) -> Result<Vec<u8>> {
        if !std::sync::Arc::ptr_eq(&self.buffer._device, &stream.inner) {
            return Err(Error::Message("readback belongs to another stream".into()));
        }
        stream.synchronize()?;
        let mut bytes = Vec::with_capacity(self.buffer.bytes);
        if self.buffer.bytes != 0 {
            let mut pointer = std::ptr::null_mut();
            unsafe {
                check(
                    sys::hrx_buffer_get_device_ptr(self.buffer.raw, &mut pointer),
                    "map completed readback",
                )?;
                std::ptr::copy_nonoverlapping(
                    pointer.cast::<u8>(),
                    bytes.as_mut_ptr(),
                    self.buffer.bytes,
                );
                // The copy initialized exactly this many bytes of spare capacity.
                bytes.set_len(self.buffer.bytes);
            }
        }
        Ok(bytes)
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("symbol", &self.symbol)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Graph<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.nodes.len())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExec").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Readback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Readback").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Submission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

const STAGING_LIMIT: usize = 64 * 1024 * 1024;

#[cfg(test)]
mod graph_completion_tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151"]
    fn dropping_a_graph_on_another_thread_does_not_submit_pending_work() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(4096)?;
        let mut graph = stream.graph()?;
        graph.fill(&[], buffer.binding(), 1)?;
        let mut replay = graph.finish()?;
        // A later replay must replace the earlier completion point.
        stream.launch(&mut replay)?;
        stream.launch(&mut replay)?;
        let mut before = sys::TimelinePoint::default();
        unsafe {
            check(
                sys::hrx_stream_get_timeline_position(stream.inner.stream, &mut before),
                "snapshot timeline before drop",
            )?;
        }
        stream.fill(buffer.binding(), 2)?;
        std::thread::scope(|scope| {
            let dropper = scope.spawn(move || drop(replay));
            for _ in 0..64 {
                stream.fill(buffer.binding(), 3).unwrap();
            }
            dropper.join().unwrap();
        });
        let mut after = sys::TimelinePoint::default();
        unsafe {
            check(
                sys::hrx_stream_get_timeline_position(stream.inner.stream, &mut after),
                "snapshot timeline after drop",
            )?;
        }
        // This is deterministic even if the threads never overlap: the old
        // destructor flushes the pending fill and advances this timeline.
        assert_eq!(after.value, before.value, "drop submitted unrelated work");
        let mut actual = [0; 4096];
        stream.read_blocking(buffer.binding(), &mut actual)?;
        assert_eq!(actual, [3; 4096]);
        Ok(())
    }
}

#[cfg(test)]
mod staging_tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151"]
    fn a_scratch_pool_only_accepts_the_stream_that_allocated_the_buffer() -> Result<()> {
        let device = Device::open(0)?;
        let mut owner = device.stream()?;
        let mut other = device.stream()?;
        let private = other.scratch(4096)?;
        let private_raw = private.raw;
        other.recycle(private)?;

        // Buffers are device-scoped, so another stream may read and write this one.
        let buffer = owner.allocate(4096)?;
        owner.fill(buffer.binding(), 0x5a)?;
        let ready = owner.record_event()?;
        other.wait_event(&ready)?;
        let mut seen = [0u8; 16];
        other.read_blocking(buffer.binding().slice(0, 16)?, &mut seen)?;
        assert_eq!(seen, [0x5a; 16], "a foreign stream can use the allocation");

        // A scratch pool is still one stream's private free list, not shared state.
        assert!(
            other
                .recycle(buffer)
                .unwrap_err()
                .to_string()
                .contains("another stream")
        );
        assert!(owner.scratch.is_empty());
        assert_eq!(other.scratch_bytes, 4096);
        assert_eq!(other.scratch(4096)?.raw, private_raw);
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn completed_staging_is_reclaimed_and_reused_on_every_completion_path() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(1024)?;
        stream.upload(buffer.binding(), &[7; 1024])?;
        let original = stream.staging[0].raw;
        let mut output = [0; 1024];
        stream.read_blocking(buffer.binding(), &mut output)?;
        assert_eq!(output, [7; 1024]);
        assert!(stream.staging.is_empty());
        stream.upload(buffer.binding(), &[9; 1024])?;
        assert_eq!(stream.staging[0].raw, original);
        stream.upload_blocking(buffer.binding(), &[11; 1024])?;
        assert!(stream.staging.is_empty());
        stream.upload(buffer.binding(), &[12; 512])?;
        assert_eq!(stream.staging[0].raw, original);
        stream.synchronize()?;
        stream.upload(buffer.binding(), &[13; 1024])?;
        stream.synchronize_native()?; // Establish completion without the cleanup being tested.
        assert!(stream.submit()?.is_complete()?);
        assert!(stream.staging.is_empty());
        for _ in 0..32 {
            stream.upload(buffer.binding(), &[1; 1024])?;
            assert!(stream.staging.len() <= 8);
        }
        stream.synchronize()?;
        assert!(stream.staging_pool.len() <= 8);
        assert!(stream.staging_pool.iter().map(|b| b.bytes).sum::<usize>() <= STAGING_LIMIT);
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn uploads_batch_until_pressure_and_reuse_the_smallest_capacity() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(4096)?;
        for size in [1024, 2048, 4096] {
            stream.upload(buffer.binding(), &vec![7; size])?;
        }
        assert_eq!(stream.staging.len(), 3);
        let medium = stream.staging[1].raw;
        stream.synchronize()?;
        stream.upload(buffer.binding(), &[9; 1500])?;
        assert_eq!(stream.staging[0].raw, medium);
        for _ in 0..7 {
            stream.upload(buffer.binding(), &[11; 1024])?;
        }
        assert_eq!(stream.staging.len(), 8);
        stream.upload(buffer.binding(), &[13; 1024])?;
        assert_eq!(stream.staging.len(), 1);
        stream.synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod dag_probe {
    use super::*;

    /// Declared edges cost scheduling time, so the graph API must be able to omit
    /// the ones a workload does not need. Latency-bound by construction: 64 tiny
    /// fills, so bandwidth cannot confound the comparison.
    #[test]
    #[ignore = "requires gfx1151"]
    fn independent_nodes_beat_a_serial_chain() -> Result<()> {
        const NODES: usize = 64;
        let mut stream = Stream::open()?;
        let buffers: Vec<Buffer> = (0..NODES)
            .map(|_| stream.allocate(4096))
            .collect::<Result<_>>()?;
        let mut timings = Vec::new();
        for chained in [true, false] {
            let mut graph = stream.graph()?;
            let mut previous: Option<Node> = None;
            for buffer in &buffers {
                let after: &[Node] = match (chained, &previous) {
                    (true, Some(node)) => std::slice::from_ref(node),
                    _ => &[],
                };
                previous = Some(graph.fill(after, buffer.binding(), 0x3c)?);
            }
            let mut exec = graph.finish()?;
            for _ in 0..3 {
                stream.launch(&mut exec)?;
                stream.synchronize()?;
            }
            let mut samples = Vec::new();
            for _ in 0..9 {
                let start = std::time::Instant::now();
                stream.launch(&mut exec)?;
                stream.synchronize()?;
                samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{:<12} {NODES} nodes: {:8.1} us total, {:6.2} us/node",
                if chained { "serial" } else { "independent" },
                samples[4],
                samples[4] / NODES as f64
            );
            timings.push(samples[4]);
            let mut seen = [0u8; 4096];
            stream.read_blocking(buffers[NODES - 1].binding(), &mut seen)?;
            assert_eq!(seen, [0x3c; 4096], "every node ran");
        }
        // A margin rather than a strict inequality: the property under test is
        // that omitting edges still reaches the runtime, not that a shared CI
        // runner is quiet. The measured gap is ~1.7x, so 1.2x is a wide floor.
        assert!(
            timings[1] * 1.2 < timings[0],
            "omitting edges should be materially cheaper: {timings:?}"
        );
        Ok(())
    }
}
