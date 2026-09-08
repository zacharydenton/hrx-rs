use std::ffi::{CStr, CString, c_void};
use std::path::Path;

/// The target the kernels in this project are compiled for.
pub const TARGET_FAMILY: &str = "amdgpu";
pub const TARGET_KEY: &str = "gfx1151";

use crate::{Error, Result, sys};

/// Turns a status into a `Result`, taking ownership of its message. A null status is success.
pub(crate) fn check(status: sys::Status, what: &str) -> Result<()> {
    if sys::is_ok(status) {
        return Ok(());
    }
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
    Err(Error(format!("{what}: {text}")))
}

/// The device and stream themselves, released when the last thing referencing them goes away.
///
/// Buffers and kernels hold one of these, so the stream cannot be released while an allocation or an
/// executable still names its device. The C implementation sidestepped the question by never tearing
/// down at all — its runtime was a function-local static that lived to process exit — which is not
/// something a safe API can leave to the caller to arrange.
struct Inner {
    device: sys::Device,
    stream: sys::Stream,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Only the stream is ours: `hrx_stream_create` hands back a reference, while
        // `hrx_gpu_device_get` returns a borrowed pointer into libhrx's own device array
        // (`*device = &g_gpu.devices[index]`) without retaining. Releasing that would be an
        // over-release of the global registry, which aborts inside libhrx.
        unsafe { sys::hrx_stream_release(self.stream) };
    }
}

// Safety: this only holds handles. They carry atomic reference counts and no thread-local state, so
// both moving one between threads and releasing it on another are sound. It is `Sync` because holding
// the handle is not using it: every stream operation goes through `&Gpu`, and `Gpu` is deliberately not
// `Sync`, so no two threads can drive one stream at once.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// The process's GPU device and its stream. libhrx initialises globally, so this is created once.
pub struct Gpu {
    inner: std::sync::Arc<Inner>,
    /// `Gpu` may be moved between threads but not shared between them: a stream's timepoint and
    /// pending command buffer are ordinary mutable fields, so two threads dispatching through one
    /// would race. `Cell` is `Send` and not `Sync`, which says exactly that.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

/// libhrx initialises once for the process, not once per handle: a second `hrx_gpu_initialize`
/// returns ALREADY_EXISTS, and dropping a `Gpu` releases its stream without shutting the runtime
/// down. So the initialisation is done once behind a lock and every handle after the first reuses it
/// — which is what lets a server create and drop sessions, and a Python caller hold two `H3`s.
///
/// The lock also serialises the call itself: the initialisation touches ordinary global state, so two
/// threads opening concurrently must not both be inside it.
pub(crate) fn initialize_runtime() -> Result<()> {
    sys::load()?;
    static READY: std::sync::OnceLock<Result<()>> = std::sync::OnceLock::new();
    READY
        .get_or_init(|| {
            let _lock = sys::runtime_lock()?;
            unsafe {
                let status = sys::hrx_gpu_initialize(0);
                if sys::hrx_status_code(status) == 6 {
                    sys::hrx_status_ignore(status);
                    Ok(())
                } else {
                    check(status, "hrx_gpu_initialize")
                }
            }
        })
        .clone()
}

impl Gpu {
    pub fn open() -> Result<Self> {
        Self::open_device(0)
    }

    pub fn open_device(index: i32) -> Result<Self> {
        initialize_runtime()?;
        unsafe {
            let mut count = 0;
            check(
                sys::hrx_gpu_device_count(&mut count),
                "hrx_gpu_device_count",
            )?;
            if index < 0 || index >= count {
                return Err(Error(
                    "no GPU device (is the HSA runtime on LD_LIBRARY_PATH? see docs/setup.md)"
                        .into(),
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
                    1,
                    architecture.as_mut_ptr().cast(),
                    architecture.len(),
                ),
                "device architecture",
            )?;
            if architecture.split(|b| *b == 0).next() != Some(TARGET_KEY.as_bytes()) {
                return Err(Error(format!(
                    "expected {TARGET_KEY}, found {}",
                    String::from_utf8_lossy(&architecture)
                )));
            }
            let mut stream = std::ptr::null_mut();
            check(
                sys::hrx_stream_create(device, 0, &mut stream),
                "hrx_stream_create",
            )?;
            Ok(Self {
                inner: std::sync::Arc::new(Inner { device, stream }),
                _not_sync: std::marker::PhantomData,
            })
        }
    }

    pub fn sync(&self) -> Result<()> {
        unsafe {
            check(
                sys::hrx_stream_synchronize(self.inner.stream),
                "hrx_stream_synchronize",
            )
        }
    }

    /// A device-local allocation. Zero bytes is rounded to one so every argument has an address.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let mut buffer = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_allocate_buffer(
                    sys::hrx_device_allocator(self.inner.device),
                    sys::BufferParams {
                        memory_type: sys::MEMORY_TYPE_DEVICE_LOCAL,
                        access: 7,
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

    /// Only the low byte of `value` is meaningful: the fill pattern is one byte wide.
    pub fn memset(&self, dst: &Buffer, value: u8, bytes: usize) -> Result<()> {
        checked_span(0, bytes, dst.bytes)?;
        unsafe {
            check(
                sys::hrx_stream_fill_buffer(
                    self.inner.stream,
                    dst.raw,
                    0,
                    bytes,
                    &value as *const u8 as *const c_void,
                    1,
                ),
                "hrx_stream_fill_buffer",
            )
        }
    }

    /// The synchronous transfers bypass the stream's pending commands, so the stream is drained first.
    pub fn h2d(&self, dst: &Buffer, src: &[u8]) -> Result<()> {
        self.h2d_at(dst, 0, src)
    }

    /// As [`Gpu::h2d`], writing at an offset: a large weight is uploaded in chunks so the host never
    /// stages more than one chunk of it.
    pub fn h2d_at(&self, dst: &Buffer, offset: usize, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if offset.checked_add(src.len()).is_none_or(|n| n > dst.bytes) {
            return Err(Error(format!(
                "upload of {} bytes at {offset} overruns a {}-byte allocation",
                src.len(),
                dst.bytes
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_h2d(
                    self.inner.device,
                    src.as_ptr() as *const c_void,
                    dst.raw,
                    offset,
                    src.len(),
                ),
                "hrx_synchronous_h2d",
            )
        }
    }

    pub fn d2h(&self, src: &Buffer, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > src.bytes {
            return Err(Error(format!(
                "read of {} bytes from a {}-byte allocation",
                dst.len(),
                src.bytes
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_d2h(
                    self.inner.device,
                    src.raw,
                    0,
                    dst.as_mut_ptr() as *mut c_void,
                    dst.len(),
                ),
                "hrx_synchronous_d2h",
            )
        }
    }

    /// Reads back from a view rather than a whole allocation, which is how the pipeline inspects rows
    /// it handed a kernel at an offset.
    pub fn d2h_ref(&self, src: View<'_>, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > src.raw.length {
            return Err(Error(format!(
                "read of {} bytes from a {}-byte view",
                dst.len(),
                src.raw.length
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_d2h(
                    self.inner.device,
                    src.raw.buffer,
                    src.raw.offset,
                    dst.as_mut_ptr() as *mut c_void,
                    dst.len(),
                ),
                "hrx_synchronous_d2h",
            )
        }
    }

    pub fn d2d(&self, dst: &Buffer, src: &Buffer, bytes: usize) -> Result<()> {
        self.d2d_at(dst, 0, src, 0, bytes)
    }

    /// A device copy into or out of the middle of an allocation — a row range of a packed sequence,
    /// or one half of a two-row table.
    pub fn d2d_at(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        checked_span(dst_offset, bytes, dst.bytes)?;
        checked_span(src_offset, bytes, src.bytes)?;
        unsafe {
            check(
                sys::hrx_stream_copy_buffer(
                    self.inner.stream,
                    src.raw,
                    src_offset,
                    dst.raw,
                    dst_offset,
                    bytes,
                ),
                "hrx_stream_copy_buffer",
            )
        }
    }

    /// Loads a code object and looks up one of its exports.
    ///
    /// # Safety
    ///
    /// The file is machine code that will run on the device with no sandbox between it and every
    /// other allocation this process holds. Loading one is trusting whoever produced it, exactly as
    /// `dlopen` is: a code object that reads or writes outside the bindings it is given corrupts
    /// memory, and nothing here can tell that it will. Load only code objects this process compiled,
    /// or that come from a source you would equally trust with a shared library.
    pub unsafe fn load(&self, path: &Path, symbol: &str) -> Result<Kernel> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error(format!("{} contains a NUL", path.display())))?;
        let c_family = CString::new(TARGET_FAMILY).unwrap();
        let c_key = CString::new(TARGET_KEY).unwrap();
        let c_symbol =
            CString::new(symbol).map_err(|_| Error(format!("{symbol} contains a NUL")))?;
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
                &format!("loading {}", path.display()),
            )?;
            let kernel = (|| {
                let mut ordinal = 0u32;
                check(
                    sys::hrx_executable_lookup_export_by_name(
                        executable,
                        c_symbol.as_ptr(),
                        &mut ordinal,
                    ),
                    &format!("looking up {symbol} in {}", path.display()),
                )?;
                let mut info = sys::ExportInfo::default();
                check(
                    sys::hrx_executable_export_info(executable, ordinal, &mut info),
                    &format!("export info for {symbol}"),
                )?;
                Ok(Kernel {
                    executable,
                    ordinal,
                    info,
                    symbol: symbol.to_string(),
                    _device: self.inner.clone(),
                })
            })();
            if kernel.is_err() {
                sys::hrx_executable_release(executable);
            }
            kernel
        }
    }

    /// Dispatch, with the export's own metadata deciding how the constants are packed.
    ///
    /// `scalars` are the kernel's by-value arguments in declaration order and `bindings` its buffer
    /// arguments in declaration order. The export reports how many bytes of constants it wants and how
    /// many bindings it has; a mismatch is a caller error and is reported as one rather than dispatched.
    /// Runs a kernel.
    ///
    /// # Safety
    ///
    /// A [`View`] proves that the allocation it names is still alive. It proves nothing about what
    /// the kernel does with it. The device code addresses its bindings itself, from the scalars it is
    /// given and from indices it computes, so a grid, a block size or a scalar that disagrees with
    /// the kernel's own expectations reads or writes outside them — the length in a binding is
    /// descriptive, not enforced.
    ///
    /// The caller must know that `kernel` is the kernel it thinks it is, and that `grid`, `block`,
    /// `scalars` and `bindings` are the shapes it was compiled for. In this workspace that knowledge
    /// lives in `h3::dispatch`, whose builders compile a kernel and launch it from the same
    /// description; nothing else should call this directly.
    pub unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        scalars: &[u32],
        bindings: &[View<'_>],
    ) -> Result<()> {
        validate_launch(grid, block)?;
        let info = &kernel.info;
        if bindings.len() != info.binding_count as usize {
            return Err(Error(format!(
                "{} takes {} buffer arguments, {} given",
                kernel.symbol,
                info.binding_count,
                bindings.len()
            )));
        }
        let mut constants = [0u8; 256];
        let size = info.constant_byte_length as usize;
        if size > constants.len() {
            return Err(Error(format!(
                "{} wants {size} constant bytes",
                kernel.symbol
            )));
        }
        if !scalars.is_empty() {
            if !size.is_multiple_of(scalars.len()) {
                return Err(Error(format!(
                    "{} wants {size} constant bytes, not divisible by {} scalars",
                    kernel.symbol,
                    scalars.len()
                )));
            }
            let width = size / scalars.len();
            if width != 4 && width != 8 {
                return Err(Error(format!(
                    "{} implies a {width}-byte scalar slot",
                    kernel.symbol
                )));
            }
            for (i, v) in scalars.iter().enumerate() {
                constants[i * width..i * width + 4].copy_from_slice(&v.to_le_bytes());
            }
        } else if size != 0 {
            return Err(Error(format!(
                "{} wants {size} constant bytes, none given",
                kernel.symbol
            )));
        }
        let config = sys::DispatchConfig {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: 32,
        };
        unsafe {
            check(
                sys::hrx_stream_dispatch(
                    self.inner.stream,
                    kernel.executable,
                    kernel.ordinal,
                    &config,
                    constants.as_ptr() as *const c_void,
                    size,
                    bindings.as_ptr().cast::<sys::BufferRef>(),
                    bindings.len(),
                    0,
                ),
                "dispatching kernel",
            )
        }
    }
}

/// A device allocation. libhrx buffers have no host-visible address, so this is the handle itself
/// rather than a pointer; bindings are made with [`Buffer::binding`].
pub struct Buffer {
    raw: sys::Buffer,
    bytes: usize,
    /// Keeps the device alive: releasing a buffer after its device is gone would be a use-after-free.
    _device: std::sync::Arc<Inner>,
}

// Safety: `hrx_buffer_s` is an immutable handle after allocation — a HAL buffer pointer, its device and
// its length — behind an atomic reference count, so moving one to another thread and releasing it there
// is sound. It is `Sync` as well because nothing mutates it through a shared reference: every transfer
// and fill goes through `&Gpu`, which is deliberately not `Sync`, so two threads cannot reach the same
// allocation concurrently through this API.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// A binding into a device allocation, borrowed from it.
///
/// The lifetime is the point. `hrx_buffer_s` is a raw handle, so a view carrying one is `Copy` and
/// would happily outlive the allocation it names — safe code could drop the buffer and still dispatch
/// against it. Borrowing the buffer makes that a compile error instead.
/// `repr(transparent)` so an array of views is an array of `hrx_buffer_ref_t` and can be handed to
/// the dispatch as it stands: the only non-zero-sized field is the binding itself.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct View<'a> {
    raw: sys::BufferRef,
    owner: std::marker::PhantomData<&'a Buffer>,
}

impl<'a> View<'a> {
    fn new(raw: sys::BufferRef) -> Self {
        Self {
            raw,
            owner: std::marker::PhantomData,
        }
    }
    /// The bytes this view covers.
    pub fn len(&self) -> usize {
        self.raw.length
    }
    pub fn is_empty(&self) -> bool {
        self.raw.length == 0
    }
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn try_slice(&self, offset: usize, length: usize) -> Result<View<'_>> {
        checked_span(offset, length, self.bytes)?;
        Ok(View::new(sys::BufferRef {
            buffer: self.raw,
            offset,
            length,
        }))
    }

    /// The whole allocation.
    pub fn binding(&self) -> View<'_> {
        View::new(sys::BufferRef {
            buffer: self.raw,
            offset: 0,
            length: self.bytes,
        })
    }

    /// A binding into part of the allocation. The pipeline hands kernels views at row offsets, which
    /// the C did with pointer arithmetic; a device buffer here has no host-visible address, so the
    /// offset travels in the binding instead.
    pub fn slice(&self, offset: usize, length: usize) -> View<'_> {
        // checked: `offset + length` wraps for a large offset, and the wrapped sum passes the
        // comparison — an eight-byte allocation would accept `slice(usize::MAX, 2)`
        self.try_slice(offset, length)
            .expect("slice past the allocation or overflow")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { sys::hrx_buffer_release(self.raw) }
    }
}

pub struct Kernel {
    executable: sys::Executable,
    ordinal: u32,
    info: sys::ExportInfo,
    symbol: String,
    /// As for a buffer: the executable names its device.
    _device: std::sync::Arc<Inner>,
}

// Safety: `hrx_executable_s` is fixed once loaded — a retained HAL executable, its device and a
// snapshot of its export names — behind an atomic reference count. Everything this type exposes is
// read-only, and dispatching with it needs `&Gpu`.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn info(&self) -> &sys::ExportInfo {
        &self.info
    }
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { sys::hrx_executable_release(self.executable) }
    }
}

#[cfg(test)]
mod tests {
    /// The bounds arithmetic, without a device: `offset + length` must not wrap into a pass.
    #[test]
    fn a_slice_past_the_allocation_is_refused_even_when_the_sum_wraps() {
        let checked = |offset: usize, length: usize, bytes: usize| -> bool {
            match offset.checked_add(length) {
                Some(end) => end <= bytes,
                None => false,
            }
        };
        assert!(checked(0, 8, 8));
        assert!(checked(4, 4, 8));
        assert!(!checked(4, 5, 8));
        // the wrapping case: 8 bytes must not accept an offset near the top of the address space
        assert!(!checked(usize::MAX, 2, 8));
        assert!(!checked(usize::MAX - 1, 4, 8));
    }
}

pub(crate) fn checked_span(offset: usize, length: usize, capacity: usize) -> Result<()> {
    if offset.checked_add(length).is_none_or(|end| end > capacity) {
        Err(Error(format!(
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
        Err(Error("invalid GPU launch dimensions".into()))
    } else {
        Ok(())
    }
}

/// A borrowed device registry entry. Opening a stream never acquires ownership
/// of the native device, and dropping a model never shuts down HRX globally.
#[derive(Clone)]
pub struct Device {
    index: i32,
}
impl Device {
    pub fn open(index: i32) -> Result<Self> {
        let stream = Gpu::open_device(index)?;
        drop(stream);
        Ok(Self { index })
    }
    pub fn stream(&self) -> Result<Stream> {
        Ok(Stream {
            gpu: Gpu::open_device(self.index)?,
            staging: Vec::new(),
            scratch: Vec::new(),
            scratch_bytes: 0,
            scratch_limit: 256 * 1024 * 1024,
        })
    }
}

/// An ordered command stream. Mutating operations require exclusive access.
/// Prepared kernels and allocations can be reused without a compiler/cache lookup.
/// `Gpu` is the legacy H3 interface; new model code should use this type.
pub struct Stream {
    gpu: Gpu,
    // Storage is retained by the stream, not by a completion token whose Drop
    // could be skipped. Forgetting a token can never free in-flight host memory.
    staging: Vec<Buffer>,
    scratch: Vec<Buffer>,
    scratch_bytes: usize,
    scratch_limit: usize,
}
impl Stream {
    pub fn open() -> Result<Self> {
        Device::open(0)?.stream()
    }
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        self.gpu.alloc(bytes)
    }
    pub fn synchronize(&mut self) -> Result<()> {
        self.gpu.sync()?;
        self.staging.clear();
        Ok(())
    }
    pub fn upload(&mut self, dst: &Buffer, bytes: &[u8]) -> Result<()> {
        self.gpu.h2d(dst, bytes)
    }
    pub fn read(&mut self, src: View<'_>, bytes: &mut [u8]) -> Result<()> {
        self.gpu.d2h_ref(src, bytes)
    }
    pub fn fill(&mut self, dst: &Buffer, value: u8) -> Result<()> {
        self.gpu.memset(dst, value, dst.bytes)
    }
    pub fn copy(
        &mut self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        self.gpu.d2d_at(dst, dst_offset, src, src_offset, bytes)
    }
    /// Copy the input into runtime-owned staging, then enqueue an actual GPU
    /// buffer copy. The caller's slice is no longer referenced when this returns.
    pub fn upload_queued(&mut self, dst: &Buffer, offset: usize, bytes: &[u8]) -> Result<()> {
        checked_span(offset, bytes.len(), dst.bytes)?;
        if bytes.is_empty() {
            return Ok(());
        }
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_allocate_buffer(
                    sys::hrx_device_allocator(self.gpu.inner.device),
                    sys::BufferParams {
                        memory_type: 0x42 | 0x10 | 4,
                        access: 7,
                        usage: sys::BUFFER_USAGE_DEFAULT | 0x0100_0000,
                        queue_affinity: u64::MAX,
                    },
                    bytes.len(),
                    &mut raw,
                ),
                "allocate staging",
            )?;
            let staging = Buffer {
                raw,
                bytes: bytes.len(),
                _device: self.gpu.inner.clone(),
            };
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
                    self.gpu.inner.stream,
                    raw,
                    0,
                    dst.raw,
                    offset,
                    bytes.len(),
                ),
                "enqueue upload",
            )
        }
    }
    /// Submit pending commands before querying completion; native query alone
    /// does not include the unsubmitted command buffer.
    pub fn submit(&mut self) -> Result<Submission<'_>> {
        unsafe {
            check(sys::hrx_stream_flush(self.gpu.inner.stream), "submit")?;
        }
        Ok(Submission { stream: self })
    }
    /// # Safety
    /// The code object must be trusted native machine code.
    pub unsafe fn load(&self, path: &Path, symbol: &str) -> Result<Kernel> {
        unsafe { self.gpu.load(path, symbol) }
    }
    /// # Safety
    /// Kernel, dimensions, constants and binding spans must agree. GPU addressing
    /// is not sandboxed by a binding's length.
    pub unsafe fn dispatch(
        &mut self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'_>],
    ) -> Result<()> {
        validate_launch(grid, block)?;
        if kernel.info.binding_count as usize != bindings.len()
            || kernel.info.constant_byte_length as usize != constants.len
        {
            return Err(Error(
                "kernel binding or constant byte count mismatch".into(),
            ));
        }
        let config = sys::DispatchConfig {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: 32,
        };
        unsafe {
            check(
                sys::hrx_stream_dispatch(
                    self.gpu.inner.stream,
                    kernel.executable,
                    kernel.ordinal,
                    &config,
                    constants.bytes.as_ptr().cast(),
                    constants.len,
                    bindings.as_ptr().cast(),
                    bindings.len(),
                    0,
                ),
                "dispatch",
            )
        }
    }
    /// Return scratch only after its last recorded use. A pool belongs to exactly
    /// one stream, so reuse is ordered without a host wait. No size-class strings.
    pub fn scratch(&mut self, bytes: usize) -> Result<Buffer> {
        let choice = self
            .scratch
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bytes >= bytes && b.bytes <= bytes.saturating_mul(2).max(1))
            .min_by_key(|(_, b)| b.bytes)
            .map(|(i, _)| i);
        if let Some(i) = choice {
            let b = self.scratch.swap_remove(i);
            self.scratch_bytes -= b.bytes;
            Ok(b)
        } else {
            self.allocate(bytes)
        }
    }
    pub fn recycle(&mut self, buffer: Buffer) -> Result<()> {
        if !std::sync::Arc::ptr_eq(&buffer._device, &self.gpu.inner) {
            return Err(Error("scratch belongs to another stream".into()));
        }
        if buffer.bytes <= self.scratch_limit.saturating_sub(self.scratch_bytes) {
            self.scratch_bytes += buffer.bytes;
            self.scratch.push(buffer);
        }
        Ok(())
    }
    pub fn set_scratch_limit(&mut self, bytes: usize) {
        self.scratch_limit = bytes;
        while self.scratch_bytes > bytes {
            if let Some(b) = self.scratch.pop() {
                self.scratch_bytes -= b.bytes;
            } else {
                break;
            }
        }
    }
}

/// A borrowed submission fence. Waiting also releases completed staging. Drop
/// does not wait; the stream continues to own everything needed by queued work.
pub struct Submission<'a> {
    stream: &'a mut Stream,
}
impl Submission<'_> {
    pub fn is_complete(&self) -> Result<bool> {
        let mut done = false;
        unsafe {
            check(
                sys::hrx_stream_query(self.stream.gpu.inner.stream, &mut done),
                "query submission",
            )?;
        }
        Ok(done)
    }
    pub fn wait(self) -> Result<()> {
        self.stream.synchronize()
    }
}

/// Explicit scalar widths, packed in declaration order for HRX binding dispatch.
/// Unlike direct kernargs this format has no implicit alignment or pointer slots.
#[derive(Clone)]
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
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push<T: Scalar>(&mut self, value: T) -> Result<&mut Self> {
        let bytes = value.bytes();
        checked_span(self.len, bytes.as_ref().len(), self.bytes.len())?;
        self.bytes[self.len..self.len + bytes.as_ref().len()].copy_from_slice(bytes.as_ref());
        self.len += bytes.as_ref().len();
        Ok(self)
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}
mod sealed {
    pub trait Sealed {}
}
pub trait Scalar: sealed::Sealed {
    type Bytes: AsRef<[u8]>;
    fn bytes(self) -> Self::Bytes;
}
macro_rules! scalars { ($($ty:ty => $n:literal),*) => { $(impl sealed::Sealed for $ty {} impl Scalar for $ty { type Bytes = [u8;$n]; fn bytes(self) -> Self::Bytes { self.to_le_bytes() } })* }; }
scalars!(u32 => 4, i32 => 4, f32 => 4, u64 => 8, i64 => 8, f64 => 8);
impl Drop for Stream {
    fn drop(&mut self) {
        if !self.staging.is_empty() && self.gpu.sync().is_err() {
            // A failed wait provides no proof that mapped staging is idle.
            std::mem::forget(std::mem::take(&mut self.staging));
        }
    }
}

/// Recording for a fixed sequence of operations. Each added operation depends
/// on the previous one. Native capture/update APIs are deliberately not exposed:
/// they are unimplemented in the pinned HRX revision.
pub struct SequenceBuilder<'a> {
    graph: sys::Graph,
    last: sys::GraphNode,
    device: sys::Device,
    _resources: std::marker::PhantomData<(&'a Buffer, &'a Kernel)>,
}
impl Stream {
    pub fn sequence<'a>(&self) -> Result<SequenceBuilder<'a>> {
        let mut graph = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_create(self.gpu.inner.device, 0, &mut graph),
                "create sequence",
            )?;
        }
        Ok(SequenceBuilder {
            graph,
            last: std::ptr::null_mut(),
            device: self.gpu.inner.device,
            _resources: std::marker::PhantomData,
        })
    }
    pub fn launch_sequence(&mut self, sequence: &mut FixedSequence<'_>) -> Result<()> {
        if sequence.device != self.gpu.inner.device {
            return Err(Error("sequence belongs to another device".into()));
        }
        unsafe {
            check(
                sys::hrx_graph_exec_launch(sequence.raw, self.gpu.inner.stream),
                "launch sequence",
            )
        }
    }
}
impl<'a> SequenceBuilder<'a> {
    fn deps(&self) -> (*const sys::GraphNode, usize) {
        (&self.last, usize::from(!self.last.is_null()))
    }
    pub fn fill(&mut self, dst: View<'a>, pattern: u8) -> Result<&mut Self> {
        if dst.is_empty() {
            return Err(Error("empty sequence fill".into()));
        }
        let attrs = sys::GraphFill {
            dst: dst.raw,
            pattern: pattern.into(),
            pattern_size: 1,
        };
        let (deps, count) = self.deps();
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_fill_buffer_node(self.graph, deps, count, &attrs, &mut next),
                "record fill",
            )?;
        }
        self.last = next;
        Ok(self)
    }
    pub fn copy(&mut self, dst: View<'a>, src: View<'a>) -> Result<&mut Self> {
        if dst.len() != src.len() || dst.is_empty() {
            return Err(Error("sequence copy requires equal nonempty spans".into()));
        }
        let attrs = sys::GraphCopy {
            src: src.raw,
            dst: dst.raw,
        };
        let (deps, count) = self.deps();
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_copy_buffer_node(self.graph, deps, count, &attrs, &mut next),
                "record copy",
            )?;
        }
        self.last = next;
        Ok(self)
    }
    /// # Safety
    /// As [`Stream::dispatch`]. Constants, addresses and grid are fixed for every
    /// replay. To change them, build a new sequence.
    pub unsafe fn dispatch(
        &mut self,
        kernel: &'a Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'a>],
    ) -> Result<&mut Self> {
        validate_launch(grid, block)?;
        if kernel.info.binding_count as usize != bindings.len()
            || kernel.info.constant_byte_length as usize != constants.len
        {
            return Err(Error(
                "sequence binding or constant byte count mismatch".into(),
            ));
        }
        let attrs = sys::GraphKernel {
            executable: kernel.executable,
            ordinal: kernel.ordinal,
            config: sys::DispatchConfig {
                workgroup_count: grid,
                workgroup_size: block,
                subgroup_size: 32,
            },
            constants: constants.bytes.as_ptr().cast(),
            constants_size: constants.len,
            bindings: bindings.as_ptr().cast(),
            binding_count: bindings.len(),
            flags: 0,
        };
        let (deps, count) = self.deps();
        let mut next = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_add_kernel_node(self.graph, deps, count, &attrs, &mut next),
                "record kernel",
            )?;
        }
        self.last = next;
        Ok(self)
    }
    pub fn finish(self) -> Result<FixedSequence<'a>> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_graph_instantiate(self.graph, 0, &mut raw),
                "instantiate sequence",
            )?;
        }
        Ok(FixedSequence {
            raw,
            device: self.device,
            _resources: std::marker::PhantomData,
        })
    }
}
impl Drop for SequenceBuilder<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::hrx_graph_release(self.graph);
        }
    }
}
pub struct FixedSequence<'a> {
    raw: sys::GraphExec,
    device: sys::Device,
    _resources: std::marker::PhantomData<(&'a Buffer, &'a Kernel)>,
}
// Exclusive launch access; all borrowed resources are Send + Sync. The native
// executable owns its recorded HAL resources and semaphore state.
unsafe impl Send for FixedSequence<'_> {}
impl Drop for FixedSequence<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::hrx_graph_exec_release(self.raw);
        }
    }
}

/// Owned host-visible destination of a queued download. Dropping it before
/// completion is safe: the native command buffer retains its unmapped storage.
pub struct Readback {
    buffer: Buffer,
}
impl Stream {
    pub fn read_queued(&mut self, source: View<'_>) -> Result<Readback> {
        let mut raw = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_allocator_allocate_buffer(
                    sys::hrx_device_allocator(self.gpu.inner.device),
                    sys::BufferParams {
                        memory_type: 0x42 | 0x10 | 4,
                        access: 7,
                        usage: sys::BUFFER_USAGE_DEFAULT | 0x0100_0000,
                        queue_affinity: u64::MAX,
                    },
                    source.len().max(1),
                    &mut raw,
                ),
                "allocate readback",
            )?;
            let buffer = Buffer {
                raw,
                bytes: source.len(),
                _device: self.gpu.inner.clone(),
            };
            if !source.is_empty() {
                check(
                    sys::hrx_stream_copy_buffer(
                        self.gpu.inner.stream,
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
    pub fn wait(self, stream: &mut Stream) -> Result<Vec<u8>> {
        if !std::sync::Arc::ptr_eq(&self.buffer._device, &stream.gpu.inner) {
            return Err(Error("readback belongs to another stream".into()));
        }
        stream.synchronize()?;
        let mut bytes = vec![0; self.buffer.bytes];
        if !bytes.is_empty() {
            let mut pointer = std::ptr::null_mut();
            unsafe {
                check(
                    sys::hrx_buffer_get_device_ptr(self.buffer.raw, &mut pointer),
                    "map completed readback",
                )?;
                std::ptr::copy_nonoverlapping(
                    pointer.cast::<u8>(),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                );
            }
        }
        Ok(bytes)
    }
}
