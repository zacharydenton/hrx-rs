//! libhrx's C API, declared by hand from `libhrx/include/hrx_runtime.h`. No bindgen, so the whole
//! surface this project uses is readable in one file.
//!
//! Two conventions from the header matter and are easy to get wrong:
//!   * `hrx_status_t` is a pointer, and **NULL means success**. `hrx_status_is_ok` is a `static inline`
//!     in the header, so it is not a linkable symbol — [`is_ok`] reimplements it.
//!   * handles are opaque pointers; a null handle is never valid.
use std::ffi::{c_char, c_int, c_void};

pub type Status = *mut c_void;
pub type Device = *mut c_void;
pub type Stream = *mut c_void;
pub type Buffer = *mut c_void;
pub type Executable = *mut c_void;

/// `hrx_status_is_ok`, which the header defines as `static inline` rather than exporting.
#[inline]
pub fn is_ok(status: Status) -> bool {
    status.is_null()
}

pub const MEMORY_TYPE_DEVICE_LOCAL: u32 = 0x0000_0030;
pub const BUFFER_USAGE_DEFAULT: u32 = 0x0000_0C03;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExportInfo {
    pub name: *const c_char,
    pub flags: u32,
    pub constant_byte_length: u32,
    pub binding_count: u32,
    pub parameter_count: u32,
    pub workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchConfig {
    pub workgroup_count: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub subgroup_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BufferRef {
    pub buffer: Buffer,
    pub offset: usize,
    pub length: usize,
}

pub type Allocator = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BufferParams {
    pub memory_type: u32,
    pub access: u16,
    pub usage: u32,
    pub queue_affinity: u64,
}

/// Symbols are resolved once. The libraries are deliberately never unloaded:
/// native global devices and objects may still be held by another model cdylib.
struct Api {
    hrx_graph_create: unsafe extern "C" fn(Device, u32, *mut Graph) -> Status,
    hrx_graph_release: unsafe extern "C" fn(Graph) -> (),
    hrx_graph_add_kernel_node: unsafe extern "C" fn(
        Graph,
        *const GraphNode,
        usize,
        *const GraphKernel,
        *mut GraphNode,
    ) -> Status,
    hrx_graph_add_copy_buffer_node: unsafe extern "C" fn(
        Graph,
        *const GraphNode,
        usize,
        *const GraphCopy,
        *mut GraphNode,
    ) -> Status,
    hrx_graph_add_fill_buffer_node: unsafe extern "C" fn(
        Graph,
        *const GraphNode,
        usize,
        *const GraphFill,
        *mut GraphNode,
    ) -> Status,
    hrx_graph_instantiate: unsafe extern "C" fn(Graph, u32, *mut GraphExec) -> Status,
    hrx_graph_exec_release: unsafe extern "C" fn(GraphExec) -> (),
    hrx_graph_exec_launch: unsafe extern "C" fn(GraphExec, Stream) -> Status,

    hrx_status_to_string: unsafe extern "C" fn(Status, *mut *mut c_char, *mut usize) -> Status,
    hrx_status_free_message: unsafe extern "C" fn(*mut c_char) -> (),
    hrx_status_ignore: unsafe extern "C" fn(Status) -> (),
    hrx_gpu_initialize: unsafe extern "C" fn(u32) -> Status,
    hrx_gpu_device_count: unsafe extern "C" fn(*mut c_int) -> Status,
    hrx_gpu_device_get: unsafe extern "C" fn(c_int, *mut Device) -> Status,
    hrx_stream_create: unsafe extern "C" fn(Device, u32, *mut Stream) -> Status,
    hrx_stream_release: unsafe extern "C" fn(Stream) -> (),
    hrx_stream_synchronize: unsafe extern "C" fn(Stream) -> Status,
    hrx_buffer_allocate: unsafe extern "C" fn(Stream, usize, u32, u32, *mut Buffer) -> Status,
    hrx_buffer_release: unsafe extern "C" fn(Buffer) -> (),
    hrx_stream_fill_buffer:
        unsafe extern "C" fn(Stream, Buffer, usize, usize, *const c_void, usize) -> Status,
    hrx_stream_copy_buffer:
        unsafe extern "C" fn(Stream, Buffer, usize, Buffer, usize, usize) -> Status,
    hrx_synchronous_h2d:
        unsafe extern "C" fn(Device, *const c_void, Buffer, usize, usize) -> Status,
    hrx_synchronous_d2h: unsafe extern "C" fn(Device, Buffer, usize, *mut c_void, usize) -> Status,
    hrx_executable_load_file: unsafe extern "C" fn(
        Device,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut Executable,
    ) -> Status,
    hrx_executable_release: unsafe extern "C" fn(Executable) -> (),
    hrx_executable_lookup_export_by_name:
        unsafe extern "C" fn(Executable, *const c_char, *mut u32) -> Status,
    hrx_executable_export_info: unsafe extern "C" fn(Executable, u32, *mut ExportInfo) -> Status,
    hrx_stream_dispatch: unsafe extern "C" fn(
        Stream,
        Executable,
        u32,
        *const DispatchConfig,
        *const c_void,
        usize,
        *const BufferRef,
        usize,
        u32,
    ) -> Status,
    hrx_status_code: unsafe extern "C" fn(Status) -> c_int,
    hrx_device_allocator: unsafe extern "C" fn(Device) -> Allocator,
    hrx_allocator_allocate_buffer:
        unsafe extern "C" fn(Allocator, BufferParams, usize, *mut Buffer) -> Status,
    hrx_buffer_get_device_ptr: unsafe extern "C" fn(Buffer, *mut *mut c_void) -> Status,
    hrx_device_get_property: unsafe extern "C" fn(Device, c_int, *mut c_void, usize) -> Status,
    hrx_stream_flush: unsafe extern "C" fn(Stream) -> Status,
    hrx_stream_query: unsafe extern "C" fn(Stream, *mut bool) -> Status,
    hrx_stream_update_buffer:
        unsafe extern "C" fn(Stream, *const c_void, usize, Buffer, usize) -> Status,
}
static API: std::sync::OnceLock<Api> = std::sync::OnceLock::new();
pub(crate) fn load() -> crate::Result<()> {
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if API.get().is_some() {
        return Ok(());
    }
    let _guard = INIT
        .lock()
        .map_err(|_| crate::Error("runtime loader poisoned".into()))?;
    if API.get().is_some() {
        return Ok(());
    }
    let directory = crate::bundle::resolve()?;
    let path = std::fs::canonicalize(directory.join("libhrx.so"))?;
    // Process-wide, independent of HOME / cache / which DSO contains this code.
    // The file records the first library choice to refuse incompatible copies.
    let mut lock = runtime_lock()?;
    use std::io::{Read, Seek, Write};
    let mut selected = String::new();
    lock.file().read_to_string(&mut selected)?;
    // PID reuse: only choices within this process lifetime are relevant.
    let start = std::fs::read_to_string("/proc/self/stat")?;
    let start = start
        .rsplit_once(") ")
        .ok_or_else(|| crate::Error("invalid proc stat".into()))?
        .1
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| crate::Error("missing process start time".into()))?;
    let identity = format!("{start} {}", path.display());
    if selected.starts_with(&format!("{start} ")) && selected != identity {
        return Err(crate::Error(format!(
            "another model already loaded {selected}; select one HRX_RUNTIME_DIR for the process"
        )));
    }
    unsafe {
        use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
        let provider = std::env::var_os("IREE_HAL_AMDGPU_LIBHSA_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| directory.join("libhsa-runtime64.so.1"));
        let provider = std::fs::canonicalize(provider)?;
        let hsa = Library::open(Some(&provider), RTLD_NOW | RTLD_LOCAL)
            .map_err(|e| crate::Error(format!("loading {}: {e}", provider.display())))?;
        // No library-controlled environment mutation, and no search-path changes.
        let library = Library::open(Some(&path), RTLD_NOW | RTLD_LOCAL)
            .map_err(|e| crate::Error(format!("loading {}: {e}", path.display())))?;
        let result = (|| -> crate::Result<Api> {
            Ok(Api {
                hrx_graph_create: *library
                    .get(b"hrx_graph_create\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_release: *library
                    .get(b"hrx_graph_release\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_add_kernel_node: *library
                    .get(b"hrx_graph_add_kernel_node\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_add_copy_buffer_node: *library
                    .get(b"hrx_graph_add_copy_buffer_node\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_add_fill_buffer_node: *library
                    .get(b"hrx_graph_add_fill_buffer_node\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_instantiate: *library
                    .get(b"hrx_graph_instantiate\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_exec_release: *library
                    .get(b"hrx_graph_exec_release\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_graph_exec_launch: *library
                    .get(b"hrx_graph_exec_launch\0")
                    .map_err(|e| crate::Error(e.to_string()))?,

                hrx_status_to_string: *library
                    .get(b"hrx_status_to_string\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_status_free_message: *library
                    .get(b"hrx_status_free_message\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_status_ignore: *library
                    .get(b"hrx_status_ignore\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_gpu_initialize: *library
                    .get(b"hrx_gpu_initialize\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_gpu_device_count: *library
                    .get(b"hrx_gpu_device_count\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_gpu_device_get: *library
                    .get(b"hrx_gpu_device_get\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_create: *library
                    .get(b"hrx_stream_create\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_release: *library
                    .get(b"hrx_stream_release\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_synchronize: *library
                    .get(b"hrx_stream_synchronize\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_buffer_allocate: *library
                    .get(b"hrx_buffer_allocate\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_buffer_release: *library
                    .get(b"hrx_buffer_release\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_fill_buffer: *library
                    .get(b"hrx_stream_fill_buffer\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_copy_buffer: *library
                    .get(b"hrx_stream_copy_buffer\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_synchronous_h2d: *library
                    .get(b"hrx_synchronous_h2d\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_synchronous_d2h: *library
                    .get(b"hrx_synchronous_d2h\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_executable_load_file: *library
                    .get(b"hrx_executable_load_file\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_executable_release: *library
                    .get(b"hrx_executable_release\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_executable_lookup_export_by_name: *library
                    .get(b"hrx_executable_lookup_export_by_name\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_executable_export_info: *library
                    .get(b"hrx_executable_export_info\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_dispatch: *library
                    .get(b"hrx_stream_dispatch\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_status_code: *library
                    .get(b"hrx_status_code\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_device_allocator: *library
                    .get(b"hrx_device_allocator\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_allocator_allocate_buffer: *library
                    .get(b"hrx_allocator_allocate_buffer\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_buffer_get_device_ptr: *library
                    .get(b"hrx_buffer_get_device_ptr\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_device_get_property: *library
                    .get(b"hrx_device_get_property\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_flush: *library
                    .get(b"hrx_stream_flush\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_query: *library
                    .get(b"hrx_stream_query\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
                hrx_stream_update_buffer: *library
                    .get(b"hrx_stream_update_buffer\0")
                    .map_err(|e| crate::Error(e.to_string()))?,
            })
        })();
        // Even a partial loader must not unload HSA/HRX behind a different DSO.
        std::mem::forget(hsa);
        std::mem::forget(library);
        let api = result?;
        lock.file().set_len(0)?;
        lock.file().rewind()?;
        lock.file().write_all(identity.as_bytes())?;
        API.set(api)
            .map_err(|_| crate::Error("runtime loader raced".into()))?;
    }
    Ok(())
}
pub(crate) fn runtime_lock() -> crate::Result<crate::bundle::Lock> {
    let path = std::path::PathBuf::from(format!(
        "/tmp/hrx-{}-{}.lock",
        unsafe { libc::getuid() },
        std::process::id()
    ));
    crate::bundle::Lock::acquire(&path)
}
fn api() -> &'static Api {
    API.get()
        .expect("call Gpu::open or Device::open before raw HRX functions")
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_status_to_string(
    status: Status,
    out_message: *mut *mut c_char,
    out_length: *mut usize,
) -> Status {
    unsafe { (api().hrx_status_to_string)(status, out_message, out_length) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_status_free_message(message: *mut c_char) {
    unsafe { (api().hrx_status_free_message)(message) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_status_ignore(status: Status) {
    unsafe { (api().hrx_status_ignore)(status) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_gpu_initialize(flags: u32) -> Status {
    unsafe { (api().hrx_gpu_initialize)(flags) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_gpu_device_count(count: *mut c_int) -> Status {
    unsafe { (api().hrx_gpu_device_count)(count) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_gpu_device_get(index: c_int, device: *mut Device) -> Status {
    unsafe { (api().hrx_gpu_device_get)(index, device) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_create(device: Device, flags: u32, out_stream: *mut Stream) -> Status {
    unsafe { (api().hrx_stream_create)(device, flags, out_stream) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_release(stream: Stream) {
    unsafe { (api().hrx_stream_release)(stream) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_synchronize(stream: Stream) -> Status {
    unsafe { (api().hrx_stream_synchronize)(stream) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_buffer_allocate(
    stream: Stream,
    size: usize,
    memory_type: u32,
    usage: u32,
    out_buffer: *mut Buffer,
) -> Status {
    unsafe { (api().hrx_buffer_allocate)(stream, size, memory_type, usage, out_buffer) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_buffer_release(buffer: Buffer) {
    unsafe { (api().hrx_buffer_release)(buffer) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_fill_buffer(
    stream: Stream,
    buffer: Buffer,
    offset: usize,
    size: usize,
    pattern: *const c_void,
    pattern_size: usize,
) -> Status {
    unsafe { (api().hrx_stream_fill_buffer)(stream, buffer, offset, size, pattern, pattern_size) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_copy_buffer(
    stream: Stream,
    src: Buffer,
    src_offset: usize,
    dst: Buffer,
    dst_offset: usize,
    size: usize,
) -> Status {
    unsafe { (api().hrx_stream_copy_buffer)(stream, src, src_offset, dst, dst_offset, size) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_synchronous_h2d(
    device: Device,
    host_src: *const c_void,
    dst: Buffer,
    dst_offset: usize,
    size: usize,
) -> Status {
    unsafe { (api().hrx_synchronous_h2d)(device, host_src, dst, dst_offset, size) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_synchronous_d2h(
    device: Device,
    src: Buffer,
    src_offset: usize,
    host_dst: *mut c_void,
    size: usize,
) -> Status {
    unsafe { (api().hrx_synchronous_d2h)(device, src, src_offset, host_dst, size) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_executable_load_file(
    device: Device,
    path: *const c_char,
    target_family: *const c_char,
    target_key: *const c_char,
    out_executable: *mut Executable,
) -> Status {
    unsafe {
        (api().hrx_executable_load_file)(device, path, target_family, target_key, out_executable)
    }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_executable_release(executable: Executable) {
    unsafe { (api().hrx_executable_release)(executable) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_executable_lookup_export_by_name(
    executable: Executable,
    name: *const c_char,
    out_ordinal: *mut u32,
) -> Status {
    unsafe { (api().hrx_executable_lookup_export_by_name)(executable, name, out_ordinal) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_executable_export_info(
    executable: Executable,
    ordinal: u32,
    out_info: *mut ExportInfo,
) -> Status {
    unsafe { (api().hrx_executable_export_info)(executable, ordinal, out_info) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_dispatch(
    stream: Stream,
    executable: Executable,
    ordinal: u32,
    config: *const DispatchConfig,
    constants: *const c_void,
    constants_size: usize,
    bindings: *const BufferRef,
    binding_count: usize,
    flags: u32,
) -> Status {
    unsafe {
        (api().hrx_stream_dispatch)(
            stream,
            executable,
            ordinal,
            config,
            constants,
            constants_size,
            bindings,
            binding_count,
            flags,
        )
    }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_status_code(status: Status) -> c_int {
    unsafe { (api().hrx_status_code)(status) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_device_allocator(device: Device) -> Allocator {
    unsafe { (api().hrx_device_allocator)(device) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_allocator_allocate_buffer(
    allocator: Allocator,
    params: BufferParams,
    size: usize,
    buffer: *mut Buffer,
) -> Status {
    unsafe { (api().hrx_allocator_allocate_buffer)(allocator, params, size, buffer) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_buffer_get_device_ptr(buffer: Buffer, pointer: *mut *mut c_void) -> Status {
    unsafe { (api().hrx_buffer_get_device_ptr)(buffer, pointer) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_device_get_property(
    device: Device,
    property: c_int,
    value: *mut c_void,
    size: usize,
) -> Status {
    unsafe { (api().hrx_device_get_property)(device, property, value, size) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_flush(stream: Stream) -> Status {
    unsafe { (api().hrx_stream_flush)(stream) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_query(stream: Stream, complete: *mut bool) -> Status {
    unsafe { (api().hrx_stream_query)(stream, complete) }
}

/// Raw native call. Caller must satisfy the HRX C header contract and initialize the runtime.
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
#[inline]
pub unsafe fn hrx_stream_update_buffer(
    stream: Stream,
    source: *const c_void,
    size: usize,
    buffer: Buffer,
    offset: usize,
) -> Status {
    unsafe { (api().hrx_stream_update_buffer)(stream, source, size, buffer, offset) }
}

#[allow(non_camel_case_types)]
pub type hrx_status_t = Status;
#[allow(non_camel_case_types)]
pub type hrx_device_t = Device;
#[allow(non_camel_case_types)]
pub type hrx_buffer_t = Buffer;
#[allow(non_camel_case_types)]
pub type hrx_stream_t = Stream;
#[allow(non_camel_case_types)]
pub type hrx_executable_t = Executable;
#[allow(non_camel_case_types)]
pub type hrx_dispatch_config_t = DispatchConfig;
#[allow(non_camel_case_types)]
pub type hrx_buffer_ref_t = BufferRef;
pub const HRX_STATUS_ALREADY_EXISTS: c_int = 6;
pub const HRX_DEVICE_PROPERTY_ARCHITECTURE: c_int = 1;
pub const HRX_MEMORY_TYPE_HOST_VISIBLE: u32 = 0x0000_0002;
pub const HRX_MEMORY_TYPE_DEVICE_LOCAL: u32 = 0x0000_0030;
pub const HRX_BUFFER_USAGE_DEFAULT: u32 = 0x0000_0C03;
pub const HRX_BUFFER_USAGE_MAPPING_SCOPED: u32 = 0x0100_0000;
pub const HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS: u32 = 1;

pub type Graph = *mut c_void;
pub type GraphNode = *mut c_void;
pub type GraphExec = *mut c_void;
#[repr(C)]
pub struct GraphKernel {
    pub executable: Executable,
    pub ordinal: u32,
    pub config: DispatchConfig,
    pub constants: *const c_void,
    pub constants_size: usize,
    pub bindings: *const BufferRef,
    pub binding_count: usize,
    pub flags: u32,
}
#[repr(C)]
pub struct GraphCopy {
    pub src: BufferRef,
    pub dst: BufferRef,
}
#[repr(C)]
pub struct GraphFill {
    pub dst: BufferRef,
    pub pattern: u32,
    pub pattern_size: usize,
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_create(device: Device, flags: u32, graph: *mut Graph) -> Status {
    unsafe { (api().hrx_graph_create)(device, flags, graph) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_release(graph: Graph) {
    unsafe { (api().hrx_graph_release)(graph) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_add_kernel_node(
    graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphKernel,
    node: *mut GraphNode,
) -> Status {
    unsafe { (api().hrx_graph_add_kernel_node)(graph, deps, count, attrs, node) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_add_copy_buffer_node(
    graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphCopy,
    node: *mut GraphNode,
) -> Status {
    unsafe { (api().hrx_graph_add_copy_buffer_node)(graph, deps, count, attrs, node) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_add_fill_buffer_node(
    graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphFill,
    node: *mut GraphNode,
) -> Status {
    unsafe { (api().hrx_graph_add_fill_buffer_node)(graph, deps, count, attrs, node) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_instantiate(graph: Graph, flags: u32, exec: *mut GraphExec) -> Status {
    unsafe { (api().hrx_graph_instantiate)(graph, flags, exec) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_exec_release(exec: GraphExec) {
    unsafe { (api().hrx_graph_exec_release)(exec) }
}

/// # Safety
/// Caller must satisfy the native HRX header contract.
pub unsafe fn hrx_graph_exec_launch(exec: GraphExec, stream: Stream) -> Status {
    unsafe { (api().hrx_graph_exec_launch)(exec, stream) }
}
