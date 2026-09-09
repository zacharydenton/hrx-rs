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
pub type Event = *mut c_void;
pub const EVENT_FLAG_DISABLE_TIMING: u32 = 2;

/// `hrx_status_is_ok`, which the header defines as `static inline` rather than exporting.
#[inline]
pub fn is_ok(status: Status) -> bool {
    status.is_null()
}

pub const MEMORY_TYPE_DEVICE_LOCAL: u32 = 0x0000_0030;
pub const BUFFER_USAGE_DEFAULT: u32 = 0x0000_0C03;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
/// Immutable native export metadata, valid while its kernel is retained.
pub struct ExportInfo {
    /// Borrowed native export name; do not dereference after dropping the kernel.
    pub name: *const c_char,
    /// Native export flags.
    pub flags: u32,
    /// Required packed scalar byte length for binding dispatch.
    pub constant_byte_length: u32,
    /// Required number of buffer bindings.
    pub binding_count: u32,
    /// Native parameter count.
    pub parameter_count: u32,
    /// Compiled workgroup dimensions; zero denotes an unspecified dimension.
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
macro_rules! native_api {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        struct Api { $($name: unsafe extern "C" fn($($ty),*) -> $ret,)* }
        impl Api {
            unsafe fn resolve(library: &libloading::os::unix::Library) -> crate::Result<Self> {
                Ok(Self {
                    $($name: *unsafe { library.get(concat!(stringify!($name), "\0").as_bytes()) }
                        .map_err(crate::Error::from)?,)*
                })
            }
        }
        $(
            /// Raw native call.
            /// # Safety
            /// Initialize the runtime and satisfy the HRX C header contract.
            #[allow(clippy::too_many_arguments)]
            #[inline]
            pub unsafe fn $name($($arg: $ty),*) -> $ret {
                unsafe { (api().$name)($($arg),*) }
            }
        )*
    };
}
native_api! {
    fn hrx_executable_retain(executable: Executable) -> ();
    fn hrx_event_create(device: Device, flags: u32, out_event: *mut Event) -> Status;
    fn hrx_event_release(event: Event) -> ();
    fn hrx_event_record(event: Event, stream: Stream) -> Status;
    fn hrx_event_query(event: Event, complete: *mut bool) -> Status;
    fn hrx_event_synchronize(event: Event) -> Status;
    fn hrx_stream_wait_event(stream: Stream, event: Event) -> Status;
    fn hrx_status_to_string(status: Status,
    out_message: *mut *mut c_char,
    out_length: *mut usize,) -> Status;
    fn hrx_status_free_message(message: *mut c_char) -> ();
    fn hrx_status_ignore(status: Status) -> ();
    fn hrx_gpu_initialize(flags: u32) -> Status;
    fn hrx_gpu_device_count(count: *mut c_int) -> Status;
    fn hrx_gpu_device_get(index: c_int, device: *mut Device) -> Status;
    fn hrx_stream_create(device: Device, flags: u32, out_stream: *mut Stream) -> Status;
    fn hrx_stream_release(stream: Stream) -> ();
    fn hrx_stream_synchronize(stream: Stream) -> Status;
    fn hrx_buffer_release(buffer: Buffer) -> ();
    fn hrx_stream_fill_buffer(stream: Stream,
    buffer: Buffer,
    offset: usize,
    size: usize,
    pattern: *const c_void,
    pattern_size: usize,) -> Status;
    fn hrx_stream_copy_buffer(stream: Stream,
    src: Buffer,
    src_offset: usize,
    dst: Buffer,
    dst_offset: usize,
    size: usize,) -> Status;
    fn hrx_synchronous_h2d(device: Device,
    host_src: *const c_void,
    dst: Buffer,
    dst_offset: usize,
    size: usize,) -> Status;
    fn hrx_synchronous_d2h(device: Device,
    src: Buffer,
    src_offset: usize,
    host_dst: *mut c_void,
    size: usize,) -> Status;
    fn hrx_executable_load_data(device: Device,
    data: *const c_void,
    size: usize,
    family: *const c_char,
    key: *const c_char,
    executable: *mut Executable,) -> Status;
    fn hrx_executable_load_file(device: Device,
    path: *const c_char,
    target_family: *const c_char,
    target_key: *const c_char,
    out_executable: *mut Executable,) -> Status;
    fn hrx_executable_release(executable: Executable) -> ();
    fn hrx_executable_lookup_export_by_name(executable: Executable,
    name: *const c_char,
    out_ordinal: *mut u32,) -> Status;
    fn hrx_executable_export_info(executable: Executable,
    ordinal: u32,
    out_info: *mut ExportInfo,) -> Status;
    fn hrx_stream_dispatch(stream: Stream,
    executable: Executable,
    ordinal: u32,
    config: *const DispatchConfig,
    constants: *const c_void,
    constants_size: usize,
    bindings: *const BufferRef,
    binding_count: usize,
    flags: u32,) -> Status;
    fn hrx_status_code(status: Status) -> c_int;
    fn hrx_device_allocator(device: Device) -> Allocator;
    fn hrx_allocator_allocate_buffer(allocator: Allocator,
    params: BufferParams,
    size: usize,
    buffer: *mut Buffer,) -> Status;
    fn hrx_buffer_get_device_ptr(buffer: Buffer, pointer: *mut *mut c_void) -> Status;
    fn hrx_device_get_property(device: Device,
    property: c_int,
    value: *mut c_void,
    size: usize,) -> Status;
    fn hrx_stream_flush(stream: Stream) -> Status;
    fn hrx_stream_query(stream: Stream, complete: *mut bool) -> Status;
    fn hrx_graph_create(device: Device, flags: u32, graph: *mut Graph) -> Status;
    fn hrx_graph_release(graph: Graph) -> ();
    fn hrx_graph_add_kernel_node(graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphKernel,
    node: *mut GraphNode,) -> Status;
    fn hrx_graph_add_copy_buffer_node(graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphCopy,
    node: *mut GraphNode,) -> Status;
    fn hrx_graph_add_fill_buffer_node(graph: Graph,
    deps: *const GraphNode,
    count: usize,
    attrs: *const GraphFill,
    node: *mut GraphNode,) -> Status;
    fn hrx_graph_instantiate(graph: Graph, flags: u32, exec: *mut GraphExec) -> Status;
    fn hrx_graph_exec_release(exec: GraphExec) -> ();
    fn hrx_graph_exec_launch(exec: GraphExec, stream: Stream) -> Status;
}
static API: std::sync::OnceLock<Api> = std::sync::OnceLock::new();
pub(crate) fn load() -> crate::Result<()> {
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if API.get().is_some() {
        return Ok(());
    }
    let _guard = INIT
        .lock()
        .map_err(|_| crate::Error::Message("runtime loader poisoned".into()))?;
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
        .ok_or_else(|| crate::Error::Message("invalid proc stat".into()))?
        .1
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| crate::Error::Message("missing process start time".into()))?;
    let identity = format!("{start} {}", path.display());
    if selected.starts_with(&format!("{start} ")) && selected != identity {
        return Err(crate::Error::Message(format!(
            "another model already loaded {selected}; select one HRX_RUNTIME_DIR for the process"
        )));
    }
    unsafe {
        use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
        let provider = std::env::var_os("IREE_HAL_AMDGPU_LIBHSA_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| directory.join("libhsa-runtime64.so.1"));
        let provider = std::fs::canonicalize(provider)?;
        let hsa = Library::open(Some(&provider), RTLD_NOW | RTLD_LOCAL).map_err(|e| {
            crate::Error::from(e).context(format!("loading {}", provider.display()))
        })?;
        // No library-controlled environment mutation, and no search-path changes.
        let library = Library::open(Some(&path), RTLD_NOW | RTLD_LOCAL)
            .map_err(|e| crate::Error::from(e).context(format!("loading {}", path.display())))?;
        let result = Api::resolve(&library);
        // Even a partial loader must not unload HSA/HRX behind a different DSO.
        std::mem::forget(hsa);
        std::mem::forget(library);
        let api = result?;
        lock.file().set_len(0)?;
        lock.file().rewind()?;
        lock.file().write_all(identity.as_bytes())?;
        API.set(api)
            .map_err(|_| crate::Error::Message("runtime loader raced".into()))?;
    }
    Ok(())
}
static RUNTIME_LOCK_PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
extern "C" fn cleanup_runtime_lock() {
    if let Some(path) = RUNTIME_LOCK_PATH.get() {
        let _ = std::fs::remove_file(path);
    }
}

fn private_runtime_directory(path: &std::path::Path) -> crate::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err(crate::Error::Message(
            "runtime lock directory must be owned by this user and private".into(),
        ));
    }
    Ok(())
}
pub(crate) fn runtime_lock() -> crate::Result<crate::bundle::Lock> {
    use std::os::unix::fs::MetadataExt;
    if let Some(path) = RUNTIME_LOCK_PATH.get() {
        return crate::bundle::Lock::acquire(path);
    }
    let directory = std::env::var_os("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_absolute() && private_runtime_directory(p).is_ok())
        .unwrap_or_else(|| {
            std::path::PathBuf::from(format!("/tmp/hrx-{}", unsafe { libc::geteuid() }))
        });
    private_runtime_directory(&directory)?;
    let path = directory.join(format!("hrx-{}.lock", std::process::id()));
    let mut lock = crate::bundle::Lock::acquire(&path)?;
    let metadata = lock.file().metadata()?;
    if !metadata.is_file()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err(crate::Error::Message(
            "runtime lock must be a private file owned by this user".into(),
        ));
    }
    if RUNTIME_LOCK_PATH.set(path).is_ok() {
        unsafe {
            libc::atexit(cleanup_runtime_lock);
        }
    }
    Ok(lock)
}
fn api() -> &'static Api {
    API.get()
        .expect("open a Device or Stream before raw HRX functions")
}

pub const STATUS_ALREADY_EXISTS: c_int = 6;
pub const DEVICE_PROPERTY_ARCHITECTURE: c_int = 1;
pub const MEMORY_TYPE_HOST_VISIBLE: u32 = 0x0000_0002;
pub const MEMORY_TYPE_HOST_COHERENT: u32 = 0x0000_0004;
pub const MEMORY_TYPE_HOST_LOCAL: u32 = 0x0000_0040 | MEMORY_TYPE_HOST_VISIBLE;
pub const MEMORY_TYPE_DEVICE_VISIBLE: u32 = 0x0000_0010;
pub const MEMORY_ACCESS_ALL: u16 = 7;
pub const BUFFER_USAGE_MAPPING_SCOPED: u32 = 0x0100_0000;
// The pinned native implementation ignores this field and uses executable metadata.
pub const SUBGROUP_SIZE_FROM_EXECUTABLE: u32 = 0;

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_lock_directory_rejects_public_modes_and_symlinks() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("private");
        private_runtime_directory(&directory).unwrap();
        assert_eq!(
            std::fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let link = temporary.path().join("link");
        symlink(&directory, &link).unwrap();
        assert!(private_runtime_directory(&link).is_err());
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(private_runtime_directory(&directory).is_err());
    }
}
