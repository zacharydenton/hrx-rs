//! The slice of the HSA runtime needed to allocate memory the NPU can import.
//!
//! The GPU runtime cannot do this itself: only GPU-agent pool allocations are exportable
//! as dma-buf, `hrx_buffer_get_device_ptr` refuses device-local buffers, and libhrx has no
//! export entry point at all. So shared allocations come from HSA directly, using the same
//! runtime libhrx already has resident.

use std::ffi::c_void;

pub type Status = u32;
pub const SUCCESS: Status = 0;

const DEVICE_TYPE_GPU: u32 = 1;
const AGENT_INFO_DEVICE: u32 = 17;
const POOL_INFO_SEGMENT: u32 = 0;
const POOL_INFO_SIZE: u32 = 2;
const POOL_INFO_ALLOC_ALLOWED: u32 = 5;
const SEGMENT_GLOBAL: u32 = 0;

/// Describes an HSA failure without pulling in the runtime's own error strings.
pub fn describe(operation: &str, status: Status) -> String {
    let meaning = match status {
        0x1000 => "generic error",
        0x1001 => "invalid argument",
        0x1003 => "invalid allocation",
        0x1004 => "invalid agent",
        0x1008 => "out of resources",
        _ => "see hsa_status_t",
    };
    format!("{operation} failed: HSA status {status:#x} ({meaning})")
}

/// The bundle's HSA runtime, plus the GPU agent and pool shared allocations come from.
pub struct Hsa {
    library: libloading::Library,
    pub agent: u64,
    pub pool: u64,
}

macro_rules! symbol {
    ($library:expr, $name:literal, $signature:ty) => {{
        // Every call site is already inside an `unsafe` block.
        let symbol: libloading::Symbol<$signature> = $library
            .get(concat!($name, "\0").as_bytes())
            .map_err(|e| format!("{}: {e}", $name))?;
        symbol
    }};
}

impl Hsa {
    /// Load the resident HSA runtime and locate a GPU agent with an allocatable pool.
    pub fn open() -> Result<Self, String> {
        let library = load_resident()?;
        unsafe {
            let init = symbol!(library, "hsa_init", unsafe extern "C" fn() -> Status);
            // ALREADY_INITIALIZED is fine: libhrx may have got here first.
            init();
        }
        let agent = find_gpu_agent(&library)?;
        let pool = find_pool(&library, agent)?;
        Ok(Hsa { library, agent, pool })
    }

    /// Allocate `bytes` from the GPU pool and make it reachable from host and agent.
    ///
    /// The pointer is page-aligned and exportable; the caller owns it until [`Hsa::free`].
    pub fn allocate(&self, bytes: usize) -> Result<*mut c_void, String> {
        unsafe {
            let allocate = symbol!(
                self.library,
                "hsa_amd_memory_pool_allocate",
                unsafe extern "C" fn(u64, usize, u32, *mut *mut c_void) -> Status
            );
            let mut pointer: *mut c_void = std::ptr::null_mut();
            let status = allocate(self.pool, bytes, 0, &raw mut pointer);
            if status != SUCCESS {
                return Err(describe("hsa_amd_memory_pool_allocate", status));
            }
            // Without this the agent may not reach the allocation for copies.
            let allow = symbol!(
                self.library,
                "hsa_amd_agents_allow_access",
                unsafe extern "C" fn(u32, *const u64, *const c_void, *const c_void) -> Status
            );
            let agents = [self.agent];
            allow(1, agents.as_ptr(), std::ptr::null(), pointer);
            Ok(pointer)
        }
    }

    pub fn free(&self, pointer: *mut c_void) {
        unsafe {
            if let Ok(free) = self
                .library
                .get::<unsafe extern "C" fn(*mut c_void) -> Status>(b"hsa_amd_memory_pool_free\0")
            {
                free(pointer);
            }
        }
    }

    /// Export the allocation as a dma-buf descriptor the NPU's driver can import.
    pub fn export_dmabuf(&self, pointer: *mut c_void, bytes: usize) -> Result<i32, String> {
        unsafe {
            let export = symbol!(
                self.library,
                "hsa_amd_portable_export_dmabuf",
                unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64) -> Status
            );
            let mut fd: i32 = -1;
            let mut offset: u64 = 0;
            let status = export(pointer, bytes, &raw mut fd, &raw mut offset);
            if status != SUCCESS {
                return Err(describe("hsa_amd_portable_export_dmabuf", status));
            }
            if offset != 0 {
                return Err(format!("dma-buf has a nonzero offset ({offset})"));
            }
            Ok(fd)
        }
    }

    /// Blocking copy between any two HSA-reachable pointers, performed by the GPU.
    ///
    /// This is the GPU<->shared transfer: it stays on the device rather than bouncing
    /// through a host staging buffer.
    pub fn copy(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<(), String> {
        unsafe {
            let copy = symbol!(
                self.library,
                "hsa_memory_copy",
                unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> Status
            );
            let status = copy(dst, src, bytes);
            if status != SUCCESS {
                return Err(describe("hsa_memory_copy", status));
            }
            Ok(())
        }
    }

    /// Fill the allocation with a repeating 32-bit pattern, on the device.
    pub fn fill(&self, pointer: *mut c_void, value: u32, words: usize) -> Result<(), String> {
        unsafe {
            let fill = symbol!(
                self.library,
                "hsa_amd_memory_fill",
                unsafe extern "C" fn(*mut c_void, u32, usize) -> Status
            );
            let status = fill(pointer, value, words);
            if status != SUCCESS {
                return Err(describe("hsa_amd_memory_fill", status));
            }
            Ok(())
        }
    }
}

/// Load the HSA runtime that ships in the prepared bundle, not a system copy: it must be
/// the same instance libhrx is using or the agent handles would not match.
fn load_resident() -> Result<libloading::Library, String> {
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
        })
        .ok_or("no cache directory for the HRX bundle")?;
    let root = cache.join("hrx/runtime");
    let candidate = std::fs::read_dir(&root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .flatten()
        .map(|entry| entry.path().join("libhsa-runtime64.so.1"))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("no libhsa-runtime64.so.1 under {}", root.display()))?;
    unsafe { libloading::Library::new(&candidate) }.map_err(|e| format!("{candidate:?}: {e}"))
}

/// Iteration state shared with the C callbacks below.
struct Scan<'a> {
    info: &'a dyn Fn(u64, u32, *mut c_void) -> Status,
    handle: u64,
}

fn find_gpu_agent(library: &libloading::Library) -> Result<u64, String> {
    unsafe {
        let iterate = symbol!(
            library,
            "hsa_iterate_agents",
            unsafe extern "C" fn(
                unsafe extern "C" fn(u64, *mut c_void) -> Status,
                *mut c_void,
            ) -> Status
        );
        let info = symbol!(
            library,
            "hsa_agent_get_info",
            unsafe extern "C" fn(u64, u32, *mut c_void) -> Status
        );
        unsafe extern "C" fn visit(agent: u64, data: *mut c_void) -> Status {
            let scan = unsafe { &mut *data.cast::<Scan>() };
            if scan.handle != 0 {
                return SUCCESS;
            }
            let mut kind: u32 = 0;
            if (scan.info)(agent, AGENT_INFO_DEVICE, (&raw mut kind).cast()) == SUCCESS
                && kind == DEVICE_TYPE_GPU
            {
                scan.handle = agent;
            }
            SUCCESS
        }
        let call = |agent, key, out| info(agent, key, out);
        let mut scan = Scan { info: &call, handle: 0 };
        iterate(visit, (&raw mut scan).cast());
        if scan.handle == 0 {
            return Err("no GPU agent found".into());
        }
        Ok(scan.handle)
    }
}

fn find_pool(library: &libloading::Library, agent: u64) -> Result<u64, String> {
    unsafe {
        let iterate = symbol!(
            library,
            "hsa_amd_agent_iterate_memory_pools",
            unsafe extern "C" fn(
                u64,
                unsafe extern "C" fn(u64, *mut c_void) -> Status,
                *mut c_void,
            ) -> Status
        );
        let info = symbol!(
            library,
            "hsa_amd_memory_pool_get_info",
            unsafe extern "C" fn(u64, u32, *mut c_void) -> Status
        );
        unsafe extern "C" fn visit(pool: u64, data: *mut c_void) -> Status {
            let scan = unsafe { &mut *data.cast::<Scan>() };
            if scan.handle != 0 {
                return SUCCESS;
            }
            let mut segment: u32 = u32::MAX;
            let mut allowed = false;
            let mut size: usize = 0;
            (scan.info)(pool, POOL_INFO_SEGMENT, (&raw mut segment).cast());
            (scan.info)(pool, POOL_INFO_ALLOC_ALLOWED, (&raw mut allowed).cast());
            (scan.info)(pool, POOL_INFO_SIZE, (&raw mut size).cast());
            if segment == SEGMENT_GLOBAL && allowed && size > 0 {
                scan.handle = pool;
            }
            SUCCESS
        }
        let call = |pool, key, out| info(pool, key, out);
        let mut scan = Scan { info: &call, handle: 0 };
        iterate(agent, visit, (&raw mut scan).cast());
        if scan.handle == 0 {
            return Err("no allocatable global memory pool on the GPU agent".into());
        }
        Ok(scan.handle)
    }
}
