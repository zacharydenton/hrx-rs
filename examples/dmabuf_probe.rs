//! Is the GPU->NPU dma-buf bridge blocked by the driver, or only by libhrx's API surface?
//!
//! `hsa_amd_portable_export_dmabuf` refuses locked host pages
//! (`HSA_STATUS_ERROR_INVALID_ALLOCATION`); it exports allocations from an HSA memory pool.
//! libhrx will not hand out a device pointer for a device-local buffer, so that path cannot
//! be driven through the GPU runtime today.
//!
//! This goes around libhrx and allocates from an HSA pool directly, purely to answer the
//! roadmap question: if the export and the XRT import both succeed here, zero-copy needs a
//! small libhrx addition. If the driver refuses, it needs a driver change.
//!
//! Usage: dmabuf_probe <xclbin> [--mib N]

use hrx::npu::raw as dvxrt;

use std::ffi::c_void;

const ARG_A: i32 = 3;

type Status = u32;
const HSA_STATUS_SUCCESS: Status = 0;
/// hsa_device_type_t
const HSA_DEVICE_TYPE_GPU: u32 = 1;
/// hsa_agent_info_t::HSA_AGENT_INFO_DEVICE
const HSA_AGENT_INFO_DEVICE: u32 = 17;
/// hsa_amd_memory_pool_info_t
const POOL_INFO_SEGMENT: u32 = 0;
const POOL_INFO_SIZE: u32 = 2;
const POOL_INFO_ALLOC_ALLOWED: u32 = 5;
/// hsa_amd_segment_t::HSA_AMD_SEGMENT_GLOBAL
const SEGMENT_GLOBAL: u32 = 0;

struct Hsa {
    library: libloading::Library,
}

impl Hsa {
    fn open() -> Result<Self, String> {
        let cache = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
            })
            .ok_or("no cache directory")?;
        let root = cache.join("hrx/runtime");
        let entry = std::fs::read_dir(&root)
            .map_err(|e| format!("{}: {e}", root.display()))?
            .flatten()
            .map(|entry| entry.path().join("libhsa-runtime64.so.1"))
            .find(|path| path.is_file())
            .ok_or_else(|| format!("no libhsa-runtime64.so.1 under {}", root.display()))?;
        let library =
            unsafe { libloading::Library::new(&entry) }.map_err(|e| format!("{entry:?}: {e}"))?;
        Ok(Hsa { library })
    }

    unsafe fn symbol<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, String> {
        unsafe { self.library.get(name) }
            .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
    }
}

/// Collected by the agent/pool iteration callbacks below.
#[derive(Default)]
struct Found {
    agent: u64,
    pool: u64,
    pool_bytes: usize,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (xclbin, mib) = match args.as_slice() {
        [xclbin] => (xclbin.clone(), 4usize),
        [xclbin, flag, value] if flag == "--mib" => (
            xclbin.clone(),
            value.parse().map_err(|_| "--mib expects an integer")?,
        ),
        _ => return Err("usage: dmabuf_probe <xclbin> [--mib N]".into()),
    };
    let bytes = mib * 1024 * 1024;
    let hsa = Hsa::open()?;

    unsafe {
        let init: libloading::Symbol<unsafe extern "C" fn() -> Status> =
            hsa.symbol(b"hsa_init\0")?;
        let status = init();
        if status != HSA_STATUS_SUCCESS {
            return Err(format!("hsa_init returned {status}"));
        }

        // --- find a GPU agent -------------------------------------------------
        let iterate_agents: libloading::Symbol<
            unsafe extern "C" fn(
                unsafe extern "C" fn(u64, *mut c_void) -> Status,
                *mut c_void,
            ) -> Status,
        > = hsa.symbol(b"hsa_iterate_agents\0")?;
        let agent_info: libloading::Symbol<unsafe extern "C" fn(u64, u32, *mut c_void) -> Status> =
            hsa.symbol(b"hsa_agent_get_info\0")?;

        // The callback needs the symbol; pass it through the user-data pointer.
        struct AgentScan<'a> {
            info: &'a dyn Fn(u64, u32, *mut c_void) -> Status,
            found: Found,
        }
        unsafe extern "C" fn on_agent(agent: u64, data: *mut c_void) -> Status {
            let scan = unsafe { &mut *data.cast::<AgentScan>() };
            let mut kind: u32 = 0;
            if (scan.info)(agent, HSA_AGENT_INFO_DEVICE, (&raw mut kind).cast())
                == HSA_STATUS_SUCCESS
                && kind == HSA_DEVICE_TYPE_GPU
                && scan.found.agent == 0
            {
                scan.found.agent = agent;
            }
            HSA_STATUS_SUCCESS
        }
        let info_fn = |agent, key, out| agent_info(agent, key, out);
        let mut scan = AgentScan {
            info: &info_fn,
            found: Found::default(),
        };
        iterate_agents(on_agent, (&raw mut scan).cast());
        if scan.found.agent == 0 {
            return Err("no GPU agent found".into());
        }
        println!("gpu agent: 0x{:x}", scan.found.agent);

        // --- find an allocatable global pool ----------------------------------
        let iterate_pools: libloading::Symbol<
            unsafe extern "C" fn(
                u64,
                unsafe extern "C" fn(u64, *mut c_void) -> Status,
                *mut c_void,
            ) -> Status,
        > = hsa.symbol(b"hsa_amd_agent_iterate_memory_pools\0")?;
        let pool_info: libloading::Symbol<unsafe extern "C" fn(u64, u32, *mut c_void) -> Status> =
            hsa.symbol(b"hsa_amd_memory_pool_get_info\0")?;

        struct PoolScan<'a> {
            info: &'a dyn Fn(u64, u32, *mut c_void) -> Status,
            found: Found,
        }
        unsafe extern "C" fn on_pool(pool: u64, data: *mut c_void) -> Status {
            let scan = unsafe { &mut *data.cast::<PoolScan>() };
            if scan.found.pool != 0 {
                return HSA_STATUS_SUCCESS;
            }
            let mut segment: u32 = u32::MAX;
            let mut allowed: bool = false;
            let mut size: usize = 0;
            (scan.info)(pool, POOL_INFO_SEGMENT, (&raw mut segment).cast());
            (scan.info)(pool, POOL_INFO_ALLOC_ALLOWED, (&raw mut allowed).cast());
            (scan.info)(pool, POOL_INFO_SIZE, (&raw mut size).cast());
            if segment == SEGMENT_GLOBAL && allowed && size > 0 {
                scan.found.pool = pool;
                scan.found.pool_bytes = size;
            }
            HSA_STATUS_SUCCESS
        }
        let pool_fn = |pool, key, out| pool_info(pool, key, out);
        let mut pools = PoolScan {
            info: &pool_fn,
            found: Found::default(),
        };
        iterate_pools(scan.found.agent, on_pool, (&raw mut pools).cast());
        if pools.found.pool == 0 {
            return Err("no allocatable global memory pool on the GPU agent".into());
        }
        println!(
            "gpu memory pool: 0x{:x} ({} MiB)",
            pools.found.pool,
            pools.found.pool_bytes / (1024 * 1024)
        );

        // --- allocate from it and export a dma-buf ----------------------------
        let allocate: libloading::Symbol<
            unsafe extern "C" fn(u64, usize, u32, *mut *mut c_void) -> Status,
        > = hsa.symbol(b"hsa_amd_memory_pool_allocate\0")?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        let status = allocate(pools.found.pool, bytes, 0, &raw mut pointer);
        if status != HSA_STATUS_SUCCESS {
            return Err(format!("hsa_amd_memory_pool_allocate returned {status}"));
        }
        println!("pool allocation: {mib} MiB at {pointer:p}");

        // The route that would make a shared buffer a first-class hrx buffer: allocate it
        // through the GPU runtime, take its device pointer, and export that.
        match hrx::Stream::open() {
            Ok(mut stream) => match stream.allocate_shared(bytes) {
                Ok(buffer) => {
                    stream.fill(buffer.binding(), 0x11).ok();
                    stream.synchronize().ok();
                    match buffer.device_ptr() {
                        Ok(hrx_pointer) => {
                            println!("hrx shared allocation: ok, device ptr {hrx_pointer:p}");
                            let mut fd: i32 = -1;
                            let mut offset: u64 = 0;
                            let export: libloading::Symbol<
                                unsafe extern "C" fn(
                                    *const c_void,
                                    usize,
                                    *mut i32,
                                    *mut u64,
                                ) -> Status,
                            > = hsa.symbol(b"hsa_amd_portable_export_dmabuf\0")?;
                            let status = export(hrx_pointer, bytes, &raw mut fd, &raw mut offset);
                            if status != HSA_STATUS_SUCCESS {
                                println!("  dma-buf export of an hrx buffer: NO (status {status})");
                            } else {
                                println!(
                                    "  dma-buf export of an hrx buffer: YES (fd {fd}, offset {offset})"
                                );
                                match dvxrt::Context::new(0, &xclbin) {
                                    Ok(context) => match context.import_dmabuf(fd, bytes) {
                                        Ok(bo) => {
                                            let mut seen = vec![0u8; 4096];
                                            bo.read(&mut seen).ok();
                                            let wrong = seen.iter().filter(|&&b| b != 0x11).count();
                                            println!(
                                                "  npu reads the hrx buffer: {}",
                                                if wrong == 0 {
                                                    "YES -- one buffer, both engines".into()
                                                } else {
                                                    format!("NO ({wrong}/4096 differ)")
                                                }
                                            );
                                        }
                                        Err(error) => println!("  npu import: NO ({error})"),
                                    },
                                    Err(error) => println!("  npu context: {error}"),
                                }
                            }
                        }
                        Err(error) => println!("hrx shared allocation device ptr: NO ({error})"),
                    }
                }
                Err(error) => println!("hrx shared allocation: NO ({error})"),
            },
            Err(error) => println!("hrx stream unavailable: {error}"),
        }

        let export: libloading::Symbol<
            unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64) -> Status,
        > = hsa.symbol(b"hsa_amd_portable_export_dmabuf\0")?;
        let mut fd: i32 = -1;
        let mut offset: u64 = 0;
        let status = export(pointer, bytes, &raw mut fd, &raw mut offset);
        if status != HSA_STATUS_SUCCESS {
            println!("hsa dma-buf export of pool memory: NO (status {status})");
            return Ok(());
        }
        println!("hsa dma-buf export of pool memory: YES (fd {fd}, offset {offset})");

        // --- and hand that descriptor to the NPU ------------------------------
        match dvxrt::Context::new(0, &xclbin) {
            Ok(context) => {
                let _group = context.group_id(ARG_A).map_err(|e| e.to_string())?;
                match context.import_dmabuf(fd, bytes) {
                    Ok(imported) => {
                        println!("npu import of the GPU dma-buf: YES");
                        // Import alone only proves the descriptor was accepted. Write with
                        // the GPU and read through the NPU's BO to prove it is one buffer.
                        let fill: libloading::Symbol<
                            unsafe extern "C" fn(*mut c_void, u32, usize) -> Status,
                        > = hsa.symbol(b"hsa_amd_memory_fill\0")?;
                        let pattern = 0xa5a5_a5a5u32;
                        let status = fill(pointer, pattern, bytes / 4);
                        if status != HSA_STATUS_SUCCESS {
                            println!("  gpu fill of pool memory failed (status {status})");
                            return Ok(());
                        }
                        let mut seen = vec![0u8; 4096];
                        match imported.read(&mut seen) {
                            Ok(()) => {
                                let wrong = seen.iter().filter(|&&b| b != 0xa5).count();
                                if wrong == 0 {
                                    println!(
                                        "  gpu write -> npu read over dma-buf: YES -- zero-copy confirmed"
                                    );
                                } else {
                                    println!(
                                        "  gpu write -> npu read over dma-buf: NO ({wrong}/4096 bytes differ)"
                                    );
                                }
                            }
                            Err(error) => {
                                println!("  read through the imported BO failed: {error}")
                            }
                        }
                    }
                    Err(error) => println!("npu import of the GPU dma-buf: NO ({error})"),
                }
            }
            Err(error) => println!("npu context: UNAVAILABLE ({error})"),
        }
    }
    Ok(())
}
