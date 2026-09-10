//! Can one host allocation back both engines at once, with no copy between them?
//!
//! Strix Halo puts the gfx1151 iGPU and the XDNA2 NPU on the same physical memory, but
//! each driver pins pages into its own IOMMU domain: the GPU through KFD, the NPU through
//! amdxdna's DRM node. Physical sharing is not the question -- whether both drivers will
//! map the *same* user pages is.
//!
//! This allocates page-aligned host memory once, imports it into the GPU runtime
//! (`hrx_allocator_import_buffer`) and into XRT as a userptr BO, then writes with the GPU
//! and reads through the NPU's BO. If the pattern survives, a GPU->NPU handoff costs a
//! cache flush instead of a copy through system memory.
//!
//! Usage: unified_probe <xclbin> [--mib N]

use hrx::npu::raw as dvxrt;

use hrx::{Result, Stream};
use std::alloc::{Layout, alloc_zeroed, dealloc};

const PAGE: usize = 4096;
/// config3's A argument, the binding a real GEMM would stream from.
const ARG_A: i32 = 3;
const GPU_PATTERN: u8 = 0xa5;
const NPU_PATTERN: u8 = 0x5c;

/// Allocate a GPU buffer, fill it, and export its allocation as a dma-buf descriptor.
///
/// Returns the descriptor, its length, and the buffer itself -- which must outlive the fd.
fn export_dmabuf(
    pointer: *mut std::ffi::c_void,
    bytes: usize,
) -> std::result::Result<(i32, usize), String> {
    // The bundle's HSA runtime is already resident; this resolves the symbol in it rather
    // than loading a second copy.
    let library = hsa_library()?;
    let export: libloading::Symbol<
        unsafe extern "C" fn(*const std::ffi::c_void, usize, *mut i32, *mut u64) -> u32,
    > = unsafe { library.get(b"hsa_amd_portable_export_dmabuf\0") }
        .map_err(|e| format!("hsa_amd_portable_export_dmabuf: {e}"))?;

    let mut fd = -1i32;
    let mut offset = 0u64;
    let status = unsafe { export(pointer.cast_const(), bytes, &mut fd, &mut offset) };
    if status != 0 {
        return Err(format!("hsa_amd_portable_export_dmabuf returned {status}"));
    }
    if offset != 0 {
        return Err(format!(
            "dma-buf carries a nonzero offset ({offset}); unsupported here"
        ));
    }
    Ok((fd, bytes))
}

fn hsa_library() -> std::result::Result<libloading::Library, String> {
    let cache = dirs_cache().ok_or("no cache directory")?;
    let pattern = cache.join("hrx/runtime");
    let entries = std::fs::read_dir(&pattern).map_err(|e| format!("{}: {e}", pattern.display()))?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("libhsa-runtime64.so.1");
        if candidate.is_file() {
            return unsafe { libloading::Library::new(&candidate) }
                .map_err(|e| format!("{}: {e}", candidate.display()));
        }
    }
    Err(format!(
        "no libhsa-runtime64.so.1 under {}",
        pattern.display()
    ))
}

fn dirs_cache() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
        })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (xclbin, mib) = match args.as_slice() {
        [xclbin] => (xclbin.clone(), 4usize),
        [xclbin, flag, value] if flag == "--mib" => (
            xclbin.clone(),
            value.parse().expect("--mib expects an integer"),
        ),
        _ => {
            eprintln!("usage: unified_probe <xclbin> [--mib N]");
            std::process::exit(2);
        }
    };
    let bytes = mib * 1024 * 1024;
    let layout = Layout::from_size_align(bytes, PAGE).expect("page-aligned layout");
    let host = unsafe { alloc_zeroed(layout) };
    assert!(!host.is_null(), "host allocation failed");

    let result = (|| -> Result<()> {
        println!("host allocation: {mib} MiB at {host:p}");

        let context = match unsafe { dvxrt::Context::new(0, &xclbin) } {
            Ok(context) => context,
            Err(error) => {
                println!("npu context: UNAVAILABLE ({error})");
                return Ok(());
            }
        };
        let group = context.group_id(ARG_A).expect("group id for A");

        // 1. Can amdxdna pin pages we allocated? Reported either way -- if it can, either
        //    runtime may own the allocation; if it cannot, only the NPU's allocator may.
        match unsafe { context.import_bo(host.cast(), bytes, dvxrt::BoKind::HostOnly, group) } {
            Ok(_) => println!("npu userptr import (host-owned pages): YES"),
            Err(error) => println!("npu userptr import (host-owned pages): NO ({error})"),
        }

        // 2. The direction that matters: let the NPU's allocator own the pages -- it is the
        //    more constrained of the two -- and import its host mapping into the GPU runtime.
        let bo = context
            .alloc_bo(bytes, dvxrt::BoKind::HostOnly, group)
            .expect("allocate an NPU BO");
        let shared = bo.map().expect("map the NPU BO");
        println!("npu allocation mapped at {shared:p}");
        assert_eq!(shared as usize % PAGE, 0, "NPU mapping is not page-aligned");

        let mut stream = Stream::open()?;
        let imported = unsafe { stream.import_host(shared.cast(), bytes) };
        match &imported {
            Ok(_) => println!("gpu import of the NPU mapping: YES"),
            Err(error) => println!("gpu import of the NPU mapping: NO ({error})"),
        }

        // 3. If the shared mapping took, check that a GPU write lands in the NPU's BO.
        if let Ok(buffer) = &imported {
            stream.fill(buffer.binding(), GPU_PATTERN)?;
            stream.synchronize()?;
            bo.sync(false, bytes).expect("invalidate for device reads");
            let mut seen = vec![0u8; 4096];
            bo.read(&mut seen).expect("read through the NPU BO");
            let wrong = seen.iter().filter(|&&b| b != GPU_PATTERN).count();
            println!(
                "gpu write -> npu read: {}",
                if wrong == 0 {
                    "YES".into()
                } else {
                    format!("NO ({wrong} bytes differ)")
                }
            );
        }

        // 3b. The dma-buf route: the GPU allocates, HSA exports the allocation as a
        //     dma-buf, and XRT imports that descriptor. Neither runtime copies, and unlike
        //     the host-pointer routes above neither driver has to pin the other's pages.
        // The GPU-side import above HSA-locked these pages, so they are exportable.
        let locked = unsafe { stream.import_host(host.cast(), bytes) };
        let export = match &locked {
            Ok(buffer) => {
                stream.fill(buffer.binding(), GPU_PATTERN)?;
                stream.synchronize()?;
                export_dmabuf(host.cast(), bytes)
            }
            Err(error) => Err(format!("host pages are not GPU-resident: {error}")),
        };
        match export {
            Ok((fd, size)) => {
                println!("hsa dma-buf export of a GPU buffer: YES (fd {fd})");
                match unsafe { context.import_dmabuf(fd, size) } {
                    Ok(imported) => {
                        println!("npu import of the GPU dma-buf: YES");
                        let mut seen = vec![0u8; 4096];
                        match imported.read(&mut seen) {
                            Ok(()) => {
                                let wrong = seen.iter().filter(|&&b| b != GPU_PATTERN).count();
                                println!(
                                    "  gpu write -> npu read over dma-buf: {}",
                                    if wrong == 0 {
                                        "YES".into()
                                    } else {
                                        format!("NO ({wrong} bytes differ)")
                                    }
                                );
                            }
                            Err(error) => {
                                println!("  read through the imported BO failed: {error}")
                            }
                        }
                    }
                    Err(error) => println!("npu import of the GPU dma-buf: NO ({error})"),
                }
            }
            Err(error) => println!("hsa dma-buf export of a GPU buffer: NO ({error})"),
        }

        // 4. And the reverse: the NPU's BO writes, the GPU reads.
        if let Ok(buffer) = &imported {
            bo.write(&vec![NPU_PATTERN; bytes])
                .expect("write through the NPU BO");
            let mirror = stream.allocate(bytes)?;
            stream.copy(mirror.binding(), buffer.binding())?;
            let mut check = [0u8; 64];
            stream.read_blocking(mirror.binding().slice(0, check.len())?, &mut check)?;
            let back = check.iter().all(|&b| b == NPU_PATTERN);
            println!("npu write -> gpu read: {}", if back { "YES" } else { "NO" });
        }
        Ok(())
    })();
    unsafe { dealloc(host, layout) };
    result
}
