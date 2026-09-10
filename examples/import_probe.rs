//! Does the GPU write straight into host pages we own? The zero-copy prerequisite.
//!
//! Allocates page-aligned host memory, imports it into the runtime with no copy, has the
//! GPU fill it, and then reads the *original host pointer* back. If the pattern is there,
//! the device wrote into our pages in place -- which is what makes the same allocation
//! usable as an XRT BO for the NPU at the same time.
//!
//! Usage: import_probe [--mib N]

use hrx::{Result, Stream};
use std::alloc::{Layout, alloc_zeroed, dealloc};

const PAGE: usize = 4096;
const PATTERN: u8 = 0xa5;

fn main() -> Result<()> {
    let mib: usize = match std::env::args().collect::<Vec<_>>().as_slice() {
        [_] => 16,
        [_, flag, value] if flag == "--mib" => value.parse().expect("--mib expects an integer"),
        _ => {
            eprintln!("usage: import_probe [--mib N]");
            std::process::exit(2);
        }
    };
    let bytes = mib * 1024 * 1024;
    let layout = Layout::from_size_align(bytes, PAGE).expect("page-aligned layout");

    // Owned for the whole scope below; the imported buffer never frees it.
    let host = unsafe { alloc_zeroed(layout) };
    assert!(!host.is_null(), "host allocation failed");
    let result = (|| -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = match unsafe { stream.import_host(host.cast(), bytes) } {
            Ok(buffer) => buffer,
            Err(error) => {
                println!("import: UNSUPPORTED ({error})");
                return Ok(());
            }
        };
        println!("import: ok ({mib} MiB at {host:p})");

        // Prove the host pages start clear, so a match afterwards can only be the GPU.
        let before = unsafe { std::slice::from_raw_parts(host, bytes) };
        assert!(before.iter().all(|&b| b == 0), "host memory was not zeroed");

        stream.fill(buffer.binding(), PATTERN)?;
        stream.synchronize()?;

        // Read the ORIGINAL host pointer -- no download, no staging buffer.
        let after = unsafe { std::slice::from_raw_parts(host, bytes) };
        let wrong = after.iter().filter(|&&b| b != PATTERN).count();
        if wrong == 0 {
            println!("gpu fill visible on host pointer: YES (zero-copy confirmed)");
        } else {
            println!("gpu fill visible on host pointer: NO ({wrong}/{bytes} bytes unwritten)");
        }

        // A device-side read of the same buffer must also see host writes.
        let mirror = stream.allocate(bytes)?;
        unsafe { std::slice::from_raw_parts_mut(host, bytes) }.fill(0x3c);
        stream.copy(mirror.binding(), buffer.binding())?;
        let mut check = [0u8; 64];
        stream.read_blocking(mirror.binding().slice(0, check.len())?, &mut check)?;
        let coherent = check.iter().all(|&b| b == 0x3c);
        println!(
            "host write visible to device: {}",
            if coherent { "YES" } else { "NO" }
        );
        Ok(())
    })();
    unsafe { dealloc(host, layout) };
    result
}
