//! Exercise every ownership transition of a shared GPU/NPU buffer and check the bytes.
//!
//! Each step hands the same allocation to a different engine and verifies the previous
//! engine's writes are visible. Nothing here copies between GPU and NPU: they address one
//! allocation, and the transitions only do cache maintenance.
//!
//! Usage: shared_roundtrip <xclbin> [--mib N]

use hrx::npu::Npu;

fn check(label: &str, bytes: &[u8], expected: u8) -> bool {
    let wrong = bytes.iter().filter(|&&b| b != expected).count();
    if wrong == 0 {
        println!("  {label}: OK");
    } else {
        println!("  {label}: MISMATCH ({wrong}/{} bytes)", bytes.len());
    }
    wrong == 0
}

fn main() -> hrx::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (xclbin, mib) = match args.as_slice() {
        [xclbin] => (xclbin.clone(), 4usize),
        [xclbin, flag, value] if flag == "--mib" => {
            (xclbin.clone(), value.parse().expect("--mib expects an integer"))
        }
        _ => {
            eprintln!("usage: shared_roundtrip <xclbin> [--mib N]");
            std::process::exit(2);
        }
    };
    let bytes = mib * 1024 * 1024;

    let npu = Npu::open(&xclbin)?;
    let group = npu.group_id(3)?;
    let mut shared = npu.alloc(bytes, group)?;
    println!("shared allocation: {} MiB", shared.len() / (1024 * 1024));
    let mut ok = true;

    // host -> npu: the flush happens inside npu().
    println!("host writes, NPU reads");
    shared.host()?.fill(0x11);
    let bo = shared.npu();
    let mut seen = vec![0u8; 4096];
    bo.read(&mut seen).map_err(hrx::Error::Message)?;
    ok &= check("npu sees host writes", &seen, 0x11);

    // npu -> host: host() invalidates first.
    println!("NPU writes, host reads");
    shared.npu().write(&vec![0x22u8; bytes]).map_err(hrx::Error::Message)?;
    ok &= check("host sees npu writes", &shared.host()?[..4096], 0x22);

    // gpu -> npu: a device-side fill, then straight to the NPU.
    println!("GPU writes, NPU reads");
    shared.fill(0x3333_3333)?;
    let bo = shared.npu();
    bo.read(&mut seen).map_err(hrx::Error::Message)?;
    ok &= check("npu sees gpu writes", &seen, 0x33);

    // A GPU buffer copied in device-side, then read by the NPU.
    println!("GPU buffer copied in, NPU reads");
    let mut stream = hrx::Stream::open()?;
    let staged = stream.allocate_shared(bytes)?;
    stream.fill(staged.binding(), 0x44)?;
    stream.synchronize()?;
    shared.copy_from(&staged)?;
    let bo = shared.npu();
    bo.read(&mut seen).map_err(hrx::Error::Message)?;
    ok &= check("npu sees the copied gpu buffer", &seen, 0x44);

    // And back out to a GPU buffer the runtime can dispatch against.
    println!("NPU writes, copied out to a GPU buffer");
    shared.npu().write(&vec![0x55u8; bytes]).map_err(hrx::Error::Message)?;
    shared.copy_to(&staged)?;
    let mut readback = [0u8; 64];
    stream.read_blocking(staged.binding().slice(0, readback.len())?, &mut readback)?;
    ok &= check("gpu buffer holds the npu result", &readback, 0x55);

    println!("\n{}", if ok { "all transitions correct" } else { "FAILURES above" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
