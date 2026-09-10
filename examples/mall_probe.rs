//! Does NPU traffic hit Strix Halo's 32 MB MALL, or does it always go to DRAM?
//!
//! The GPU clearly benefits from a MALL-resident working set (measured: ~260 GB/s inside
//! it versus ~162 GB/s from DRAM). Whether the NPU does was unanswerable with a GEMM: at
//! ~1200 FLOP/byte it is compute-bound and would look identical either way.
//!
//! This drives `passthrough_dmas` instead -- shim → memtile → shim, no compute tile at
//! all, so throughput is purely a function of where the bytes come from. Sweeping the
//! working set across the 32 MB MALL boundary should show a knee if the NPU allocates in
//! it, and a flat line if its DMA bypasses MALL for DRAM.
//!
//! Usage: mall_probe <dir-with-nN.xclbin/nN.bin> [--seconds F]

use std::time::{Duration, Instant};

/// MLIR_AIE kernel argument indices.
const ARG_INSTS: i32 = 1;
const ARG_A: i32 = 3;
const ARG_B: i32 = 4;
const ARG_C: i32 = 5;
/// The design leaves its second input unused; it still needs a bound BO.
const UNUSED_BYTES: usize = 4096;
const TIMEOUT_MS: u32 = 10_000;

fn measure(dir: &std::path::Path, n: usize, seconds: f64) -> Result<(f64, f64), String> {
    let xclbin = dir.join(format!("n{n}.xclbin"));
    let insts_path = dir.join(format!("n{n}.bin"));
    let insts_bytes = std::fs::read(&insts_path).map_err(|e| format!("{insts_path:?}: {e}"))?;
    let bytes = n * 4;

    let context = dvxrt::Context::new(0, xclbin.to_str().ok_or("non-UTF-8 path")?)?;
    let insts = context.alloc_bo(
        insts_bytes.len(),
        dvxrt::BoKind::Cacheable,
        context.group_id(ARG_INSTS)?,
    )?;
    insts.write(&insts_bytes)?;
    let a = context.alloc_bo(bytes, dvxrt::BoKind::HostOnly, context.group_id(ARG_A)?)?;
    let b = context.alloc_bo(
        UNUSED_BYTES,
        dvxrt::BoKind::HostOnly,
        context.group_id(ARG_B)?,
    )?;
    let c = context.alloc_bo(bytes, dvxrt::BoKind::HostOnly, context.group_id(ARG_C)?)?;
    a.write(&vec![0x5au8; bytes])?;

    let dispatch = || -> Result<(), String> {
        let handle = unsafe { context.run_start(&insts, insts_bytes.len() as u32, &a, &b, &c) }?;
        let state = handle.wait(TIMEOUT_MS)?;
        // ERT_CMD_STATE_COMPLETED
        if state != 4 {
            return Err(format!("dispatch returned state {state}"));
        }
        Ok(())
    };

    // Warm up: the first dispatch pays context and paging costs.
    dispatch()?;
    // Confirm the design actually moved the bytes before timing it.
    let mut check = [0u8; 64];
    c.read(&mut check)?;
    if check.iter().any(|&byte| byte != 0x5a) {
        return Err("passthrough did not copy its input".into());
    }

    let window = Duration::from_secs_f64(seconds);
    let start = Instant::now();
    let mut dispatches = 0usize;
    while start.elapsed() < window {
        dispatch()?;
        dispatches += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    // Each pass reads the input and writes the output.
    let moved = 2.0 * bytes as f64 * dispatches as f64;
    Ok((moved / elapsed / 1e9, dispatches as f64 / elapsed))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, seconds) = match args.as_slice() {
        [dir] => (dir.clone(), 4.0f64),
        [dir, flag, value] if flag == "--seconds" => {
            (dir.clone(), value.parse().map_err(|_| "--seconds")?)
        }
        _ => return Err("usage: mall_probe <dir> [--seconds F]".into()),
    };
    let dir = std::path::PathBuf::from(dir);

    // Working set is input + output, so the MALL boundary falls at n = 4 Mi elements.
    let sizes = [262_144usize, 1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216, 33_554_432];
    println!(
        "{:>12} {:>12} {:>10} {:>12} {:>10}",
        "elements", "working set", "vs MALL", "GB/s", "disp/s"
    );
    println!("{}", "-".repeat(60));
    for n in sizes {
        let working_set = 2 * n * 4;
        let mib = working_set / (1024 * 1024);
        let position = if working_set <= 32 * 1024 * 1024 { "fits" } else { "exceeds" };
        match measure(&dir, n, seconds) {
            Ok((gb_s, dispatches)) => println!(
                "{n:>12} {:>9} MiB {position:>10} {gb_s:>12.1} {dispatches:>10.1}",
                mib
            ),
            Err(error) => println!("{n:>12} {:>9} MiB {position:>10}   error: {error}", mib),
        }
    }
    Ok(())
}
