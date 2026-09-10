//! What does a GPU->NPU handoff cost on a shared buffer versus staging a copy?
//!
//! The shared path moves no bytes: handing the buffer over is cache maintenance on an
//! allocation both engines already address. The staged path is what you would write without
//! it -- fill a GPU buffer, copy it into the NPU's, hand that over.
//!
//! Usage: shared_bench <xclbin> [--mib N] [--iters N]

use hrx::npu::Npu;
use std::time::Instant;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() -> hrx::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut xclbin = String::new();
    let mut mib = 64usize;
    let mut iters = 20usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--mib" => {
                mib = args[index + 1].parse().expect("--mib");
                index += 2;
            }
            "--iters" => {
                iters = args[index + 1].parse().expect("--iters");
                index += 2;
            }
            other => {
                xclbin = other.to_string();
                index += 1;
            }
        }
    }
    if xclbin.is_empty() {
        eprintln!("usage: shared_bench <xclbin> [--mib N] [--iters N]");
        std::process::exit(2);
    }
    let bytes = mib * 1024 * 1024;

    let npu = Npu::open(&xclbin)?;
    let group = npu.group_id(3)?;
    let mut shared = npu.alloc(bytes, group)?;
    let mut stream = hrx::Stream::open()?;
    let staged = stream.allocate_shared(bytes)?;
    stream.fill(staged.binding(), 0x7e)?;
    stream.synchronize()?;

    // Zero-copy: the GPU writes in place, then the buffer is handed to the NPU.
    let mut direct = Vec::new();
    for _ in 0..iters {
        shared.fill(0x7e7e_7e7e)?;
        let start = Instant::now();
        let _ = shared.npu();
        direct.push(start.elapsed().as_secs_f64());
    }

    // Staged: the same result, but the bytes are copied into the NPU's buffer first.
    let mut copied = Vec::new();
    for _ in 0..iters {
        stream.fill(staged.binding(), 0x7e)?;
        stream.synchronize()?;
        let start = Instant::now();
        shared.copy_from(&staged)?;
        let _ = shared.npu();
        copied.push(start.elapsed().as_secs_f64());
    }

    let direct_ms = median(direct) * 1e3;
    let copied_ms = median(copied) * 1e3;
    println!("buffer: {mib} MiB, {iters} iterations\n");
    println!("{:<34} {:>10} {:>14}", "handoff", "median ms", "effective GB/s");
    println!("{}", "-".repeat(60));
    println!(
        "{:<34} {:>10.3} {:>14.1}",
        "shared (cache maintenance only)",
        direct_ms,
        bytes as f64 / (direct_ms * 1e-3) / 1e9
    );
    println!(
        "{:<34} {:>10.3} {:>14.1}",
        "staged (device copy, then hand off)",
        copied_ms,
        bytes as f64 / (copied_ms * 1e-3) / 1e9
    );
    println!("\nshared handoff is {:.1}x cheaper", copied_ms / direct_ms);
    Ok(())
}
