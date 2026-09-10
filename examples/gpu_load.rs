//! Sustained GPU memory-bandwidth load generator for heterogeneous contention probing.
//!
//! Strix Halo's iGPU and XDNA2 NPU share one LPDDR5X memory controller. This drives a
//! bandwidth-bound stream of device-to-device copies (or fills) for a fixed wall-clock
//! window and reports sustained GB/s, so the same load can be measured alone and while
//! an NPU workload runs concurrently.
//!
//! Emits one JSON object on stdout; progress goes to stderr.
//!
//! Usage: gpu_load [--seconds F] [--mib N] [--mode copy|fill] [--batch N]

use hrx::{Result, Stream};
use std::time::{Duration, Instant};

struct Options {
    seconds: f64,
    mib: usize,
    mode: Mode,
    batch: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Device-to-device copy: reads `size` and writes `size` per iteration.
    Copy,
    /// Fill: writes `size` per iteration, with no read traffic.
    Fill,
}

impl Mode {
    /// Bytes crossing the memory controller for one iteration over a `size`-byte buffer.
    fn traffic(self, size: usize) -> usize {
        match self {
            Mode::Copy => 2 * size,
            Mode::Fill => size,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Copy => "copy",
            Mode::Fill => "fill",
        }
    }
}

fn parse() -> std::result::Result<Options, String> {
    let mut options = Options {
        seconds: 10.0,
        // Far past any cache: this must be a memory-controller measurement, not an L2 one.
        mib: 512,
        mode: Mode::Copy,
        // Enqueue depth per synchronize, so submission overhead does not bound the rate.
        batch: 16,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = || -> std::result::Result<String, String> {
            args.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[index]))
        };
        match args[index].as_str() {
            "--seconds" => options.seconds = value()?.parse().map_err(|e| format!("{e}"))?,
            "--mib" => options.mib = value()?.parse().map_err(|e| format!("{e}"))?,
            "--batch" => options.batch = value()?.parse().map_err(|e| format!("{e}"))?,
            "--mode" => {
                options.mode = match value()?.as_str() {
                    "copy" => Mode::Copy,
                    "fill" => Mode::Fill,
                    other => return Err(format!("unknown mode {other}")),
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
        index += 2;
    }
    if options.mib == 0 || options.batch == 0 || !(options.seconds > 0.0) {
        return Err("seconds, mib and batch must all be positive".into());
    }
    Ok(options)
}

fn main() -> Result<()> {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("gpu_load: {message}");
            std::process::exit(2);
        }
    };
    let size = options.mib * 1024 * 1024;
    let mut stream = Stream::open()?;
    let source = stream.allocate(size)?;
    let destination = stream.allocate(size)?;
    // Touch both allocations so the steady-state loop never pays first-use faults.
    stream.fill(source.binding(), 0x3c)?;
    stream.fill(destination.binding(), 0)?;
    stream.synchronize()?;

    let run_batch = |stream: &mut Stream| -> Result<()> {
        for _ in 0..options.batch {
            match options.mode {
                Mode::Copy => stream.copy(destination.binding(), source.binding())?,
                Mode::Fill => stream.fill(destination.binding(), 0x5a)?,
            }
        }
        stream.synchronize()
    };

    // Warm up outside the measured window: the first batch pays queue and paging costs.
    run_batch(&mut stream)?;

    let window = Duration::from_secs_f64(options.seconds);
    let start = Instant::now();
    let mut iterations = 0usize;
    // Per-batch rates, so the harness can see whether contention was steady or bursty.
    let mut batch_rates = Vec::new();
    while start.elapsed() < window {
        let batch_start = Instant::now();
        run_batch(&mut stream)?;
        let seconds = batch_start.elapsed().as_secs_f64();
        iterations += options.batch;
        let bytes = options.mode.traffic(size) * options.batch;
        batch_rates.push(bytes as f64 / seconds / 1e9);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total = options.mode.traffic(size) as f64 * iterations as f64;
    let mean = total / elapsed / 1e9;
    batch_rates.sort_by(f64::total_cmp);
    let median = batch_rates[batch_rates.len() / 2];
    let low = batch_rates[batch_rates.len() / 20];

    println!(
        concat!(
            r#"{{"device":"gpu","mode":"{}","buffer_mib":{},"batch":{},"#,
            r#""seconds":{:.4},"iterations":{},"gb_s_mean":{:.3},"#,
            r#""gb_s_median":{:.3},"gb_s_p5":{:.3}}}"#
        ),
        options.mode.name(),
        options.mib,
        options.batch,
        elapsed,
        iterations,
        mean,
        median,
        low
    );
    eprintln!(
        "gpu_load: {} {:.1} GB/s mean over {:.1}s ({} iterations)",
        options.mode.name(),
        mean,
        elapsed,
        iterations
    );
    Ok(())
}
