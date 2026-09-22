//! Sweep native XDNA copy working sets. Includes core copy and submission costs;
//! throughput alone does not distinguish MALL caching from other bottlenecks.
//! Usage: mall_probe [seconds-per-size]
use hrx::{
    Result,
    fabric::{Device, Engine, XdnaBinding},
    loom::{Compiler, Specialization},
};
use std::time::{Duration, Instant};
fn main() -> Result<()> {
    let seconds: f64 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "2".into())
        .parse()
        .map_err(|e: std::num::ParseFloatError| hrx::Error::Message(e.to_string()))?;
    if !seconds.is_finite() || !(0.01..=3600.0).contains(&seconds) {
        return Err(hrx::Error::Message("seconds must be in 0.01..=3600".into()));
    }
    let device = Device::open(Engine::Xdna, 0)?;
    let compiler = Compiler::for_target(None, device.target())?;
    println!("working_set_MiB GB/s dispatches/s");
    for mib in [1, 4, 8, 16, 32, 64] {
        let bytes = mib * 1024 * 1024;
        let artifact = compiler
            .module(include_str!("../tests/kernels/copy.xdna.loom"))
            .compile(
                &Specialization::new("copy")
                    .with_config("copy.packets", (bytes / 4096).to_string()),
            )?;
        let input = device
            .fabric()
            .allocate(bytes, std::slice::from_ref(&device))?;
        let output = device
            .fabric()
            .allocate(bytes, std::slice::from_ref(&device))?;
        input.write(0, &vec![0x5a; bytes])?;
        let program = unsafe {
            device.prepare_xdna(
                &artifact,
                1,
                &[
                    XdnaBinding {
                        buffer: &input,
                        offset: 0,
                        length: bytes,
                    },
                    XdnaBinding {
                        buffer: &output,
                        offset: 0,
                        length: bytes,
                    },
                ],
            )
        }?;
        unsafe { program.dispatch() }?.wait()?;
        let mut check = vec![0; bytes];
        output.read(0, &mut check)?;
        assert!(check.iter().all(|&b| b == 0x5a));
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed() < Duration::from_secs_f64(seconds) {
            unsafe { program.dispatch() }?.wait()?;
            count += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "{} {:.3} {:.1}",
            2 * mib,
            2.0 * bytes as f64 * count as f64 / elapsed / 1e9,
            count as f64 / elapsed
        );
    }
    Ok(())
}
