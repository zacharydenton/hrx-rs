//! Idle allocation and scratch reuse latency. Run with --release.
use hrx::Stream;
fn main() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let mut direct = Vec::new();
    let mut pooled = Vec::new();
    for i in 0..1100 {
        let start = std::time::Instant::now();
        let buffer = stream.allocate(1024 * 1024)?;
        let allocation_time = start.elapsed();
        drop(buffer);
        let start = std::time::Instant::now();
        let buffer = stream.scratch(1024 * 1024)?;
        let scratch_time = start.elapsed();
        stream.recycle(buffer)?;
        if i >= 100 {
            direct.push(allocation_time);
            pooled.push(scratch_time);
        }
    }
    direct.sort();
    pooled.sort();
    println!(
        "1 MiB idle allocation, 1000 interleaved samples: allocator median {:?}; scratch reuse median {:?}",
        direct[direct.len() / 2],
        pooled[pooled.len() / 2]
    );
    Ok(())
}
