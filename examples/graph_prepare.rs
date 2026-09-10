//! Compare inferred repeated-write dependencies with an explicit GPU chain.
use hrx::{Runtime, Stream, execution::MemoryPlacement};
use std::time::Instant;
fn main() -> hrx::Result<()> {
    let runtime = Runtime::new()?;
    let buffer = runtime.allocate(4096, MemoryPlacement::GpuLocal)?;
    let stream = Stream::open()?;
    let raw = stream.allocate(4096)?;
    for count in [128, 512, 1024, 2048] {
        let begin = Instant::now();
        let mut graph = runtime.graph();
        for _ in 0..count {
            graph.fill(buffer.view(), 0)?;
        }
        let graph = graph.prepare()?;
        let inferred = begin.elapsed();
        let begin = Instant::now();
        let mut native = stream.graph()?;
        let mut last = None;
        for _ in 0..count {
            last = Some(native.fill(last.as_slice(), raw.binding(), 0)?);
        }
        let _native = native.finish()?;
        println!(
            "nodes={count} inferred_us={} explicit_us={}",
            inferred.as_micros(),
            begin.elapsed().as_micros()
        );
        graph.submit()?.wait()?;
    }
    Ok(())
}
