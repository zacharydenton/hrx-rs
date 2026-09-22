//! Warm end-to-end latency for a real NPU passthrough pipeline.
//! Usage: shared_bench [bytes]
use hrx::{
    Result,
    benchmark::percentile,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
};
use std::time::Instant;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 2 {
        return Err(hrx::Error::Message("usage: shared_bench [bytes]".into()));
    }
    let bytes = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("1048576")
        .parse::<usize>()
        .map_err(|e| hrx::Error::Message(e.to_string()))?;
    if bytes == 0 || bytes % 4096 != 0 || bytes > 64 * 1024 * 1024 {
        return Err(hrx::Error::Message(
            "bytes must be a multiple of 4096 in 4096..=67108864".into(),
        ));
    }
    let runtime = Runtime::new()?;
    let device = runtime.npu(0)?;
    let artifact = hrx::loom::Compiler::for_target(None, device.target())?
        .module(include_str!("../tests/kernels/copy.xdna.loom"))
        .compile(
            &hrx::loom::Specialization::new("copy")
                .with_config("copy.packets", (bytes / 4096).to_string()),
        )?;
    let binding = |bytes, access| BindingContract {
        bytes,
        access,
        alignment: 4,
        layout: "passthrough".into(),
    };
    let kernel = unsafe {
        device.load_artifact(
            &artifact,
            1,
            KernelContract {
                bindings: vec![binding(bytes, Access::Read), binding(bytes, Access::Write)],
                constants: vec![],
            },
        )
    }?;
    let a = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let c = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let mut graph = runtime.graph();
    graph.fill(a.view(), 0x35)?;
    graph.npu(&kernel, &[a.view(), c.view()])?;
    let graph = graph.prepare()?;
    for _ in 0..5 {
        graph.submit()?.wait()?;
    }
    let mut samples = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        graph.submit()?.wait()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    if c.map_read()?.iter().any(|&b| b != 0x35) {
        return Err(hrx::Error::Message("incorrect benchmark output".into()));
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "bytes={bytes} p50_us={:.2} p95_us={:.2}",
        percentile(&samples, 50) * 1e6,
        percentile(&samples, 95) * 1e6
    );
    Ok(())
}
