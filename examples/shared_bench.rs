//! Warm end-to-end latency for a real NPU passthrough pipeline.
//! Usage: shared_bench <passthrough.xclbin> <instructions.bin> <bytes>
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
};
use std::time::Instant;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(hrx::Error::Message(
            "usage: shared_bench <passthrough.xclbin> <instructions.bin> <bytes>".into(),
        ));
    }
    let bytes = args[3]
        .parse::<usize>()
        .map_err(|e| hrx::Error::Message(e.to_string()))?;
    let runtime = Runtime::new()?;
    let program = unsafe { runtime.npu(0)?.load_program(&args[1]) }?;
    let binding = |bytes, access| BindingContract {
        bytes,
        access,
        alignment: 4,
        layout: "passthrough".into(),
    };
    let kernel = unsafe {
        program.kernel(
            &std::fs::read(&args[2])?,
            KernelContract {
                bindings: vec![
                    binding(bytes, Access::Read),
                    binding(4096, Access::Read),
                    binding(bytes, Access::Write),
                ],
                constants: vec![],
            },
        )
    }?;
    let a = runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()))?;
    let b = runtime.allocate(4096, MemoryPlacement::Shared(program.clone()))?;
    let c = runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()))?;
    let mut graph = runtime.graph();
    graph.fill(a.view(), 0x35)?;
    graph.npu(&kernel, &[a.view(), b.view(), c.view()])?;
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
        samples[50] * 1e6,
        samples[95] * 1e6
    );
    Ok(())
}
