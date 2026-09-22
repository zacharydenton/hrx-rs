//! Execute GPU fill -> NPU passthrough -> GPU copy on one imported allocation.
//! Usage: shared_roundtrip [bytes]
//! Compiles the bundled Loom copy kernel for the active native XDNA device.
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 2 {
        return Err(hrx::Error::Message(
            "usage: shared_roundtrip [bytes]".into(),
        ));
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
        layout: "u32 passthrough".into(),
    };
    let contract = KernelContract {
        bindings: vec![binding(bytes, Access::Read), binding(bytes, Access::Write)],
        constants: vec![],
    };
    let kernel = unsafe { device.load_artifact(&artifact, 1, contract) }?;
    let a = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let c = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let result = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let mut graph = runtime.graph();
    graph.fill(a.view(), 0x5a)?;
    graph.npu(&kernel, &[a.view(), c.view()])?;
    graph.copy(result.view(), c.view())?;
    let graph = graph.prepare()?;
    for _ in 0..10 {
        graph.submit()?.wait()?;
        let mapped = result.map_read()?;
        if mapped.iter().any(|&b| b != 0x5a) {
            return Err(hrx::Error::Message("GPU/NPU roundtrip mismatch".into()));
        }
    }
    println!(
        "GPU -> NPU -> GPU: all {bytes} bytes correct, 10 executions, {} prepared regions",
        graph.regions()
    );
    Ok(())
}
