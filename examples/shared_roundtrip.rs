//! Execute GPU fill -> NPU passthrough -> GPU copy on one imported allocation.
//! Usage: shared_roundtrip <passthrough.xclbin> <instructions.bin> <bytes>
//! The supplied design must use the MLIR_AIE ABI (A, unused B, C), copying A to C.
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(hrx::Error::Message(
            "usage: shared_roundtrip <passthrough.xclbin> <instructions.bin> <bytes>".into(),
        ));
    }
    let bytes = args[3]
        .parse::<usize>()
        .map_err(|e| hrx::Error::Message(e.to_string()))?;
    let runtime = Runtime::new()?;
    // This example accepts trusted local artifacts with the documented ABI.
    let program = unsafe { runtime.npu(0)?.load_program(&args[1]) }?;
    let binding = |bytes, access| BindingContract {
        bytes,
        access,
        alignment: 4,
        layout: "u32 passthrough".into(),
    };
    let contract = KernelContract {
        bindings: vec![
            binding(bytes, Access::Read),
            binding(4096, Access::Read),
            binding(bytes, Access::Write),
        ],
        constants: vec![],
    };
    let kernel = unsafe { program.kernel(&std::fs::read(&args[2])?, contract) }?;
    let a = runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()))?;
    let b = runtime.allocate(4096, MemoryPlacement::Shared(program.clone()))?;
    let c = runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()))?;
    let result = runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()))?;
    let mut graph = runtime.graph();
    graph.fill(a.view(), 0x5a)?;
    graph.npu(&kernel, &[a.view(), b.view(), c.view()])?;
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
