//! Real GPU preprocessing -> BF16 NPU GEMM -> GPU f32 epilogue.
//! Usage: gemm_pipeline <row-major-bf16.xclbin> <instructions.bin> M K N
//! Artifacts must implement A[M,K] * B[K,N] -> C[M,N] f32, using MLIR_AIE.
use hrx::{
    Result,
    execution::{
        Access, BindingContract, Buffer, ExecutableGraph, GpuKernel, KernelContract,
        MemoryPlacement, Runtime,
    },
};
use std::time::Instant;
fn binding(bytes: usize, access: Access) -> BindingContract {
    BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "contiguous row-major".into(),
    }
}
fn gpu_kernel(
    runtime: &Runtime,
    source: &str,
    symbol: &str,
    config: &[(&str, String)],
    elements: usize,
    bindings: Vec<BindingContract>,
) -> Result<GpuKernel> {
    let compiler = hrx::loom::Compiler::with_options(
        None,
        hrx::loom::CompilerOptions {
            target: runtime.gpu()?.target().clone(),
            ..Default::default()
        },
    )?;
    let module = compiler.module(source);
    let mut spec = hrx::loom::Specialization::new(symbol);
    spec.config
        .extend(config.iter().map(|(k, v)| (k.to_string(), v.clone())));
    let artifact = module.compile(&spec)?;
    let stream = hrx::gpu::Stream::open()?;
    let raw = unsafe { stream.load_artifact(&artifact) }?;
    let mut constants = hrx::gpu::Constants::new();
    match raw.info().constant_byte_length {
        8 => constants.push(elements as u32)?,
        12 => constants.push(elements as u64)?,
        _ => return Err(hrx::Error::Message("unexpected scalar ABI".into())),
    }
    constants.push(1f32)?;
    unsafe {
        runtime.load_gpu_kernel(
            artifact.path(),
            symbol,
            [elements.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            KernelContract {
                bindings,
                constants: constants.as_bytes().to_vec(),
            },
        )
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(hrx::Error::Message(
            "usage: gemm_pipeline <row-major-bf16.xclbin> <instructions.bin> M K N".into(),
        ));
    }
    let parse = |i: usize| {
        args[i]
            .parse::<usize>()
            .map_err(|e| hrx::Error::Message(e.to_string()))
    };
    let (m, k, n) = (parse(3)?, parse(4)?, parse(5)?);
    let extent = |a: usize, b: usize, width: usize| {
        a.checked_mul(b)
            .filter(|&count| count > 0 && count <= (1 << 30))
            .and_then(|count| count.checked_mul(width))
            .ok_or_else(|| hrx::Error::Message("invalid matrix extent".into()))
    };
    let (a_bytes, b_bytes, c_bytes) = (extent(m, k, 2)?, extent(k, n, 2)?, extent(m, n, 4)?);
    let runtime = Runtime::new()?;
    let program = unsafe { runtime.npu(0)?.load_program(&args[1]) }?;
    let npu = unsafe {
        program.kernel(
            &std::fs::read(&args[2])?,
            KernelContract {
                bindings: vec![
                    binding(a_bytes, Access::Read),
                    binding(b_bytes, Access::Read),
                    binding(c_bytes, Access::Write),
                ],
                constants: vec![],
            },
        )
    }?;
    let prep = gpu_kernel(
        &runtime,
        include_str!("../tests/kernels/euler.loom"),
        "krea2_euler",
        &[
            ("krea2.euler.grid_x", (m * k).div_ceil(256).to_string()),
            ("krea2.euler.grid_y", "1".into()),
        ],
        m * k,
        vec![
            binding(a_bytes, Access::ReadWrite),
            binding(a_bytes, Access::Read),
        ],
    )?;
    let epilogue = gpu_kernel(
        &runtime,
        include_str!("../tests/kernels/add_f32.loom"),
        "add_f32",
        &[("add.grid", (m * n).div_ceil(256).to_string())],
        m * n,
        vec![binding(c_bytes, Access::ReadWrite)],
    )?;
    let allocate = |bytes| runtime.allocate(bytes, MemoryPlacement::Shared(program.clone()));
    let weights = allocate(b_bytes)?;
    let ones = allocate(a_bytes)?;
    for buffer in [&weights, &ones] {
        for word in buffer.map_write()?.chunks_exact_mut(2) {
            word.copy_from_slice(&0x3f80u16.to_le_bytes());
        }
    }
    let prepare = || -> Result<(ExecutableGraph, Buffer)> {
        let a = allocate(a_bytes)?;
        let c = allocate(c_bytes)?;
        let mut graph = runtime.graph();
        graph.fill(a.view(), 0)?;
        graph.gpu(&prep, &[a.view(), ones.view()])?;
        graph.npu(&npu, &[a.view(), weights.view(), c.view()])?;
        graph.gpu(&epilogue, &[c.view()])?;
        Ok((graph.prepare()?, c))
    };
    let (first, c0) = prepare()?;
    let (second, c1) = prepare()?;
    for _ in 0..5 {
        first.submit()?.wait()?;
        second.submit()?.wait()?;
    }
    let before = runtime.statistics();
    let mut latency = Vec::with_capacity(50);
    for _ in 0..50 {
        let start = Instant::now();
        first.submit()?.wait()?;
        latency.push(start.elapsed().as_secs_f64());
    }
    let start = Instant::now();
    for _ in 0..50 {
        let a = first.submit()?;
        let b = second.submit()?;
        a.wait()?;
        b.wait()?;
    }
    let requests_per_second = 100.0 / start.elapsed().as_secs_f64();
    for c in [&c0, &c1] {
        if c.map_read()?
            .chunks_exact(4)
            .any(|v| f32::from_le_bytes(v.try_into().unwrap()) != k as f32 + 1.0)
        {
            return Err(hrx::Error::Message("GEMM pipeline output mismatch".into()));
        }
    }
    let after = runtime.statistics();
    assert_eq!(before.allocations, after.allocations);
    assert_eq!(before.imports, after.imports);
    assert_eq!(after.copied_bytes, 0);
    latency.sort_by(f64::total_cmp);
    println!(
        "M={m} K={k} N={n} p50_ms={:.3} p95_ms={:.3} pipelined_requests_s={requests_per_second:.2}",
        latency[25] * 1e3,
        latency[47] * 1e3
    );
    println!("{}", serde_json::to_string(&after)?);
    Ok(())
}
