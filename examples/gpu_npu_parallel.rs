//! Independent GPU vector processing and batched NPU matrix multiplication.
//! Usage: gpu_npu_parallel [gpu-elements] [npu-matrices] [samples] [trace.json]
//! Timings cover warm, completed graphs, excluding compilation and host I/O.
use hrx::{
    Error, Result,
    execution::{
        Access, BindingContract, Buffer, ExecutableGraph, KernelContract, MemoryPlacement, Runtime,
    },
    loom::{Compiler, CxxSource, Specialization},
};
use std::{path::PathBuf, time::Instant};

fn contract(bindings: &[(usize, Access)]) -> KernelContract {
    KernelContract {
        bindings: bindings
            .iter()
            .map(|&(bytes, access)| BindingContract {
                bytes,
                alignment: 64,
                access,
                layout: "contiguous row-major".into(),
            })
            .collect(),
        constants: vec![],
    }
}

fn gpu_value(index: usize, pass: usize) -> i32 {
    ((index + pass * 17) % 2048) as i32 - 1024
}

fn matrix_value(matrix: usize, index: usize, operand: usize, pass: usize) -> f32 {
    ((index * (operand + 3) + matrix + pass) % 7) as f32 - 3.0
}

fn initialize(gpu: &Buffer, a: &Buffer, b: &Buffer, pass: usize) -> Result<()> {
    for (index, word) in gpu.map_write()?.chunks_exact_mut(4).enumerate() {
        word.copy_from_slice(&gpu_value(index, pass).to_le_bytes());
    }
    for (operand, buffer) in [a, b].into_iter().enumerate() {
        for (index, word) in buffer.map_write()?.chunks_exact_mut(2).enumerate() {
            let value = matrix_value(index / 64, index % 64, operand, pass);
            // Integers in -3..=3 are exactly representable in BF16 and BFP16.
            word.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    Ok(())
}

fn validate(gpu: &Buffer, matrices: &Buffer, pass: usize) -> Result<()> {
    for (index, word) in gpu.map_read()?.chunks_exact(4).enumerate() {
        let actual = i32::from_le_bytes(word.try_into().unwrap());
        if actual != gpu_value(index, pass) * 3 + 7 {
            return Err(Error::Message(format!("GPU mismatch at element {index}")));
        }
    }
    for (index, word) in matrices.map_read()?.chunks_exact(4).enumerate() {
        let matrix = index / 64;
        let row = index % 64 / 8;
        let column = index % 8;
        let expected: f32 = (0..8)
            .map(|k| {
                matrix_value(matrix, row * 8 + k, 0, pass)
                    * matrix_value(matrix, k * 8 + column, 1, pass)
            })
            .sum();
        if f32::from_le_bytes(word.try_into().unwrap()) != expected {
            return Err(Error::Message(format!(
                "NPU mismatch at matrix {matrix}, ({row}, {column})"
            )));
        }
    }
    Ok(())
}

fn completed_ms(graph: &ExecutableGraph) -> Result<f64> {
    let start = Instant::now();
    graph.submit()?.wait()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 4 || args.iter().any(|arg| arg == "--help") {
        println!(
            "usage: gpu_npu_parallel [gpu-elements=16777216] [npu-matrices=1024] [samples=21] [trace.json]"
        );
        return if args.len() <= 4 {
            Ok(())
        } else {
            Err(Error::Message("too many arguments".into()))
        };
    }
    let number = |index: usize, default: usize| -> Result<usize> {
        args.get(index).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| Error::Message(format!("invalid integer: {value}")))
        })
    };
    let elements = number(0, 1 << 24)?;
    let matrices = number(1, 1024)?;
    let samples = number(2, 21)?;
    if !(256..=1 << 26).contains(&elements)
        || !elements.is_multiple_of(256)
        || !(1..=16384).contains(&matrices)
        || !(3..=1000).contains(&samples)
    {
        return Err(Error::Message("gpu-elements must be a multiple of 256 in 256..=67108864; npu-matrices in 1..=16384; samples in 3..=1000".into()));
    }
    let trace_path = args.get(3).map(PathBuf::from);
    let runtime = Runtime::new()?;
    let gpu = runtime.gpu()?;
    let npu = runtime.npu(0)?;
    println!(
        "GPU: {}; NPU: {}",
        gpu.target().as_str(),
        npu.target().as_str()
    );
    let gpu_artifact = Compiler::for_target(None, gpu.target())?
        .import_cxx(CxxSource::new(
            "affine.cpp",
            format!(
                r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(256, 1, 1), loom::workgroup_count({}, 1, 1)]]
void affine(const int* input, int* output) {{
  unsigned i = blockIdx.x * 256u + threadIdx.x;
  output[i] = input[i] * 3 + 7;
}}
"#,
                elements / 256
            ),
        ))?
        .compile(&Specialization::new("affine"))?;
    // SAFETY: the fixed launch covers exactly elements i32 values in each
    // nonaliasing binding; bounded input values cannot overflow the arithmetic.
    let gpu_kernel = unsafe {
        runtime.load_gpu_kernel(
            gpu_artifact.path(),
            "affine",
            [(elements / 256) as u32, 1, 1],
            [256, 1, 1],
            contract(&[(elements * 4, Access::Read), (elements * 4, Access::Write)]),
        )
    }?;
    let npu_artifact = Compiler::for_target(None, npu.target())?
        .module(include_str!("kernels/batched_gemm.xdna.loom"))
        .compile(&Specialization::new("gemm").with_config("gemm.batch", matrices.to_string()))?;
    // SAFETY: the pipeline consumes matrices pairs of 8x8 BF16 inputs and
    // produces the same number of 8x8 FP32 outputs on one NPU column.
    let npu_kernel = unsafe {
        npu.load_artifact(
            &npu_artifact,
            1,
            contract(&[
                (matrices * 128, Access::Read),
                (matrices * 128, Access::Read),
                (matrices * 256, Access::Write),
            ]),
        )
    }?;
    let input = runtime.allocate(elements * 4, MemoryPlacement::GpuLocal)?;
    let output = runtime.allocate(elements * 4, MemoryPlacement::GpuLocal)?;
    let host_input = runtime.allocate(elements * 4, MemoryPlacement::HostVisible)?;
    let host_output = runtime.allocate(elements * 4, MemoryPlacement::HostVisible)?;
    let mut upload = runtime.graph();
    upload.copy(input.view(), host_input.view())?;
    let upload = upload.prepare()?;
    let mut download = runtime.graph();
    download.copy(host_output.view(), output.view())?;
    let download = download.prepare()?;
    let a = runtime.allocate(matrices * 128, MemoryPlacement::NpuLocal(npu.clone()))?;
    let b = runtime.allocate(matrices * 128, MemoryPlacement::NpuLocal(npu.clone()))?;
    let c = runtime.allocate(matrices * 256, MemoryPlacement::NpuLocal(npu))?;

    let mut gpu_only = runtime.graph();
    gpu_only.gpu(&gpu_kernel, &[input.view(), output.view()])?;
    let gpu_only = gpu_only.prepare()?;
    let mut npu_only = runtime.graph();
    npu_only.npu(&npu_kernel, &[a.view(), b.view(), c.view()])?;
    let npu_only = npu_only.prepare()?;

    // Disjoint buffers imply no dependency edge. One submit admits both
    // branches to separate GPU/NPU workers; one wait joins their completions.
    let mut parallel = runtime.graph();
    parallel.gpu(&gpu_kernel, &[input.view(), output.view()])?;
    parallel.npu(&npu_kernel, &[a.view(), b.view(), c.view()])?;
    let parallel = parallel.prepare()?;
    for pass in 0..3 {
        initialize(&host_input, &a, &b, pass * 2)?;
        upload.submit()?.wait()?;
        gpu_only.submit()?.wait()?;
        npu_only.submit()?.wait()?;
        download.submit()?.wait()?;
        validate(&host_output, &c, pass * 2)?;
        // Change both inputs so validation cannot accept the sequential run's
        // old output if either parallel branch fails to write its result.
        initialize(&host_input, &a, &b, pass * 2 + 1)?;
        upload.submit()?.wait()?;
        parallel.submit()?.wait()?;
        download.submit()?.wait()?;
        validate(&host_output, &c, pass * 2 + 1)?;
    }

    let before = runtime.statistics();
    let mut sequential_ms = Vec::with_capacity(samples);
    let mut parallel_ms = Vec::with_capacity(samples);
    for sample in 0..samples {
        // Alternate order to reduce systematic clock/thermal ordering bias.
        for concurrent in if sample % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            if concurrent {
                parallel_ms.push(completed_ms(&parallel)?);
            } else {
                let start = Instant::now();
                gpu_only.submit()?.wait()?;
                npu_only.submit()?.wait()?;
                sequential_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    let after = runtime.statistics();
    if before.allocations != after.allocations || before.imports != after.imports {
        return Err(Error::Message(
            "replay unexpectedly allocated or imported storage".into(),
        ));
    }
    download.submit()?.wait()?;
    validate(&host_output, &c, 5)?;
    let sequential = median(&mut sequential_ms);
    let concurrent = median(&mut parallel_ms);
    println!("{elements} GPU elements + {matrices} NPU matrices; {samples} paired samples");
    println!(
        "sequential median: {sequential:.3} ms; parallel median: {concurrent:.3} ms; speedup: {:.2}x",
        sequential / concurrent
    );
    println!(
        "Both CPU references match exactly, including changed-input replays. No native allocations/imports during timed replay."
    );

    // Capture separately so tracing does not affect the benchmark samples.
    runtime.start_trace(32)?;
    for _ in 0..4 {
        parallel.submit()?.wait()?;
    }
    let trace = runtime
        .finish_trace()
        .ok_or_else(|| Error::Message("missing execution trace".into()))?;
    let overlap: f64 = trace
        .events
        .iter()
        .filter(|e| e.lane == 1)
        .flat_map(|gpu| {
            trace.events.iter().filter(|e| e.lane == 3).map(move |npu| {
                ((gpu.start_us + gpu.duration_us).min(npu.start_us + npu.duration_us)
                    - gpu.start_us.max(npu.start_us))
                .max(0.0)
            })
        })
        .sum();
    println!(
        "GPU/NPU host-observed overlap across four replays: {overlap:.1} us (not device timestamps)"
    );
    if let Some(path) = trace_path {
        std::fs::write(&path, trace.to_json()?)?;
        println!("Trace: {}", path.display());
    }
    Ok(())
}
