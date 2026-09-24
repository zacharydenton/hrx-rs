//! Native integration through the tracked scheduler and its memory contracts.
#![cfg(feature = "npu")]
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
    loom::{Compiler, Specialization},
};
#[test]
#[ignore = "requires native compiler, libamdf, bridge, gfx1151 and NPU5"]
fn tracked_graph_orders_gpu_xdna_and_host_visibility() -> Result<()> {
    let runtime = Runtime::new()?;
    let device = runtime.npu(0)?;
    let artifact = Compiler::for_target(None, device.target())?
        .module(include_str!("kernels/copy.xdna.loom"))
        .compile(&Specialization::new("copy").with_config("copy.packets", "256"))?;
    let bytes = 1 << 20;
    let contract = KernelContract {
        bindings: [(bytes, Access::Read), (bytes, Access::Write)]
            .into_iter()
            .map(|(bytes, access)| BindingContract {
                bytes,
                access,
                alignment: 4,
                layout: "copy bytes".into(),
            })
            .collect(),
        constants: vec![],
    };
    let kernel = unsafe { device.load_artifact(&artifact, 1, contract) }?;
    let source = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let output = runtime.allocate(bytes, MemoryPlacement::Shared(device.clone()))?;
    let mut graph = runtime.graph();
    graph.fill(source.view(), 0x3d)?;
    graph.npu(&kernel, &[source.view(), output.view()])?;
    let graph = graph.prepare()?;
    for _ in 0..4 {
        graph.submit()?.wait()?;
        assert!(output.map_read()?.iter().all(|byte| *byte == 0x3d));
    }
    Ok(())
}

#[test]
#[ignore = "requires native C++ importer, libamdf and NPU5"]
fn cpp_vector_worker_executes_with_changed_inputs() -> Result<()> {
    use hrx::loom::{CxxSource, Source};
    let runtime = Runtime::new()?;
    let device = runtime.npu(0)?;
    let pipeline = include_str!("kernels/copy.xdna.loom").replace(
        "vector.store %value,",
        "%transformed = func.call @transform(%value) : (vector<16xi32>) -> (vector<16xi32>)\n    vector.store %transformed,");
    let pipeline =
        format!("func.decl @transform(%input: vector<16xi32>) -> (vector<16xi32>)\n{pipeline}");
    let artifact = Compiler::for_target(None, device.target())?.sources(vec![
        Source::loom("pipeline.loom", pipeline),
        Source::Cxx(CxxSource::new("worker.cpp", "typedef int V __attribute__((vector_size(64))); extern \"C\" V transform(V input) { return input + 1; }")),
    ])?.compile(&Specialization::new("copy").with_config("copy.packets", "1"))?;
    let contract = KernelContract {
        bindings: [Access::Read, Access::Write]
            .into_iter()
            .map(|access| BindingContract {
                bytes: 4096,
                access,
                alignment: 64,
                layout: "1024 i32".into(),
            })
            .collect(),
        constants: vec![],
    };
    let kernel = unsafe { device.load_artifact(&artifact, 1, contract) }?;
    let source = runtime.allocate(4096, MemoryPlacement::Shared(device.clone()))?;
    let output = runtime.allocate(4096, MemoryPlacement::Shared(device))?;
    let mut graph = runtime.graph();
    graph.npu(&kernel, &[source.view(), output.view()])?;
    let graph = graph.prepare()?;
    for pass in 0..3i32 {
        for (i, word) in source
            .map_write()?
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .enumerate()
        {
            word.copy_from_slice(&(i as i32 * 7 - pass * 13).to_le_bytes());
        }
        graph.submit()?.wait()?;
        for (i, word) in output.map_read()?.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(i32::from_le_bytes(*word), i as i32 * 7 - pass * 13 + 1);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires native compiler, libamdf and NPU5"]
fn bf16_matrix_multiply_matches_independent_cpu_oracle() -> Result<()> {
    let runtime = Runtime::new()?;
    let device = runtime.npu(0)?;
    let artifact = Compiler::for_target(None, device.target())?
        .module(include_str!("kernels/gemm_bf16.xdna.loom"))
        .compile(&Specialization::new("gemm"))?;
    let contract = KernelContract {
        bindings: [
            (128, Access::Read),
            (128, Access::Read),
            (256, Access::Write),
        ]
        .into_iter()
        .map(|(bytes, access)| BindingContract {
            bytes,
            access,
            alignment: 64,
            layout: "8x8 row-major".into(),
        })
        .collect(),
        constants: vec![],
    };
    let kernel = unsafe { device.load_artifact(&artifact, 1, contract) }?;
    let make = || -> Result<_> {
        let a = runtime.allocate(128, MemoryPlacement::Shared(device.clone()))?;
        let b = runtime.allocate(128, MemoryPlacement::Shared(device.clone()))?;
        let c = runtime.allocate(256, MemoryPlacement::Shared(device.clone()))?;
        let mut graph = runtime.graph();
        graph.npu(&kernel, &[a.view(), b.view(), c.view()])?;
        Ok((graph.prepare()?, a, b, c))
    };
    let runs = [make()?, make()?];
    for pass in 0..4 {
        for (request, (_, a, b, _)) in runs.iter().enumerate() {
            for (which, buffer) in [a, b].iter().enumerate() {
                for (i, word) in buffer
                    .map_write()?
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    let value = ((i * (which + 3) + request + pass) % 7) as f32 - 3.0;
                    word.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                }
            }
        }
        let first = runs[0].0.submit()?;
        let second = runs[1].0.submit()?;
        first.wait()?;
        second.wait()?;
        for (request, (_, _, _, c)) in runs.iter().enumerate() {
            for (i, word) in c.map_read()?.as_chunks::<4>().0.iter().enumerate() {
                let expected: f32 = (0..8)
                    .map(|k| {
                        let a = (((i / 8 * 8 + k) * 3 + request + pass) % 7) as f32 - 3.0;
                        let b = (((k * 8 + i % 8) * 4 + request + pass) % 7) as f32 - 3.0;
                        a * b
                    })
                    .sum();
                assert_eq!(f32::from_le_bytes(*word), expected, "entry {i}");
            }
        }
    }
    Ok(())
}
