//! Native SDMA transfers, visibility, completion, and ring reuse.
use hrx::fabric::{Device, Engine, MemorySite, QueueCommand};

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn sdma_copies_fills_wraps_and_orders_compute_dependencies() -> hrx::Result<()> {
    let gpu = Device::open(Engine::Gpu, 0)?;
    let families = gpu.endpoint().queue_capabilities()?;
    assert!(families.iter().any(|f| f.command == QueueCommand::Sdma));
    let queue = gpu.sdma_queue()?;
    let compute = gpu.queue()?;
    let fabric = gpu.fabric();
    let source = fabric.allocate_shared(12288, std::slice::from_ref(&gpu))?;
    let target = fabric.allocate_shared(12288, std::slice::from_ref(&gpu))?;
    let input: Vec<u8> = (0..12288)
        .map(|i| ((i * 73 + (i >> 8) * 19 + 7) % 256) as u8)
        .collect();
    source.write(0, &input)?;
    let site = MemorySite::Device(&gpu, queue.family_ordinal());
    let relation = target.visibility(site, MemorySite::Host)?;
    assert!(relation.shared_backing_reachable);
    for length in [1, 2, 3, 4, 31, 4101] {
        target.write(0, &vec![0xa5; target.len()])?;
        let command = queue.prepare_copy(&target, 4091, &source, 4095, length)?;
        let done = unsafe { command.dispatch() }?;
        assert!(!done.is_complete()?); // Cached observations never retire native work.
        assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
        assert!(done.is_complete()?);
        let mut actual = vec![0; target.len()];
        target.read(0, &mut actual)?;
        let mut expected = vec![0xa5; target.len()];
        expected[4091..4091 + length].copy_from_slice(&input[4095..4095 + length]);
        assert_eq!(actual, expected);
    }
    // More than a full ring of commands, repeatedly reusing completion storage.
    let fill = queue.prepare_fill(&target, 0, target.len(), 0x37)?;
    for _ in 0..2048 {
        let done = unsafe { fill.dispatch() }?;
        assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
    }
    let compute_fill = compute.prepare_fill(&source, 0, source.len(), 0x91)?;
    let producer = unsafe { compute_fill.dispatch() }?;
    unsafe { queue.prepare_wait(&producer)?.dispatch() }?;
    let copied = unsafe {
        queue
            .prepare_copy(&target, 0, &source, 0, source.len())?
            .dispatch()
    }?;
    unsafe { compute.prepare_wait(&copied)?.dispatch() }?.wait()?;
    let mut output = vec![0; target.len()];
    target.read(0, &mut output)?;
    assert_eq!(output, vec![0x91; target.len()]);
    let tail = queue.prepare_fill(&target, 3, 17, 0x53)?;
    unsafe { tail.dispatch() }?.wait()?;
    target.read(0, &mut output)?;
    assert_eq!(&output[3..20], &[0x53; 17]);
    assert_eq!(output[2], 0x91);
    assert_eq!(output[20], 0x91);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn execution_graph_uses_prepared_sdma_copies() -> hrx::Result<()> {
    use hrx::execution::{CopyEngine, MemoryPlacement, Runtime, RuntimeOptions};
    let runtime = Runtime::with_options(RuntimeOptions {
        copy_engine: CopyEngine::Sdma,
        ..Default::default()
    })?;
    let source = runtime.allocate(4096, MemoryPlacement::HostVisible)?;
    let target = runtime.allocate(4096, MemoryPlacement::HostVisible)?;
    source.map_write()?.fill(0x5a);
    let mut graph = runtime.graph();
    graph.copy(target.view(), source.view())?;
    let graph = graph.prepare()?;
    for _ in 0..8 {
        graph.submit()?.wait()?;
    }
    assert!(target.map_read()?.iter().all(|byte| *byte == 0x5a));
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
#[ignore = "requires NPU5 and provisioned native libraries"]
fn xdna_retains_bindings_across_a_pending_window() -> hrx::Result<()> {
    use hrx::{
        fabric::XdnaBinding,
        loom::{Compiler, Specialization},
    };
    let npu = Device::open(Engine::Xdna, 0)?;
    let fabric = npu.fabric();
    let input = fabric.allocate(4096, std::slice::from_ref(&npu))?;
    let output = fabric.allocate(4096, std::slice::from_ref(&npu))?;
    input.write(0, &[0x63; 4096])?;
    let artifact = Compiler::for_target(None, npu.target())?
        .module(include_str!("kernels/copy.xdna.loom"))
        .compile(&Specialization::new("copy").with_config("copy.packets", "1"))?;
    let program = unsafe {
        npu.prepare_xdna(
            &artifact,
            1,
            &[
                XdnaBinding {
                    buffer: &input,
                    offset: 0,
                    length: 4096,
                },
                XdnaBinding {
                    buffer: &output,
                    offset: 0,
                    length: 4096,
                },
            ],
        )
    }?;
    let mut completions = Vec::new();
    for _ in 0..32 {
        completions.push(unsafe { program.dispatch() }?);
    }
    completions[0].wait()?;
    assert!(completions[0].is_complete()?);
    assert!(matches!(output.read(0, &mut [0]), Err(hrx::Error::Busy(_))));
    assert!(
        completions
            .last()
            .unwrap()
            .wait_timeout(std::time::Duration::from_secs(10))?
    );
    assert!(completions.iter().all(|done| done.is_complete().unwrap()));
    let mut actual = [0; 4096];
    output.read(0, &mut actual)?;
    assert_eq!(actual, [0x63; 4096]);
    unsafe { program.dispatch() }?.wait()?;
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn aql_dispatch_preserves_signal_epochs_and_payloads() -> hrx::Result<()> {
    use hrx::{
        fabric::Argument,
        loom::{Compiler, CxxSource, Specialization},
    };
    let gpu = Device::open(Engine::Gpu, 0)?;
    let queue = gpu.aql_queue(0)?;
    let source = CxxSource::new(
        "aql.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void affine(const unsigned* input, unsigned* output) {
  output[threadIdx.x] = input[threadIdx.x] * 3u + 7u;
}
"#,
    );
    let artifact = Compiler::for_target(None, gpu.target())?
        .import_cxx(source)?
        .compile(&Specialization::new("affine"))?;
    let kernel = unsafe { gpu.load(&artifact) }?;
    let input = gpu
        .fabric()
        .allocate_shared(256, std::slice::from_ref(&gpu))?;
    let output = gpu
        .fabric()
        .allocate_shared(256, std::slice::from_ref(&gpu))?;
    let command = unsafe {
        queue.prepare(
            &kernel,
            [1; 3],
            [64, 1, 1],
            &[Argument::Buffer(&input, 0), Argument::Buffer(&output, 0)],
        )
    }?;
    let mut earlier: Option<hrx::fabric::AqlCompletion> = None;
    for round in 0..2048u32 {
        input.write(
            0,
            &(0..64u32)
                .flat_map(|i| (i + round).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        output.write(0, &[0xa5; 256])?;
        let done = unsafe { command.dispatch() }?;
        assert!(!done.is_complete());
        if let Some(earlier) = &earlier {
            assert!(earlier.is_complete());
        }
        assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
        let mut bytes = [0; 256];
        output.read(0, &mut bytes)?;
        for (i, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(u32::from_le_bytes(*word), (i as u32 + round) * 3 + 7);
        }
        earlier = Some(done);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn aql_admits_fixed_scratch_and_rejects_undersized_queues() -> hrx::Result<()> {
    use hrx::{
        fabric::Argument,
        loom::{Compiler, ReportMode, Specialization},
    };
    let gpu = Device::open(Engine::Gpu, 0)?;
    let artifact = Compiler::for_target(None, gpu.target())?
        .module(include_str!("kernels/scratch.loom"))
        .compile(&Specialization::new("scratch").with_report(ReportMode::Summary))?;
    let private = artifact.report().unwrap().entries()?[0]
        .private_bytes
        .unwrap();
    assert!(private > 0);
    let kernel = unsafe { gpu.load(&artifact) }?;
    let output = gpu
        .fabric()
        .allocate_shared(256, std::slice::from_ref(&gpu))?;
    let indices = gpu
        .fabric()
        .allocate_shared(256, std::slice::from_ref(&gpu))?;
    indices.write(
        0,
        &(0..64u32)
            .flat_map(|i| ((i * 7) % 64).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let arguments = [Argument::Buffer(&output, 0), Argument::Buffer(&indices, 0)];
    assert!(
        unsafe {
            gpu.aql_queue(0)?
                .prepare(&kernel, [1; 3], [64, 1, 1], &arguments)
        }
        .is_err()
    );
    let queue = gpu.aql_queue(u32::try_from(private).unwrap())?;
    let prepared = unsafe { queue.prepare(&kernel, [1; 3], [64, 1, 1], &arguments) }?;
    for _ in 0..4 {
        assert!(unsafe { prepared.dispatch() }?.wait_timeout(std::time::Duration::from_secs(10))?);
        let mut actual = [0; 256];
        output.read(0, &mut actual)?;
        for (i, word) in actual.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(*word),
                ((i as u32 * 7) % 64) * 3 + i as u32
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn execution_graph_orders_aql_sdma_and_pm4_with_bounded_storage() -> hrx::Result<()> {
    use hrx::{
        execution::{
            Access, BindingContract, ComputeEngine, CopyEngine, KernelContract, MemoryPlacement,
            Runtime, RuntimeOptions,
        },
        loom::{Compiler, CxxSource, Specialization},
        residency::ResidencyManager,
    };
    let manager = ResidencyManager::new(64 << 20)?;
    let budget = manager.budget();
    {
        let runtime = Runtime::with_options(RuntimeOptions {
            memory_budget: Some(budget.clone()),
            compute_engine: ComputeEngine::Aql {
                maximum_private_bytes: 0,
            },
            copy_engine: CopyEngine::Sdma,
            ..Default::default()
        })?;
        let device = Device::open(Engine::Gpu, 0)?;
        let artifact = Compiler::for_target(None, device.target())?
            .import_cxx(CxxSource::new(
                "graph_aql.cpp",
                r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void affine(const unsigned* input, unsigned* output) {
  output[threadIdx.x] = input[threadIdx.x] * 3u + 7u;
}
"#,
            ))?
            .compile(&Specialization::new("affine"))?;
        let kernel = unsafe {
            runtime.load_gpu_artifact(
                &artifact,
                &[],
                KernelContract {
                    bindings: [Access::Read, Access::Write]
                        .into_iter()
                        .map(|access| BindingContract {
                            bytes: 256,
                            alignment: 4,
                            access,
                            layout: "u32[64]".into(),
                        })
                        .collect(),
                    constants: Vec::new(),
                },
            )
        }?;
        let input = runtime.allocate(256, MemoryPlacement::HostVisible)?;
        let middle = runtime.allocate(256, MemoryPlacement::HostVisible)?;
        let copied = runtime.allocate(256, MemoryPlacement::HostVisible)?;
        let output = runtime.allocate(256, MemoryPlacement::HostVisible)?;
        let before = budget.reserved_bytes();
        let mut graph = runtime.graph();
        graph.fill(middle.view(), 0)?;
        graph.gpu(&kernel, &[input.view(), middle.view()])?;
        graph.copy(copied.view(), middle.view())?;
        graph.gpu(&kernel, &[copied.view(), output.view()])?;
        let graph = graph.prepare()?;
        let prepared = budget.reserved_bytes();
        assert!(prepared >= before + 2 * (4096 + 64 + 64));
        for round in 0..16u32 {
            input.map_write()?.copy_from_slice(
                &(0..64u32)
                    .flat_map(|i| (i + round).to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            graph.submit()?.wait()?;
            let expected: Vec<_> = (0..64u32)
                .flat_map(|i| (9 * (i + round) + 28).to_le_bytes())
                .collect();
            assert_eq!(&*output.map_read()?, expected);
            assert_eq!(budget.reserved_bytes(), prepared);
        }
        drop(graph);
        assert!(budget.reserved_bytes() <= prepared - 2 * (4096 + 64 + 64));
    }
    assert_eq!(budget.reserved_bytes(), 0);
    // Failed preparation cannot allocate native backing outside the ceiling.
    let tiny = ResidencyManager::new(63)?;
    let gpu = Device::open(Engine::Gpu, 0)?;
    assert!(gpu.aql_queue_budgeted(16, Some(&tiny.budget())).is_err());
    assert_eq!(tiny.budget().reserved_bytes(), 0);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and provisioned native libraries"]
fn prospective_visibility_matches_selected_allocation() -> hrx::Result<()> {
    use hrx::fabric::TransitionKind;
    let gpu = Device::open(Engine::Gpu, 0)?;
    let queue = gpu.queue()?;
    let family = MemorySite::Device(&gpu, queue.family_ordinal());
    for coherent in [false, true] {
        let profiles = gpu
            .fabric()
            .allocation_profiles(std::slice::from_ref(&gpu), coherent)?;
        assert!(!profiles.is_empty());
        for profile in profiles {
            let prospective = profile.visibility(MemorySite::Host, family)?;
            let buffer = profile.allocate(4096, None)?;
            let concrete = buffer.visibility(MemorySite::Host, family)?;
            assert_eq!(
                prospective.shared_backing_reachable,
                concrete.shared_backing_reachable
            );
            assert_eq!(prospective.release.kind, concrete.release.kind);
            assert_eq!(prospective.release.executor, concrete.release.executor);
            assert_eq!(prospective.acquire.kind, concrete.acquire.kind);
            assert_eq!(prospective.acquire.operation, concrete.acquire.operation);
            assert_eq!(prospective.atomic_scope_32, concrete.atomic_scope_32);
            let publish = concrete.prepare_release_host(&[0..16, 8..32, 32..64, 256..320])?;
            assert_eq!(
                publish.operation_count(),
                if concrete.release.kind == TransitionKind::Range {
                    2
                } else {
                    1
                }
            );
            unsafe { publish.execute() }?;
            let done = unsafe { queue.prepare_fill(&buffer, 0, 4096, 0x39)?.dispatch() }?;
            done.wait()?;
            let acquire = buffer
                .visibility(family, MemorySite::Host)?
                .prepare_acquire_host(std::slice::from_ref(&(0..4096)))?;
            unsafe { acquire.execute() }?;
            let mut bytes = vec![0; 4096];
            buffer.read(0, &mut bytes)?;
            assert_eq!(bytes, vec![0x39; 4096]);
        }
    }
    Ok(())
}
