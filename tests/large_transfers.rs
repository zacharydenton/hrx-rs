//! Logical transfers spanning native packet, ring and compute-index limits.
use hrx::execution::{ComputeEngine, CopyEngine};

#[test]
#[ignore = "requires gfx1151 and at least 12 GiB available memory"]
fn streams_and_recorded_graphs_copy_and_fill_beyond_four_gib() -> hrx::Result<()> {
    let bytes = (1usize << 32) + 257;
    for compute_engine in [
        ComputeEngine::Pm4,
        ComputeEngine::Aql {
            maximum_private_bytes: 4096,
        },
    ] {
        for copy_engine in [CopyEngine::Compute, CopyEngine::Sdma] {
            let manager = hrx::residency::ResidencyManager::new(12 << 30)?;
            {
                let device = hrx::Device::open(0)?;
                let mut stream = device.stream_with_options(hrx::StreamOptions {
                    compute_engine,
                    copy_engine,
                    memory_budget: Some(manager.budget()),
                    ..Default::default()
                })?;
                let source = stream.allocate(bytes + 16)?;
                let target = stream.allocate(bytes + 16)?;
                stream.fill(source.binding(), 0x35)?;
                stream.fill(target.binding(), 0x79)?;
                let mut probes: Vec<_> = (0..bytes - 3).step_by(64 << 20).collect();
                probes.extend([(64 << 20) - 4, 68 << 20, (1usize << 32) - 16, bytes - 4]);
                probes.sort_unstable();
                probes.dedup();
                for (i, offset) in probes.iter().enumerate() {
                    stream.upload_blocking_at(&source, 1 + offset, &[i as u8; 4])?;
                }
                assert!(
                    stream
                        .copy(target.slice(0, bytes), target.slice(8, bytes))
                        .is_err()
                );
                stream.copy(target.slice(3, bytes), source.slice(1, bytes))?;
                let done = stream.record_event()?;
                // A consumer on another queue must see the final physical chunk.
                let mut observer = device.stream_with_options(hrx::StreamOptions {
                    compute_engine,
                    copy_engine,
                    memory_budget: Some(manager.budget()),
                    ..Default::default()
                })?;
                let observed = observer.allocate(4)?;
                observer.wait_event(&done)?;
                observer.copy(observed.binding(), target.slice(3 + bytes - 4, 4))?;
                let mut tail = [0; 4];
                observer.read_blocking(observed.binding(), &mut tail)?;
                assert_eq!(tail, [probes.len() as u8 - 1; 4]);
                let verify = |stream: &mut hrx::Stream, phase: &str| -> hrx::Result<()> {
                    for (i, offset) in probes.iter().enumerate() {
                        let mut actual = [0; 4];
                        stream.read_blocking_at(&target, 3 + offset, &mut actual)?;
                        assert_eq!(
                            actual, [i as u8; 4],
                            "{compute_engine:?}/{copy_engine:?} {phase} at {offset}"
                        );
                    }
                    for (offset, value) in [(2, 0x79), (3 + bytes, 0x79), (15, 0x35)] {
                        let mut actual = [0];
                        stream.read_blocking_at(&target, offset, &mut actual)?;
                        assert_eq!(actual, [value]);
                    }
                    Ok(())
                };
                verify(&mut stream, "stream")?;
                let mut graph = stream.owned_graph()?;
                let fill = graph.fill(&[], target.binding(), 0x79)?;
                graph.copy(&[fill], target.slice(3, bytes), source.slice(1, bytes))?;
                let mut graph = graph.finish()?;
                for _ in 0..2 {
                    stream.launch(&mut graph)?;
                    stream.synchronize()?;
                    verify(&mut stream, "graph")?;
                }
                // An earlier immutable completion remains complete after replay.
                assert!(done.is_complete()?);
            }
            assert_eq!(manager.statistics().reserved_bytes, 0);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and native SDMA"]
fn tracked_sdma_graph_spans_ring_capacity_and_releases_budget() -> hrx::Result<()> {
    use hrx::execution::{MemoryPlacement, Runtime, RuntimeOptions};
    let manager = hrx::residency::ResidencyManager::new(1 << 30)?;
    {
        let runtime = Runtime::with_options(RuntimeOptions {
            copy_engine: CopyEngine::Sdma,
            memory_budget: Some(manager.budget()),
            ..Default::default()
        })?;
        let bytes = (256 << 20) + 19;
        let source = runtime.allocate(bytes, MemoryPlacement::HostVisible)?;
        let target = runtime.allocate(bytes, MemoryPlacement::HostVisible)?;
        let mut graph = runtime.graph();
        graph.copy(target.view(), source.view())?;
        let graph = graph.prepare()?;
        for value in [0x35, 0x79, 0xa5] {
            source.map_write()?.fill(value);
            graph.submit()?.wait()?;
            assert!(target.map_read()?.iter().all(|&byte| byte == value));
        }
    }
    assert_eq!(manager.statistics().reserved_bytes, 0);
    Ok(())
}
