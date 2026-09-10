use super::graph::{Backend, NodeState, Operation, Prepared, Slot, Use};
use super::*;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};
fn buffer(runtime: &Runtime) -> Buffer {
    // Storage owns the only handle and supplies the same leases as native memory.
    #[allow(clippy::arc_with_non_send_sync)]
    let memory = Arc::new(std::cell::UnsafeCell::new([0u8; 256]));
    Buffer {
        storage: Arc::new(Storage {
            accounted: false,
            #[cfg(feature = "npu")]
            bo: None,
            #[cfg(feature = "npu")]
            descriptor: None,
            gpu: None,
            pointer: memory.get().cast::<u8>(),
            bytes: 256,
            #[cfg(feature = "npu")]
            npu_device: None,
            #[cfg(feature = "npu")]
            group: None,
            shared: true,
            host: Mutex::new(HostState::default()),
            visibility: Mutex::new(Visibility::new()),
            runtime: runtime.inner.clone(),
            _test_memory: Some(memory),
        }),
    }
}
fn mock(
    runtime: &Runtime,
    engine: Engine,
    buffer: &Buffer,
    access: Access,
    action: impl Fn() -> Result<()> + Send + Sync + 'static,
) -> ExecutableGraph {
    let uses = vec![Use {
        view: buffer.view(),
        access,
    }];
    ExecutableGraph {
        inner: Arc::new(Prepared {
            signals: (0..runtime.options.graph_slots)
                .map(|_| completion::Signal::new())
                .collect(),
            runtime: runtime.inner.clone(),
            parallel: false,
            operations: vec![Operation {
                copy_bytes: 0,
                backend: Backend::Mock(engine, Arc::new(action)),
                uses: uses.clone(),
                dependencies: vec![],
            }],
            uses,
            slots: Mutex::new(
                (0..runtime.options.graph_slots)
                    .map(|_| Slot {
                        nodes: vec![NodeState::Pending],
                        occupied: false,
                        failure: None,
                    })
                    .collect(),
            ),
        }),
    }
}
#[test]
fn aliases_and_host_guards_share_one_access_domain() {
    let runtime = Runtime::new().unwrap();
    let buffer = buffer(&runtime);
    let alias = buffer.clone();
    let mut write = buffer.map_write().unwrap();
    write[0] = 7;
    assert!(matches!(alias.try_map_read(), Err(Error::Busy(_))));
    assert!(matches!(alias.try_map_write(), Err(Error::Busy(_))));
    let graph = mock(&runtime, Engine::Gpu, &alias, Access::Read, || Ok(()));
    assert!(matches!(graph.submit(), Err(Error::Busy(_))));
    drop(write);
    let read = alias.map_read().unwrap();
    assert_eq!(read[0], 7);
    let second = buffer.map_read().unwrap();
    assert_eq!(second[0], 7);
    graph.submit().unwrap().wait().unwrap();
    assert!(matches!(buffer.try_map_write(), Err(Error::Busy(_))));
}
#[test]
fn conflicts_order_across_submissions_and_detached_handles() {
    let runtime = Runtime::new().unwrap();
    let buffer = buffer(&runtime);
    let value = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Mutex::new(release_rx);
    let writer_value = value.clone();
    let producer = mock(&runtime, Engine::Gpu, &buffer, Access::Write, move || {
        started_tx.send(()).unwrap();
        release_rx.lock().unwrap().recv().unwrap();
        writer_value.store(42, Ordering::Release);
        Ok(())
    });
    let reader_value = value.clone();
    let consumer = mock(&runtime, Engine::Npu, &buffer, Access::Read, move || {
        assert_eq!(reader_value.load(Ordering::Acquire), 42);
        Ok(())
    });
    drop(producer.submit().unwrap());
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let done = consumer.submit().unwrap();
    assert!(!done.wait_timeout(Duration::from_millis(10)).unwrap());
    assert!(matches!(buffer.try_map_read(), Err(Error::Busy(_))));
    drop(producer);
    release_tx.send(()).unwrap();
    assert!(done.wait_timeout(Duration::from_secs(2)).unwrap());
}
#[test]
fn independent_engines_progress_while_another_device_waits() {
    let runtime = Runtime::new().unwrap();
    let a = buffer(&runtime);
    let b = buffer(&runtime);
    let (tx, rx) = mpsc::sync_channel(1);
    let rx = Mutex::new(rx);
    let gpu = mock(&runtime, Engine::Gpu, &a, Access::Write, move || {
        rx.lock().unwrap().recv().unwrap();
        Ok(())
    });
    let npu = mock(&runtime, Engine::Npu, &b, Access::Write, || Ok(()));
    let first = gpu.submit().unwrap();
    let second = npu.submit().unwrap();
    assert!(second.wait_timeout(Duration::from_secs(2)).unwrap());
    assert!(!first.is_complete());
    tx.send(()).unwrap();
    first.wait().unwrap();
}
#[test]
fn independent_regions_in_one_graph_progress_with_a_host_waiter() {
    let runtime = Runtime::new().unwrap();
    let a = buffer(&runtime);
    let b = buffer(&runtime);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Mutex::new(release_rx);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let mut graph = mock(&runtime, Engine::Gpu, &a, Access::Write, move || {
        release_rx.lock().unwrap().recv().unwrap();
        Ok(())
    });
    let prepared = Arc::get_mut(&mut graph.inner).unwrap();
    prepared.parallel = true;
    let usage = Use {
        view: b.view(),
        access: Access::Write,
    };
    prepared.uses.push(usage.clone());
    prepared.operations.push(Operation {
        copy_bytes: 0,
        dependencies: vec![],
        uses: vec![usage],
        backend: Backend::Mock(
            Engine::Npu,
            Arc::new(move || {
                ready_tx.send(()).unwrap();
                Ok(())
            }),
        ),
    });
    for slot in prepared.slots.get_mut().unwrap() {
        slot.nodes.push(NodeState::Pending);
    }
    let completion = graph.submit().unwrap();
    std::thread::scope(|scope| {
        let mapping = scope.spawn(|| {
            let guard = a.map_read().unwrap();
            assert_eq!(guard[0], 0);
        });
        let progressed = ready_rx.recv_timeout(Duration::from_secs(2));
        release_tx.send(()).unwrap();
        completion.wait().unwrap();
        mapping.join().unwrap();
        assert!(progressed.is_ok(), "independent NPU region was stalled");
    });
}
#[test]
fn cancellation_drains_running_work_and_prevents_dependents() {
    let runtime = Runtime::new().unwrap();
    let buffer = buffer(&runtime);
    let (tx, rx) = mpsc::sync_channel(1);
    let rx = Mutex::new(rx);
    let producer = mock(&runtime, Engine::Gpu, &buffer, Access::Write, move || {
        rx.lock().unwrap().recv().unwrap();
        Ok(())
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let consumer = mock(&runtime, Engine::Npu, &buffer, Access::Read, move || {
        count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let first = producer.submit().unwrap();
    let second = consumer.submit().unwrap();
    second.cancel();
    tx.send(()).unwrap();
    first.wait().unwrap();
    assert!(matches!(second.wait(), Err(Error::Cancelled)));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}
#[test]
fn retained_observers_apply_bounded_backpressure() {
    let runtime = Runtime::with_options(RuntimeOptions {
        graph_slots: 1,
        ..Default::default()
    })
    .unwrap();
    let buffer = buffer(&runtime);
    let graph = mock(&runtime, Engine::Gpu, &buffer, Access::Read, || Ok(()));
    let done = graph.submit().unwrap();
    done.wait().unwrap();
    assert!(matches!(graph.submit(), Err(Error::Busy(_))));
}
#[test]
fn view_bounds_and_writable_alias_contracts_are_rejected() {
    let runtime = Runtime::new().unwrap();
    let buffer = buffer(&runtime);
    assert!(buffer.slice(0..257).is_err());
    assert!(buffer.slice(128..256).unwrap().slice(0..129).is_err());
    let binding = BindingContract {
        bytes: 128,
        alignment: 4,
        access: Access::ReadWrite,
        layout: "bytes".into(),
    };
    let contract = KernelContract {
        bindings: vec![binding.clone(), binding],
        constants: vec![],
    };
    assert!(
        contract
            .check(&[buffer.view(), buffer.slice(0..128).unwrap()])
            .is_err()
    );
    assert!(
        contract
            .check(&[
                buffer.slice(0..128).unwrap(),
                buffer.slice(128..256).unwrap()
            ])
            .is_ok()
    );
}

#[test]
fn a_completed_single_slot_can_be_reused_immediately() {
    let runtime = Runtime::with_options(RuntimeOptions {
        graph_slots: 1,
        ..Default::default()
    })
    .unwrap();
    let buffer = buffer(&runtime);
    let graph = mock(&runtime, Engine::Gpu, &buffer, Access::Read, || Ok(()));
    for _ in 0..256 {
        graph.submit().unwrap().wait().unwrap();
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "intentionally quarantines an uncertain native allocation"
)]
fn uncertain_failure_poison_is_shared_and_resources_are_quarantined() {
    let runtime = Runtime::new().unwrap();
    let buffer = buffer(&runtime);
    let weak = Arc::downgrade(&buffer.storage);
    let alias = buffer.clone();
    let graph = mock(&runtime, Engine::Gpu, &buffer, Access::Write, || {
        Err(crate::Error::Backend {
            backend: "test",
            operation: "partial submit",
            code: -2,
            message: "injected abort failure".into(),
        })
    });
    let completion = graph.submit().unwrap();
    assert!(matches!(
        completion.wait(),
        Err(crate::Error::Execution { .. })
    ));
    assert!(matches!(
        alias.try_map_read(),
        Err(crate::Error::DeviceLost(_))
    ));
    assert!(matches!(graph.submit(), Err(crate::Error::DeviceLost(_))));
    drop(completion);
    drop(graph);
    drop(alias);
    drop(buffer);
    drop(runtime);
    assert!(weak.upgrade().is_some());
}

#[test]
#[cfg(feature = "npu")]
#[ignore = "requires XDNA2, shared ABI 1 and HRX_TEST_NPU_DIR; prints paired backend baseline"]
fn heterogeneous_latency_against_direct_backend() -> Result<()> {
    use std::time::Instant;
    let directory = std::path::PathBuf::from(
        std::env::var_os("HRX_TEST_NPU_DIR")
            .ok_or_else(|| crate::Error::Message("set HRX_TEST_NPU_DIR".into()))?,
    );
    let runtime = Runtime::new()?;
    let program = unsafe { runtime.npu(0)?.load_program(directory.join("x.xclbin")) }?;
    let bytes = 1 << 20;
    let binding = |bytes, access| BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "passthrough".into(),
    };
    let kernel = unsafe {
        program.kernel(
            &std::fs::read(directory.join("x.bin"))?,
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
    graph.fill(a.view(), 0x3d)?;
    graph.npu(&kernel, &[a.view(), b.view(), c.view()])?;
    let graph = graph.prepare()?;
    let direct = || -> Result<()> {
        for operation in &graph.inner.operations {
            operation.run()?;
        }
        Ok(())
    };
    for _ in 0..10 {
        graph.submit()?.wait()?;
        direct()?;
    }
    let mut scheduled = Vec::with_capacity(100);
    let mut baseline = Vec::with_capacity(100);
    let before = runtime.statistics();
    for iteration in 0..100 {
        if iteration % 2 == 0 {
            let start = Instant::now();
            direct()?;
            baseline.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            graph.submit()?.wait()?;
            scheduled.push(start.elapsed().as_secs_f64());
        } else {
            let start = Instant::now();
            graph.submit()?.wait()?;
            scheduled.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            direct()?;
            baseline.push(start.elapsed().as_secs_f64());
        }
    }
    baseline.sort_by(f64::total_cmp);
    scheduled.sort_by(f64::total_cmp);
    let after = runtime.statistics();
    assert_eq!(before.allocations, after.allocations);
    assert_eq!(before.imports, after.imports);
    assert_eq!(after.copied_bytes, 0);
    assert!(c.map_read()?.iter().all(|&b| b == 0x3d));
    println!(
        "paired_direct_p50_us={:.3} scheduled_p50_us={:.3} ratio={:.3} scheduled_p95_us={:.3}",
        baseline[50] * 1e6,
        scheduled[50] * 1e6,
        scheduled[50] / baseline[50],
        scheduled[95] * 1e6
    );
    Ok(())
}
