use super::super::{
    Access, Buffer, Engine, ExecutableGraph, NativeSession, Runtime,
    graph::{Backend, Operation},
    tests::{buffer, mock, runtime_without_workers},
};
use super::*;
use std::{
    sync::{atomic::AtomicUsize, mpsc},
    time::Duration,
};

#[test]
fn finalization_reserves_capacity_until_publication() {
    let runtime = runtime_without_workers();
    let core = &runtime.inner.core;
    core.state.lock().unwrap().capacity = 1;
    let graph = mock(
        &runtime,
        Engine::Gpu,
        &buffer(&runtime),
        Access::Write,
        || Ok(()),
    );
    let completion = graph.submit().unwrap();
    let run = next_action(&mut core.state.lock().unwrap(), None).unwrap();
    assert!(matches!(run, Action::Run(..)));
    perform(core, run);
    let finish = next_action(&mut core.state.lock().unwrap(), None).unwrap();
    assert!(matches!(finish, Action::Finish(..)));
    {
        let state = core.state.lock().unwrap();
        assert!(state.pending.is_empty());
        assert_eq!(state.occupied(), 1);
        assert!(!completion.is_complete());
    }
    assert!(matches!(
        NativeLease::acquire(core),
        Err(crate::Error::Busy(_))
    ));
    perform(core, finish);
    assert!(completion.is_complete());
    let lease = NativeLease::acquire(core).unwrap();
    assert_eq!(core.state.lock().unwrap().occupied(), 1);
    drop(lease);
    assert_eq!(core.state.lock().unwrap().occupied(), 0);
}

#[test]
fn native_waiter_runs_after_prior_submissions_before_later_compute() {
    let runtime = Runtime::new().unwrap();
    let order = Arc::new(AtomicUsize::new(0));
    let (entered, entering) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let released = Mutex::new(released);
    let first = mock(&runtime, Engine::Gpu, &buffer(&runtime), Access::Write, {
        let order = order.clone();
        move || {
            assert_eq!(order.fetch_add(1, Ordering::SeqCst), 0);
            entered.send(()).unwrap();
            released
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            Ok(())
        }
    })
    .submit()
    .unwrap();
    entering.recv_timeout(Duration::from_secs(10)).unwrap();
    let (native_entered, native_entering) = mpsc::channel();
    let (native_release, native_released) = mpsc::channel();
    let native = std::thread::spawn({
        let runtime = runtime.clone();
        let order = order.clone();
        move || {
            // SAFETY: fixture performs no native GPU operations.
            let mut session = unsafe { NativeSession::new(&runtime, (), |_| Ok(())) };
            unsafe {
                session.run(|_| {
                    assert_eq!(order.fetch_add(1, Ordering::SeqCst), 1);
                    native_entered.send(()).unwrap();
                    native_released
                        .recv_timeout(Duration::from_secs(10))
                        .unwrap();
                })
            }
            .unwrap();
        }
    });
    let core = &runtime.inner.core;
    let state = core.state.lock().unwrap();
    let (state, timeout) = core
        .changed
        .wait_timeout_while(state, Duration::from_secs(10), |state| {
            state.native.is_empty()
        })
        .unwrap();
    assert!(!timeout.timed_out());
    assert_eq!(state.occupied(), 2);
    drop(state);
    let later = mock(&runtime, Engine::Gpu, &buffer(&runtime), Access::Write, {
        let order = order.clone();
        move || {
            assert_eq!(order.fetch_add(1, Ordering::SeqCst), 2);
            Ok(())
        }
    })
    .submit()
    .unwrap();
    release.send(()).unwrap();
    native_entering
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert!(first.is_complete());
    assert!(!later.is_complete());
    native_release.send(()).unwrap();
    native.join().unwrap();
    later.wait().unwrap();
    assert_eq!(order.load(Ordering::SeqCst), 3);
}

#[test]
fn cooperative_handoff_retains_capacity_and_reacquires_before_later_arrivals() {
    let runtime = Runtime::with_options(super::super::RuntimeOptions {
        max_submissions: 3,
        ..Default::default()
    })
    .unwrap();
    let core = &runtime.inner.core;
    let (during, arriving) = mpsc::channel();
    let order = Arc::new(AtomicUsize::new(0));
    let first = mock(&runtime, Engine::Gpu, &buffer(&runtime), Access::Write, {
        let runtime = runtime.clone();
        let order = order.clone();
        move || {
            assert_eq!(order.fetch_add(1, Ordering::SeqCst), 1);
            let state = runtime.inner.core.state.lock().unwrap();
            assert_eq!(state.occupied(), 2);
            assert_eq!(state.native.len(), 1, "yield retained its ticket");
            drop(state);
            let later = mock(&runtime, Engine::Gpu, &buffer(&runtime), Access::Write, {
                let order = order.clone();
                move || {
                    assert_eq!(order.fetch_add(1, Ordering::SeqCst), 3);
                    Ok(())
                }
            })
            .submit()
            .unwrap();
            // Native yield must retain its capacity even while waiting.
            assert!(matches!(
                NativeLease::acquire(&runtime.inner.core),
                Err(crate::Error::Busy(_))
            ));
            during.send(later).unwrap();
            Ok(())
        }
    });
    // SAFETY: all callbacks are host-only scheduling fixtures.
    let mut session = unsafe { NativeSession::new(&runtime, (), |_| Ok(())) };
    let later = unsafe {
        session.run_cooperative(|_, cooperate| {
            let done = first.submit().unwrap();
            assert!(!done.is_complete());
            assert!(
                cooperate
                    .checkpoint(|| {
                        assert_eq!(order.fetch_add(1, Ordering::SeqCst), 0);
                        assert!(!done.is_complete(), "compute started before the fence");
                        Ok(())
                    })
                    .unwrap()
            );
            assert!(done.is_complete());
            assert_eq!(order.fetch_add(1, Ordering::SeqCst), 2);
            let later = arriving.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(!later.is_complete(), "later job overtook resumption");
            assert_eq!(core.state.lock().unwrap().occupied(), 2);
            later
        })
    }
    .unwrap();
    later.wait().unwrap();
    assert_eq!(order.load(Ordering::SeqCst), 4);
    assert_eq!(core.state.lock().unwrap().occupied(), 0);
}

#[test]
fn cooperative_stage_yields_to_another_native_waiter() {
    let runtime = Runtime::new().unwrap();
    let core = &runtime.inner.core;
    let order = Arc::new(AtomicUsize::new(0));
    let mut outer = unsafe { NativeSession::new(&runtime, (), |_| Ok(())) };
    unsafe {
        outer.run_cooperative(|_, cooperate| {
            let native = std::thread::spawn({
                let runtime = runtime.clone();
                let order = order.clone();
                move || {
                    let mut inner = NativeSession::new(&runtime, (), |_| Ok(()));
                    inner
                        .run(|_| assert_eq!(order.fetch_add(1, Ordering::SeqCst), 0))
                        .unwrap();
                }
            });
            let state = core.state.lock().unwrap();
            let (state, timeout) = core
                .changed
                .wait_timeout_while(state, Duration::from_secs(10), |state| {
                    state.native.is_empty()
                })
                .unwrap();
            assert!(!timeout.timed_out());
            drop(state);
            assert!(cooperate.checkpoint(|| Ok(())).unwrap());
            assert_eq!(order.fetch_add(1, Ordering::SeqCst), 1);
            native.join().unwrap();
        })
    }
    .unwrap();
}

fn chain(
    runtime: &Runtime,
    storage: &Buffer,
    name: char,
    regions: usize,
    order: &Arc<Mutex<Vec<char>>>,
) -> ExecutableGraph {
    let mut graph = mock(runtime, Engine::Gpu, storage, Access::Write, || Ok(()));
    let prepared = Arc::get_mut(&mut graph.inner).unwrap();
    prepared.operations = (0..regions)
        .map(|index| {
            let order = order.clone();
            Operation {
                lane: super::super::GpuLane::Compute,
                copy_bytes: 0,
                backend: Backend::Mock(
                    Engine::Gpu,
                    Arc::new(move || {
                        order.lock().unwrap().push(name);
                        Ok(())
                    }),
                ),
                uses: prepared.uses.clone(),
                dependencies: index.checked_sub(1).into_iter().collect(),
            }
        })
        .collect();
    for slot in prepared.slots.get_mut().unwrap() {
        slot.nodes = vec![NodeState::Pending; regions];
    }
    graph
}

fn step(core: &Arc<Core>) {
    let action = next_action(&mut core.state.lock().unwrap(), None).expect("ready action");
    perform(core, action);
}

fn drain(core: &Arc<Core>) {
    while core.state.lock().unwrap().occupied() != 0 {
        step(core);
    }
}

#[test]
fn ready_graphs_take_turns_between_regions() {
    let runtime = runtime_without_workers();
    let order = Arc::new(Mutex::new(Vec::new()));
    let a = chain(&runtime, &buffer(&runtime), 'a', 3, &order)
        .submit()
        .unwrap();
    let b = chain(&runtime, &buffer(&runtime), 'b', 1, &order)
        .submit()
        .unwrap();
    let c = chain(&runtime, &buffer(&runtime), 'c', 2, &order)
        .submit()
        .unwrap();
    drain(&runtime.inner.core);
    assert_eq!(*order.lock().unwrap(), ['a', 'b', 'c', 'a', 'c', 'a']);
    for done in [a, b, c] {
        assert!(done.is_complete());
    }
}

#[test]
fn late_short_job_gets_a_turn_even_when_long_job_has_a_blocking_helper() {
    let runtime = runtime_without_workers();
    let core = &runtime.inner.core;
    let order = Arc::new(Mutex::new(Vec::new()));
    let long = chain(&runtime, &buffer(&runtime), 'a', 4, &order)
        .submit()
        .unwrap();
    step(core);
    let short = chain(&runtime, &buffer(&runtime), 'b', 1, &order)
        .submit()
        .unwrap();
    help(core, &long.signal);
    assert_eq!(*order.lock().unwrap(), ['a']);
    help(core, &short.signal);
    assert_eq!(*order.lock().unwrap(), ['a', 'b']);
    assert!(!long.is_complete());
    drain(core);
    assert!(short.is_complete());
    assert!(long.is_complete());
}

#[test]
fn transfer_activity_does_not_reset_compute_fairness() {
    let runtime = runtime_without_workers();
    let core = &runtime.inner.core;
    let order = Arc::new(Mutex::new(Vec::new()));
    let long = chain(&runtime, &buffer(&runtime), 'a', 3, &order)
        .submit()
        .unwrap();
    let short = chain(&runtime, &buffer(&runtime), 'b', 1, &order)
        .submit()
        .unwrap();
    let running = next_action(&mut core.state.lock().unwrap(), None).unwrap();
    assert!(matches!(running, Action::Run(..)));
    let mut upload = chain(&runtime, &buffer(&runtime), 'u', 4, &order);
    for operation in &mut Arc::get_mut(&mut upload.inner).unwrap().operations {
        operation.lane = super::super::GpuLane::Upload;
    }
    let uploaded = upload.submit().unwrap();
    // Several transfers can finish while the original compute region runs.
    step(core);
    step(core);
    perform(core, running);
    step(core);
    assert_eq!(*order.lock().unwrap(), ['u', 'u', 'a', 'b']);
    drain(core);
    for done in [long, short, uploaded] {
        assert!(done.is_complete());
    }
}

#[test]
fn fairness_preserves_conflicting_submission_order() {
    let runtime = runtime_without_workers();
    let storage = buffer(&runtime);
    let order = Arc::new(Mutex::new(Vec::new()));
    let a = chain(&runtime, &storage, 'a', 3, &order).submit().unwrap();
    let b = chain(&runtime, &storage, 'b', 1, &order).submit().unwrap();
    let c = chain(&runtime, &buffer(&runtime), 'c', 2, &order)
        .submit()
        .unwrap();
    drain(&runtime.inner.core);
    assert_eq!(*order.lock().unwrap(), ['a', 'c', 'a', 'c', 'a', 'b']);
    for done in [a, b, c] {
        assert!(done.is_complete());
    }
}

#[test]
fn cancellation_and_new_arrivals_do_not_skip_ready_graphs() {
    let runtime = runtime_without_workers();
    let core = &runtime.inner.core;
    let storage = buffer(&runtime);
    let order = Arc::new(Mutex::new(Vec::new()));
    let a = chain(&runtime, &storage, 'a', 3, &order).submit().unwrap();
    let b = chain(&runtime, &storage, 'b', 1, &order).submit().unwrap();
    step(core);
    a.cancel();
    let c = chain(&runtime, &buffer(&runtime), 'c', 1, &order)
        .submit()
        .unwrap();
    drain(core);
    assert_eq!(*order.lock().unwrap(), ['a', 'b', 'c']);
    assert!(matches!(a.result(), Some(Err(crate::Error::Cancelled))));
    assert!(b.is_complete());
    assert!(c.is_complete());
}
