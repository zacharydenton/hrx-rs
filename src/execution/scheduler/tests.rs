use super::super::{
    Access, Engine, NativeSession, Runtime,
    tests::{buffer, mock},
};
use super::*;
use std::{
    sync::{atomic::AtomicUsize, mpsc},
    time::Duration,
};

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
