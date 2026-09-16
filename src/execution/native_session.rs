use super::{GpuLane, Runtime, scheduler::NativeLease};
use crate::{Error, Result};
use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::Arc,
};

thread_local! {
    static ACTIVE: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct Entered(usize);
impl Entered {
    fn new(runtime: &Runtime) -> Result<Self> {
        let id = Arc::as_ptr(&runtime.inner.core) as usize;
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.contains(&id) {
                return Err(Error::Busy(
                    "recursive native session in the same runtime".into(),
                ));
            }
            active.push(id);
            Ok(Self(id))
        })
    }
}
impl Drop for Entered {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.borrow_mut().retain(|id| *id != self.0));
    }
}

/// Owned stream-bound model state scheduled alongside prepared GPU graphs.
///
/// Stages execute synchronously on the calling thread and may borrow inputs and
/// progress callbacks. They reserve the compute lane, without blocking unrelated
/// upload/download/NPU lanes. Earlier submissions drain before admission; later
/// compute cannot overtake a waiting stage. Admission uses the runtime's bounded
/// submission capacity. This is a stage boundary, not asynchronous inference or
/// separate transfer scheduling within a stage.
pub struct NativeSession<T> {
    runtime: Runtime,
    state: Option<T>,
    fence: fn(&mut T) -> Result<()>,
}

impl<T> NativeSession<T> {
    /// Own a model's private native state and its completion fence.
    ///
    /// # Safety
    /// State must own every private stream, graph and allocation used by its
    /// stages, on this runtime's selected device. `fence` must drain all those
    /// streams before returning success, including after a failed stage. No
    /// concurrent external access to this state is allowed. Construction must
    /// leave no pending work; subsequent work must go through [`Self::run`].
    pub unsafe fn new(runtime: &Runtime, state: T, fence: fn(&mut T) -> Result<()>) -> Self {
        Self {
            runtime: runtime.clone(),
            state: Some(state),
            fence,
        }
    }

    /// Inspect idle model metadata. Returns `DeviceLost` after quarantine.
    pub fn state(&self) -> Result<&T> {
        self.state
            .as_ref()
            .ok_or_else(|| Error::DeviceLost("native session is quarantined".into()))
    }

    /// Reserve compute, run a borrowed stage, then fence on every exit path.
    /// Client errors may be returned as `R = Result<_, ClientError>` and do not
    /// poison a successfully drained session. A panic is resumed after fencing.
    /// A failed or panicking fence quarantines the entire owner, including its
    /// allocation charges, and rejects all subsequent stages.
    ///
    /// # Safety
    /// The stage must retain all resources until the fence, submit only on the
    /// owned streams, and not expose pending raw uses outside this session.
    /// It must not transfer private native owners out of the state, including
    /// through its return value; quarantine must retain the complete owner.
    /// Stages may access only private native storage, not coordinated buffers;
    /// use `Graph::gpu_scoped` for operations needing tracked memory hazards.
    /// Do not wait for coordinated compute or another native session in this
    /// runtime from the callback: this stage already holds its compute lane.
    pub unsafe fn run<R>(&mut self, stage: impl FnOnce(&mut T) -> R) -> Result<R> {
        self.state()?;
        let _entered = Entered::new(&self.runtime)?;
        let _lane = NativeLease::acquire(&self.runtime.inner.core)?;
        let started = std::time::Instant::now();
        let state = self.state.as_mut().expect("checked before admission");
        let outcome = catch_unwind(AssertUnwindSafe(|| stage(state)));
        let completion = catch_unwind(AssertUnwindSafe(|| (self.fence)(state)));
        let failed = !matches!(completion, Ok(Ok(())));
        self.runtime.inner.core.tracer.record(
            started,
            started.elapsed(),
            GpuLane::Compute as usize,
            0,
            failed || outcome.is_err(),
        );
        if failed {
            // Keep graphs, stream storage, checkpoint mappings and charges alive
            // beneath possibly in-flight native work. Never retry this owner.
            std::mem::forget(self.state.take());
        }
        match outcome {
            Err(panic) => resume_unwind(panic),
            Ok(value) => match completion {
                Err(panic) => resume_unwind(panic),
                Ok(result) => result.map(|()| value),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct State {
        fences: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        fail: bool,
        panic: bool,
    }
    impl Drop for State {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn session(runtime: &Runtime) -> NativeSession<State> {
        // SAFETY: mock state performs no device work.
        unsafe {
            NativeSession::new(
                runtime,
                State {
                    fences: Arc::new(AtomicUsize::new(0)),
                    drops: Arc::new(AtomicUsize::new(0)),
                    fail: false,
                    panic: false,
                },
                |state| {
                    state.fences.fetch_add(1, Ordering::SeqCst);
                    assert!(!state.panic, "fence panic");
                    if state.fail {
                        Err(Error::DeviceLost("fence failed".into()))
                    } else {
                        Ok(())
                    }
                },
            )
        }
    }

    #[test]
    fn borrowed_stages_drain_errors_and_panics_then_reuse() {
        let runtime = Runtime::new().unwrap();
        let mut session = session(&runtime);
        let drops = session.state().unwrap().drops.clone();
        let mut output = 0;
        // SAFETY: all test callbacks only mutate host state.
        unsafe {
            session.run(|_| {
                output = 42;
                Err::<(), _>("client error")
            })
        }
        .unwrap()
        .unwrap_err();
        assert_eq!(output, 42);
        assert!(
            catch_unwind(AssertUnwindSafe(|| unsafe {
                session.run::<()>(|_| panic!("stage panic"))
            }))
            .is_err()
        );
        assert_eq!(unsafe { session.run(|_| 7) }.unwrap(), 7);
        assert_eq!(session.state().unwrap().fences.load(Ordering::SeqCst), 3);
        drop(session);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore = "deliberate quarantine leak")]
    fn uncertain_fences_keep_owners_and_reject_retry() {
        for panic in [false, true] {
            let runtime = Runtime::new().unwrap();
            let mut session = session(&runtime);
            let drops = session.state().unwrap().drops.clone();
            let fences = session.state().unwrap().fences.clone();
            let result = catch_unwind(AssertUnwindSafe(|| unsafe {
                session.run(|state| {
                    state.fail = true;
                    state.panic = panic;
                })
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert!(matches!(
                unsafe { session.run::<()>(|_| unreachable!()) },
                Err(Error::DeviceLost(_))
            ));
            drop(session);
            assert_eq!(fences.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            // The failed owner is quarantined, not the entire runtime lane.
            let mut next = super::tests::session(&runtime);
            unsafe { next.run(|_| ()) }.unwrap();
        }
    }

    #[test]
    fn recursive_admission_returns_busy_instead_of_deadlocking() {
        let runtime = Runtime::new().unwrap();
        let mut outer = session(&runtime);
        let mut inner = session(&runtime);
        let result = unsafe { outer.run(|_| inner.run(|_| ())) }.unwrap();
        assert!(matches!(result, Err(Error::Busy(_))));
        unsafe { inner.run(|_| ()) }.unwrap();
    }
}
