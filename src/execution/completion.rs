use crate::{Error, Result};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

pub(super) struct State {
    pub done: bool,
    pub failure: Option<Arc<Error>>,
    pub cancelled: bool,
    wakers: [Option<Waker>; 4],
    overflow_wakers: Vec<Waker>,
}
pub(super) struct Signal {
    pub state: Mutex<State>,
    changed: Condvar,
    pub cancel: AtomicBool,
}
impl Signal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                done: true,
                failure: None,
                cancelled: false,
                wakers: std::array::from_fn(|_| None),
                overflow_wakers: Vec::new(),
            }),
            changed: Condvar::new(),
            cancel: AtomicBool::new(false),
        })
    }
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.done = false;
        state.failure = None;
        state.cancelled = false;
        state.wakers.fill(None);
        state.overflow_wakers.clear();
        self.cancel.store(false, Ordering::Release);
    }
    pub fn finish(&self, failure: Option<Arc<Error>>) {
        let (wakers, overflow) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.failure = failure;
            state.cancelled = self.cancel.load(Ordering::Acquire);
            state.done = true;
            (
                std::mem::replace(&mut state.wakers, std::array::from_fn(|_| None)),
                std::mem::take(&mut state.overflow_wakers),
            )
        };
        self.changed.notify_all();
        for waker in wakers.into_iter().flatten().chain(overflow) {
            // A task-supplied waker must not take down an execution worker.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}
fn outcome(state: &State) -> Result<()> {
    if let Some(message) = &state.failure {
        return Err(Error::Execution {
            source: message.clone(),
        });
    }
    if state.cancelled {
        return Err(Error::Cancelled);
    }
    Ok(())
}
/// An owned completion observer. Dropping it does not cancel submitted work.
///
/// Polling never waits for a device. Clones may be waited on by different tasks.
#[derive(Clone)]
pub struct Completion {
    pub(super) signal: Arc<Signal>,
    pub(super) core: std::sync::Weak<super::scheduler::Core>,
}
impl Completion {
    /// Whether the operation reached a terminal state, including failure.
    pub fn is_complete(&self) -> bool {
        self.signal
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .done
    }
    /// Wait for device completion, propagating execution failure.
    /// The caller may execute ready regions from this submission to avoid a
    /// worker handoff. Engine limits and all dependency checks still apply.
    pub fn wait(&self) -> Result<()> {
        if let Some(core) = self.core.upgrade() {
            super::scheduler::help(&core, &self.signal);
        }
        let mut state = self.signal.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.done {
            state = self
                .signal
                .changed
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        outcome(&state)
    }
    /// Wait at most `timeout`. `false` leaves execution and ownership intact.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let start = Instant::now();
        let mut state = self.signal.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.done {
            let Some(left) = timeout.checked_sub(start.elapsed()) else {
                return Ok(false);
            };
            let (next, _) = self
                .signal
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }
        outcome(&state)?;
        Ok(true)
    }
    /// Cancel unscheduled nodes. Running native work is drained before completion.
    /// This request does not revoke resources or abort an unrelated submission.
    pub fn cancel(&self) {
        self.signal.cancel.store(true, Ordering::Release);
    }
}
impl Future for Completion {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.signal.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.done {
            return Poll::Ready(outcome(&state));
        }
        if !state
            .wakers
            .iter()
            .flatten()
            .chain(&state.overflow_wakers)
            .any(|w| w.will_wake(context.waker()))
        {
            if let Some(slot) = state.wakers.iter_mut().find(|slot| slot.is_none()) {
                *slot = Some(context.waker().clone());
            } else {
                state.overflow_wakers.push(context.waker().clone());
            }
        }
        Poll::Pending
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timeout_does_not_cancel_and_detach_retains_state() {
        let signal = Signal::new();
        signal.reset();
        let completion = Completion {
            signal: signal.clone(),
            core: std::sync::Weak::new(),
        };
        assert!(!completion.wait_timeout(Duration::ZERO).unwrap());
        assert!(!signal.cancel.load(Ordering::Acquire));
        drop(completion);
        signal.finish(None);
        Completion {
            signal,
            core: std::sync::Weak::new(),
        }
        .wait()
        .unwrap();
    }
    #[test]
    fn future_wakeup_and_error() {
        use std::task::Wake;
        struct Counter(std::sync::atomic::AtomicUsize);
        impl Wake for Counter {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let counter = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        let mut context = Context::from_waker(&waker);
        let signal = Signal::new();
        signal.reset();
        let mut completion = Completion {
            signal: signal.clone(),
            core: std::sync::Weak::new(),
        };
        assert!(Pin::new(&mut completion).poll(&mut context).is_pending());
        signal.finish(Some(Arc::new(Error::DeviceLost(
            "injected native failure".into(),
        ))));
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        assert!(matches!(
            Pin::new(&mut completion).poll(&mut context),
            Poll::Ready(Err(Error::Execution { .. }))
        ));
    }
}
