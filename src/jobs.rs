//! Bounded host orchestration for asynchronous FFI callers and streaming pipelines.
//! Jobs may block on coordinated GPU completions without occupying the caller's
//! scheduler. Cancellation never interrupts native work: running work drains.
use crate::{Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

type Work = Box<dyn FnOnce() + Send>;
struct Owner {
    sender: Option<mpsc::SyncSender<Work>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
    }
}
/// Shared bounded worker pool. Capacity counts queued and running work through
/// its completion callback; retained application results are not counted.
#[derive(Clone)]
pub struct JobPool {
    owner: Arc<Mutex<Owner>>,
    live: Arc<AtomicUsize>,
    capacity: usize,
}
#[derive(Default)]
struct State {
    cancelled: bool,
    finished: bool,
}
/// Cooperative cancellation/observation. Dropping a handle does not abort work.
#[derive(Clone)]
pub struct JobHandle(Arc<Mutex<State>>);
impl JobHandle {
    /// Request cancellation. Returns false if native work already finished.
    pub fn cancel(&self) -> bool {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.finished {
            return false;
        }
        state.cancelled = true;
        true
    }
    /// Whether cancellation has been requested; a running closure may check at
    /// safe stage boundaries after draining any native submissions.
    pub fn is_cancelled(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).cancelled
    }
    /// Whether work has drained (the notification callback may still be running).
    pub fn is_finished(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).finished
    }
}
struct Permit(Arc<AtomicUsize>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
impl JobPool {
    /// Start a fixed number of workers and limit all outstanding jobs.
    pub fn new(workers: usize, capacity: usize) -> Result<Self> {
        if workers == 0 || capacity < workers {
            return Err(Error::Message(
                "job capacity must be >= nonzero worker count".into(),
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Work>(capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut owner = Owner {
            sender: Some(sender),
            workers: Vec::new(),
        };
        for index in 0..workers {
            let receiver = receiver.clone();
            owner.workers.push(
                std::thread::Builder::new()
                    .name(format!("hrx-job-{index}"))
                    .spawn(move || {
                        loop {
                            let work = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            match work {
                                Ok(work) => work(),
                                Err(_) => break,
                            }
                        }
                    })
                    .map_err(|e| Error::Message(e.to_string()))?,
            );
        }
        Ok(Self {
            owner: Arc::new(Mutex::new(owner)),
            live: Arc::new(AtomicUsize::new(0)),
            capacity,
        })
    }
    /// Current queued/running/notification jobs.
    pub fn outstanding(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }
    /// Submit without waiting for capacity. The callback runs once, including
    /// cancellation and panics. A cancellation during work discards its result
    /// only after the closure returns; native ownership must drain inside work.
    pub fn submit<T: Send + 'static>(
        &self,
        work: impl FnOnce(JobHandle) -> Result<T> + Send + 'static,
        completed: impl FnOnce(Result<T>) + Send + 'static,
    ) -> Result<JobHandle> {
        self.live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.capacity).then_some(n + 1)
            })
            .map_err(|_| Error::Busy("job capacity is occupied".into()))?;
        let permit = Permit(self.live.clone());
        let handle = JobHandle(Arc::new(Mutex::new(State::default())));
        let task_handle = handle.clone();
        let task: Work = Box::new(move || {
            let _permit = permit;
            let result = if task_handle.is_cancelled() {
                Err(Error::Cancelled)
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(task_handle.clone())))
                    .unwrap_or_else(|_| Err(Error::Message("job panicked".into())))
            };
            let result = {
                let mut state = task_handle.0.lock().unwrap_or_else(|e| e.into_inner());
                state.finished = true;
                if state.cancelled {
                    Err(Error::Cancelled)
                } else {
                    result
                }
            };
            // A callback panic must not kill a worker or strand queued jobs.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| completed(result)));
        });
        self.owner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sender
            .as_ref()
            .unwrap()
            .try_send(task)
            .map_err(|_| Error::Busy("job queue unavailable".into()))?;
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_cancellation_drain_and_panic_recovery() {
        let pool = JobPool::new(1, 2).unwrap();
        let (started, start) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (done, results) = mpsc::channel();
        let done1 = done.clone();
        let first = pool
            .submit(
                move |_| {
                    started.send(()).unwrap();
                    released.recv().unwrap();
                    Ok(7)
                },
                move |r| done1.send(r).unwrap(),
            )
            .unwrap();
        start.recv().unwrap();
        let second = pool
            .submit(
                |_| panic!("cancelled work must not execute"),
                move |r| done.send(r).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            pool.submit(|_| Ok(()), |_| {}),
            Err(Error::Busy(_))
        ));
        first.cancel();
        second.cancel();
        assert!(!first.is_finished());
        assert_eq!(pool.outstanding(), 2);
        release.send(()).unwrap();
        assert!(matches!(results.recv().unwrap(), Err(Error::Cancelled)));
        assert!(matches!(results.recv().unwrap(), Err(Error::Cancelled)));
        drop(pool); // joins workers and finishes callbacks
        assert!(first.is_finished());
        assert!(!first.cancel());
        let pool = JobPool::new(1, 3).unwrap();
        let (send, recv) = mpsc::channel();
        pool.submit::<()>(|_| panic!("test panic"), move |r| send.send(r).unwrap())
            .unwrap();
        let callback_panic = pool
            .submit(|_| Ok(()), |_| panic!("test callback panic"))
            .unwrap();
        let (send, recovered) = mpsc::channel();
        let after_panic = pool
            .submit(|_| Ok(42), move |result| send.send(result).unwrap())
            .unwrap();
        assert!(recv.recv().unwrap().is_err());
        assert_eq!(recovered.recv().unwrap().unwrap(), 42);
        drop(pool);
        assert!(callback_panic.is_finished());
        assert!(after_panic.is_finished());
    }
}
