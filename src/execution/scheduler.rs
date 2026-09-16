use super::{
    Access, BufferView,
    graph::{NodeState, Prepared},
};
use std::sync::{Arc, Condvar, Mutex, atomic::Ordering};

#[cfg(test)]
mod tests;

pub(super) struct Pending {
    sequence: u64,
    pub graph: Arc<Prepared>,
    pub slot: usize,
    // Number of earlier, still-pending submissions with conflicting uses.
    blockers: usize,
    dependencies: Vec<super::Completion>,
}
pub(super) struct Scheduler {
    pub pending: Vec<Pending>,
    pub capacity: usize,
    pub shutdown: bool,
    running: [bool; 5],
    sequence: u64,
    native: std::collections::VecDeque<u64>,
    native_active: bool,
}
pub(super) struct Core {
    pub tracer: super::trace::Tracer,
    pub counters: super::statistics::Counters,
    pub state: Mutex<Scheduler>,
    pub changed: Condvar,
    pub host_changed: Condvar,
}
impl Core {
    pub fn new(capacity: usize) -> Self {
        Self {
            tracer: Default::default(),
            counters: Default::default(),
            state: Mutex::new(Scheduler {
                pending: Vec::with_capacity(capacity),
                capacity,
                shutdown: false,
                running: [false; 5],
                sequence: 0,
                native: Default::default(),
                native_active: false,
            }),
            changed: Condvar::new(),
            host_changed: Condvar::new(),
        }
    }
}
impl Scheduler {
    pub fn occupied(&self) -> usize {
        self.pending.len() + self.native.len() + usize::from(self.native_active)
    }
    fn sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("scheduler sequence exhausted");
        sequence
    }
    pub fn conflicts(&self, view: &BufferView, access: Access) -> bool {
        self.pending.iter().any(|job| {
            job.graph.uses.iter().any(|other| {
                view.conflicts(&other.view) && (access.writes() || other.access.writes())
            })
        })
    }
    pub fn enqueue(
        &mut self,
        graph: Arc<Prepared>,
        slot: usize,
        dependencies: Vec<super::Completion>,
    ) {
        let blockers = self
            .pending
            .iter()
            .filter(|earlier| super::dependencies::conflicts(&earlier.graph.uses, &graph.uses))
            .count();
        let sequence = self.sequence();
        self.pending.push(Pending {
            sequence,
            graph,
            slot,
            blockers,
            dependencies,
        });
    }
    fn remove(&mut self, index: usize) -> Pending {
        let job = self.pending.remove(index);
        // Vec order is submission order. Only later submissions counted this
        // job, including when cancellation removes a blocked job out of order.
        for later in &mut self.pending[index..] {
            if super::dependencies::conflicts(&job.graph.uses, &later.graph.uses) {
                later.blockers -= 1;
            }
        }
        job
    }
}

/// A synchronous native stage reserves the same compute lane as prepared work.
/// Earlier submissions drain first; later compute cannot overtake the waiter.
pub(super) struct NativeLease(Arc<Core>);
impl NativeLease {
    pub fn acquire(core: &Arc<Core>) -> crate::Result<Self> {
        let mut state = core.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.occupied() >= state.capacity {
            return Err(crate::Error::Busy(
                "runtime submission capacity exhausted".into(),
            ));
        }
        let ticket = state.sequence();
        state.native.push_back(ticket);
        core.changed.notify_all();
        while state.native.front() != Some(&ticket)
            || state.running[super::GpuLane::Compute as usize]
            || state.pending.iter().any(|job| job.sequence < ticket)
        {
            state = core.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.native.pop_front();
        state.native_active = true;
        state.running[super::GpuLane::Compute as usize] = true;
        Ok(Self(core.clone()))
    }
}

impl Drop for NativeLease {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.native_active = false;
        state.running[super::GpuLane::Compute as usize] = false;
        self.0.changed.notify_all();
    }
}
enum Action {
    Run(Arc<Prepared>, usize, usize),
    Finish(Pending, Option<Arc<crate::Error>>),
}
fn next_action(
    scheduler: &mut Scheduler,
    target: Option<&Arc<super::completion::Signal>>,
) -> Option<Action> {
    let mut action = None;
    for job_index in 0..scheduler.pending.len() {
        let job = &scheduler.pending[job_index];
        if target.is_some_and(|signal| !Arc::ptr_eq(signal, &job.graph.signals[job.slot])) {
            continue;
        }
        let mut slots = job.graph.slots.lock().unwrap_or_else(|e| e.into_inner());
        let slot = &mut slots[job.slot];
        let mut waiting = false;
        for dependency in &job.dependencies {
            match dependency.result() {
                None => waiting = true,
                Some(Ok(())) => {}
                Some(Err(error)) => {
                    slot.failure = Some(Arc::new(error.context("inference dependency failed")));
                    break;
                }
            }
        }
        if job.graph.signals[job.slot].cancel.load(Ordering::Acquire) || slot.failure.is_some() {
            for node in &mut slot.nodes {
                if *node == NodeState::Pending {
                    *node = NodeState::Done;
                }
            }
        }
        if slot.nodes.iter().all(|node| *node == NodeState::Done) {
            let failure = slot.failure.take();
            // Keep occupied until signal publication, preventing a
            // detached completion from being reset during finish.
            drop(slots);
            action = Some(Action::Finish(scheduler.remove(job_index), failure));
            break;
        }
        if job.blockers != 0 || waiting {
            continue;
        }
        for (index, operation) in job.graph.operations.iter().enumerate() {
            let native_precedes = operation.queue() == super::GpuLane::Compute as usize
                && scheduler
                    .native
                    .front()
                    .is_some_and(|ticket| job.sequence > *ticket);
            if !scheduler.running[operation.queue()]
                && !native_precedes
                && slot.nodes[index] == NodeState::Pending
                && operation
                    .dependencies
                    .iter()
                    .all(|&dep| slot.nodes[dep] == NodeState::Done)
            {
                slot.nodes[index] = NodeState::Running;
                action = Some(Action::Run(job.graph.clone(), job.slot, index));
                break;
            }
        }
        if action.is_some() {
            break;
        }
    }
    if let Some(Action::Run(graph, _, node)) = &action {
        scheduler.running[graph.operations[*node].queue()] = true;
    }
    action
}

pub(super) fn worker(core: Arc<Core>) {
    loop {
        let action = {
            let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(action) = next_action(&mut scheduler, None) {
                    break action;
                }
                if scheduler.shutdown && scheduler.pending.is_empty() {
                    return;
                }
                scheduler = core
                    .changed
                    .wait(scheduler)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        perform(&core, action);
    }
}

// Blocking callers can help their own submission, with the exact same engine
// reservations, dependency checks and failure handling as background workers.
// Never called by Future::poll or a bounded timeout wait.
pub(super) fn help(core: &Arc<Core>, signal: &Arc<super::completion::Signal>) {
    loop {
        let action = {
            let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
            next_action(&mut scheduler, Some(signal))
        };
        match action {
            Some(action) => perform(core, action),
            None => return,
        }
    }
}
fn perform(core: &Arc<Core>, action: Action) {
    match action {
        Action::Run(graph, slot_index, node) => {
            let started = std::time::Instant::now();
            graph.signals[slot_index].started(started);
            // A serial chain needs one worker. Wake a peer only when the
            // graph can use both engines independently.
            if graph.parallel {
                core.changed.notify_all();
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                graph.operations[node].run()
            }));
            let failure = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(Arc::new(
                    error.context(format!("executing graph region {node}")),
                )),
                Err(_) => Some(Arc::new(crate::Error::DeviceLost(
                    "native worker panicked".into(),
                ))),
            };
            core.tracer.record(
                started,
                started.elapsed(),
                graph.operations[node].queue(),
                graph.operations[node].copy_bytes,
                failure.is_some(),
            );
            let mut _scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
            _scheduler.running[graph.operations[node].queue()] = false;
            if let Some(message) = &failure {
                graph.quarantined.store(true, Ordering::Release);
                // The backend may have partially submitted work. Every
                // alias must reject subsequent access, including reads.
                for access in &graph.uses {
                    access
                        .view
                        .buffer
                        .storage
                        .host
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .poison = Some(message.to_string());
                }
            }
            {
                let mut slots = graph.slots.lock().unwrap_or_else(|e| e.into_inner());
                slots[slot_index].nodes[node] = NodeState::Done;
                if failure.is_some() {
                    slots[slot_index].failure = failure;
                }
            }
            if _scheduler.pending.len() > 1 {
                core.changed.notify_all();
            }
            drop(_scheduler);
        }
        Action::Finish(job, failure) => {
            job.graph.slots.lock().unwrap_or_else(|e| e.into_inner())[job.slot].occupied = false;
            core.counters.completions.fetch_add(1, Ordering::Relaxed);
            job.graph.signals[job.slot].finish(failure.clone());
            core.host_changed.notify_all();
            core.changed.notify_all();
            if job.graph.quarantined.load(Ordering::Acquire) {
                // Uncertain completion must not release imported pages or
                // native graph structures. Keep the quarantine until exit.
                std::mem::forget(job.graph);
            }
        }
    }
}
