use super::{
    Access, BufferView, Engine,
    graph::{NodeState, Prepared},
};
use std::sync::{Arc, Condvar, Mutex, atomic::Ordering};

pub(super) struct Pending {
    pub graph: Arc<Prepared>,
    pub slot: usize,
    pub order: u64,
}
pub(super) struct Scheduler {
    pub pending: Vec<Pending>,
    pub capacity: usize,
    pub next_order: u64,
    pub shutdown: bool,
    running: [bool; 3],
}
pub(super) struct Core {
    pub counters: super::statistics::Counters,
    pub state: Mutex<Scheduler>,
    pub changed: Condvar,
    pub host_changed: Condvar,
}
impl Core {
    pub fn new(capacity: usize) -> Self {
        Self {
            counters: Default::default(),
            state: Mutex::new(Scheduler {
                pending: Vec::with_capacity(capacity),
                capacity,
                next_order: 0,
                shutdown: false,
                running: [false; 3],
            }),
            changed: Condvar::new(),
            host_changed: Condvar::new(),
        }
    }
}
impl Scheduler {
    pub fn conflicts(&self, view: &BufferView, access: Access) -> bool {
        self.pending.iter().any(|job| {
            job.graph.uses.iter().any(|other| {
                view.conflicts(&other.view) && (access.writes() || other.access.writes())
            })
        })
    }
    fn blocked(&self, job: &Pending) -> bool {
        self.pending.iter().any(|earlier| {
            earlier.order < job.order
                && earlier
                    .graph
                    .uses
                    .iter()
                    .any(|a| job.graph.uses.iter().any(|b| a.conflicts(b)))
        })
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
            action = Some(Action::Finish(scheduler.pending.remove(job_index), failure));
            break;
        }
        if scheduler.blocked(job) {
            continue;
        }
        for (index, operation) in job.graph.operations.iter().enumerate() {
            if !scheduler.running[operation.engine() as usize]
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
        scheduler.running[graph.operations[*node].engine() as usize] = true;
    }
    action
}

pub(super) fn worker(core: Arc<Core>, _engine: Engine) {
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
            let mut _scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
            _scheduler.running[graph.operations[node].engine() as usize] = false;
            if let Some(message) = &failure {
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
            if failure.is_some() {
                // Uncertain completion must not release imported pages or
                // native graph structures. Keep the quarantine until exit.
                std::mem::forget(job.graph);
            }
        }
    }
}
