use super::{
    Access, BufferView, Completion, Engine, KernelContract, Runtime, RuntimeOwner,
    completion::Signal,
};
use crate::{Error, Result};
use std::sync::{Arc, Mutex};

/// Trusted, fixed GPU launch dimensions and argument contract.
#[derive(Clone)]
pub struct GpuKernel {
    pub(super) raw: crate::gpu::Kernel,
    pub(super) contract: KernelContract,
    pub(super) grid: [u32; 3],
    pub(super) block: [u32; 3],
    pub(super) runtime: Arc<RuntimeOwner>,
}
impl GpuKernel {
    /// Fixed invocation contract.
    pub fn contract(&self) -> &KernelContract {
        &self.contract
    }
}
/// An operation index, useful for diagnostics. Ordering is inferred from access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node(pub(crate) usize);
#[derive(Clone)]
pub(super) struct Use {
    pub view: BufferView,
    pub access: Access,
}
impl Use {
    #[cfg(test)]
    pub fn conflicts(&self, other: &Self) -> bool {
        self.view.conflicts(&other.view) && (self.access.writes() || other.access.writes())
    }
}
enum Description {
    Fill(BufferView, u8),
    Copy(BufferView, BufferView),
    Gpu(GpuKernel, Vec<BufferView>),
    #[cfg(feature = "npu")]
    Npu(crate::npu::NpuKernel, Vec<BufferView>),
}
struct Entry {
    operation: Description,
    uses: Vec<Use>,
}
/// A pipeline builder. Each operation declares placement and actual buffer use.
pub struct Graph {
    runtime: Runtime,
    entries: Vec<Entry>,
}
impl Graph {
    pub(super) fn new(runtime: Runtime) -> Self {
        Self {
            runtime,
            entries: Vec::new(),
        }
    }
    fn push(&mut self, operation: Description, uses: Vec<Use>) -> Result<Node> {
        for binding in &uses {
            if !Arc::ptr_eq(&binding.view.buffer.storage.runtime, &self.runtime.inner) {
                return Err(Error::Message("buffer belongs to another runtime".into()));
            }
        }
        let node = Node(self.entries.len());
        self.entries.push(Entry { operation, uses });
        Ok(node)
    }
    /// Fill a GPU-visible range with a byte pattern.
    pub fn fill(&mut self, destination: BufferView, value: u8) -> Result<Node> {
        destination.gpu()?;
        self.push(
            Description::Fill(destination.clone(), value),
            vec![Use {
                view: destination,
                access: Access::Write,
            }],
        )
    }
    /// Explicit GPU copy. Lengths must match and ranges must not overlap.
    pub fn copy(&mut self, destination: BufferView, source: BufferView) -> Result<Node> {
        destination.gpu()?;
        source.gpu()?;
        if destination.len() != source.len() || destination.overlaps(&source) {
            return Err(Error::Message(
                "copy needs equal, nonoverlapping ranges".into(),
            ));
        }
        self.push(
            Description::Copy(destination.clone(), source.clone()),
            vec![
                Use {
                    view: destination,
                    access: Access::Write,
                },
                Use {
                    view: source,
                    access: Access::Read,
                },
            ],
        )
    }
    /// Invoke a trusted GPU kernel; binding extents and alignment are checked.
    /// The selected runtime must expose allocation-address queries (interop ABI 1),
    /// including for GPU-local bindings. No base-pointer alignment is assumed.
    pub fn gpu(&mut self, kernel: &GpuKernel, bindings: &[BufferView]) -> Result<Node> {
        if !Arc::ptr_eq(&kernel.runtime, &self.runtime.inner) {
            return Err(Error::Message("kernel belongs to another runtime".into()));
        }
        kernel.contract.check(bindings)?;
        for (binding, contract) in bindings.iter().zip(&kernel.contract.bindings) {
            let address = binding.gpu()?.owner().allocation_address()?;
            if !(address + binding.offset() as u64).is_multiple_of(contract.alignment as u64) {
                return Err(Error::Message(
                    "GPU allocation does not satisfy kernel alignment".into(),
                ));
            }
        }
        let uses = bindings
            .iter()
            .zip(&kernel.contract.bindings)
            .map(|(view, c)| Use {
                view: view.clone(),
                access: c.access,
            })
            .collect();
        self.push(Description::Gpu(kernel.clone(), bindings.to_vec()), uses)
    }
    /// Invoke a trusted NPU specialization with checked device and memory-bank compatibility.
    /// Host-only and imported BOs may originate in a different program context on
    /// the same device and memory bank. Their owning contexts remain retained.
    #[cfg(feature = "npu")]
    pub fn npu(&mut self, kernel: &crate::npu::NpuKernel, bindings: &[BufferView]) -> Result<Node> {
        kernel.contract().check(bindings)?;
        for (index, (binding, group)) in bindings.iter().zip(&kernel.inner.groups).enumerate() {
            let storage = &binding.buffer.storage;
            let address = storage
                .bo
                .as_ref()
                .ok_or_else(|| Error::Unsupported("missing NPU import".into()))?
                .address()
                .map_err(Error::Message)?;
            let contract = &kernel.contract().bindings[index];
            if !(address + binding.offset() as u64).is_multiple_of(contract.alignment as u64) {
                return Err(Error::Message(
                    "NPU allocation does not satisfy kernel alignment".into(),
                ));
            }
            if storage.npu_device != Some(kernel.inner.program.inner.device)
                || storage.group.map(crate::npu::raw::memory_bank)
                    != Some(crate::npu::raw::memory_bank(*group))
            {
                return Err(Error::Unsupported(
                    "NPU binding device or host memory bank differs from the program".into(),
                ));
            }
        }
        let uses = bindings
            .iter()
            .zip(&kernel.contract().bindings)
            .map(|(view, c)| Use {
                view: view.clone(),
                access: c.access,
            })
            .collect();
        self.push(Description::Npu(kernel.clone(), bindings.to_vec()), uses)
    }
    /// Lower GPU regions and import NPU views once; reserve reusable run slots.
    pub fn prepare(self) -> Result<ExecutableGraph> {
        if self.entries.is_empty() {
            return Err(Error::Message("cannot prepare an empty graph".into()));
        }
        let mut operations: Vec<Operation> = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            #[cfg(feature = "npu")]
            if let Description::Npu(kernel, bindings) = &self.entries[index].operation {
                let arguments = bindings
                    .iter()
                    .map(|view| {
                        let bo = view.buffer.storage.bo.as_ref().ok_or_else(|| {
                            Error::Unsupported("NPU binding lacks an import".into())
                        })?;
                        if view.offset() == 0 && view.len() == bo.len() {
                            Ok(bo.clone())
                        } else {
                            bo.sub(view.len(), view.offset()).map_err(Error::Message)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                // HostOnly/imported BOs are device-global. Graph::npu checked
                // their device and argument memory bank; sub-BOs retain that mapping.
                // PreparedRun retains both the dispatch and allocation contexts.
                let run = unsafe {
                    crate::npu::raw::PreparedRun::new(
                        &kernel.inner.program.inner.context,
                        &kernel.inner.instructions,
                        arguments,
                    )
                }
                .map_err(Error::Message)?;
                operations.push(Operation {
                    copy_bytes: 0,
                    backend: Backend::Npu(Mutex::new(run)),
                    uses: self.entries[index].uses.clone(),
                    dependencies: Vec::new(),
                });
                index += 1;
                continue;
            }
            let start = index;
            while index < self.entries.len() {
                #[cfg(feature = "npu")]
                if matches!(self.entries[index].operation, Description::Npu(..)) {
                    break;
                }
                index += 1;
            }
            let stream = crate::gpu::Device::open(self.runtime.inner.gpu_index)?.stream()?;
            let mut graph = stream.graph()?;
            let mut nodes = Vec::new();
            let mut uses = Vec::new();
            let mut frontier = super::dependencies::Frontier::default();
            for (i, entry) in self.entries[start..index].iter().enumerate() {
                let dependencies = frontier
                    .dependencies(i, &entry.uses)
                    .into_iter()
                    .map(|j| nodes[j])
                    .collect::<Vec<_>>();
                let node = match &entry.operation {
                    Description::Fill(dst, value) => {
                        graph.fill(&dependencies, dst.gpu()?, *value)?
                    }
                    Description::Copy(dst, src) => {
                        graph.copy(&dependencies, dst.gpu()?, src.gpu()?)?
                    }
                    Description::Gpu(kernel, bindings) => {
                        let views = bindings
                            .iter()
                            .map(BufferView::gpu)
                            .collect::<Result<Vec<_>>>()?;
                        let constants =
                            crate::gpu::Constants::from_bytes(&kernel.contract.constants)?;
                        unsafe {
                            graph.dispatch(
                                &dependencies,
                                &kernel.raw,
                                kernel.grid,
                                kernel.block,
                                &constants,
                                &views,
                            )
                        }?
                    }
                    #[cfg(feature = "npu")]
                    Description::Npu(..) => unreachable!(),
                };
                nodes.push(node);
                uses.extend(entry.uses.iter().cloned());
            }
            let executable = graph.finish()?;
            operations.push(Operation {
                copy_bytes: self.entries[start..index]
                    .iter()
                    .filter_map(|entry| match &entry.operation {
                        Description::Copy(dst, _) => Some(dst.len()),
                        _ => None,
                    })
                    .sum(),
                backend: Backend::Gpu(Mutex::new((stream, executable))),
                uses: super::dependencies::summarize(uses),
                dependencies: Vec::new(),
            });
        }
        let mut frontier = super::dependencies::Frontier::default();
        for (index, operation) in operations.iter_mut().enumerate() {
            operation.uses = super::dependencies::summarize(operation.uses.drain(..));
            operation.dependencies = frontier.dependencies(index, &operation.uses);
        }
        let uses =
            super::dependencies::summarize(operations.iter().flat_map(|o| o.uses.iter().cloned()));
        let slots = (0..self.runtime.options.graph_slots)
            .map(|_| Slot {
                nodes: vec![NodeState::Pending; operations.len()],
                occupied: false,
                failure: None,
            })
            .collect();
        Ok(ExecutableGraph {
            inner: Arc::new(Prepared {
                signals: (0..self.runtime.options.graph_slots)
                    .map(|_| Signal::new())
                    .collect(),
                runtime: self.runtime.inner,
                parallel: operations
                    .iter()
                    .enumerate()
                    .skip(1)
                    .any(|(index, op)| !op.dependencies.contains(&(index - 1))),
                operations,
                uses,
                slots: Mutex::new(slots),
            }),
        })
    }
}
pub(super) enum Backend {
    Gpu(Mutex<(crate::gpu::Stream, crate::gpu::GraphExec)>),
    #[cfg(feature = "npu")]
    Npu(Mutex<crate::npu::raw::PreparedRun>),
    #[cfg(test)]
    Mock(Engine, Arc<dyn Fn() -> Result<()> + Send + Sync>),
}
pub(super) struct Operation {
    pub copy_bytes: usize,
    pub backend: Backend,
    pub uses: Vec<Use>,
    pub dependencies: Vec<usize>,
}
impl Operation {
    pub fn engine(&self) -> Engine {
        match &self.backend {
            Backend::Gpu(_) => Engine::Gpu,
            #[cfg(feature = "npu")]
            Backend::Npu(_) => Engine::Npu,
            #[cfg(test)]
            Backend::Mock(engine, _) => *engine,
        }
    }
    pub fn run(&self) -> Result<()> {
        for access in &self.uses {
            if let Some(message) = &access
                .view
                .buffer
                .storage
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .poison
            {
                return Err(Error::DeviceLost(message.clone()));
            }
            access.view.buffer.storage.make_visible(self.engine())?;
        }
        match &self.backend {
            Backend::Gpu(pair) => {
                let mut pair = pair.lock().unwrap_or_else(|e| e.into_inner());
                let (stream, graph) = &mut *pair;
                stream.launch(graph)?;
                stream.synchronize()?;
            }
            #[cfg(feature = "npu")]
            Backend::Npu(run) => run.lock().unwrap_or_else(|e| e.into_inner()).execute()?,
            #[cfg(test)]
            Backend::Mock(_, action) => action()?,
        }
        if self.copy_bytes != 0
            && let Some(access) = self.uses.first()
        {
            access
                .view
                .buffer
                .storage
                .runtime
                .core
                .counters
                .copied_bytes
                .fetch_add(self.copy_bytes as u64, std::sync::atomic::Ordering::Relaxed);
        }
        for access in &self.uses {
            if access.access.writes() {
                access
                    .view
                    .buffer
                    .storage
                    .visibility
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .wrote(self.engine());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeState {
    Pending,
    Running,
    Done,
}
pub(super) struct Slot {
    pub nodes: Vec<NodeState>,
    pub occupied: bool,
    pub failure: Option<Arc<Error>>,
}
pub(super) struct Prepared {
    pub runtime: Arc<RuntimeOwner>,
    pub parallel: bool,
    pub operations: Vec<Operation>,
    pub uses: Vec<Use>,
    pub slots: Mutex<Vec<Slot>>,
    pub signals: Vec<Arc<Signal>>,
}
/// A prepared graph with fixed bindings and reusable bounded submission slots.
#[derive(Clone)]
pub struct ExecutableGraph {
    pub(super) inner: Arc<Prepared>,
}
impl ExecutableGraph {
    /// Enqueue without waiting for a device. Conflicting prior work is ordered.
    /// `Busy` indicates live host mappings or exhausted pending/completion slots.
    pub fn submit(&self) -> Result<Completion> {
        let core = &self.inner.runtime.core;
        let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
        if scheduler.pending.len() >= scheduler.capacity {
            return Err(Error::Busy("runtime submission capacity exhausted".into()));
        }
        for access in &self.inner.uses {
            access.view.buffer.storage.host_conflict(access.access)?;
        }
        let order = scheduler.next_order;
        let next_order = order
            .checked_add(1)
            .ok_or_else(|| Error::Message("submission sequence exhausted".into()))?;
        let mut slots = self.inner.slots.lock().unwrap_or_else(|e| e.into_inner());
        let (slot_index, slot) = slots.iter_mut().enumerate().find(|(index, slot)| !slot.occupied && Arc::strong_count(&self.inner.signals[*index]) == 1 && self.inner.signals[*index].state.lock().unwrap_or_else(|e| e.into_inner()).done).ok_or_else(|| Error::Busy("prepared run slots are occupied; drop completed observers or reserve more slots".into()))?;
        self.inner.signals[slot_index].reset();
        slot.nodes.fill(NodeState::Pending);
        slot.failure = None;
        slot.occupied = true;
        let completion = Completion {
            signal: self.inner.signals[slot_index].clone(),
            core: Arc::downgrade(core),
        };
        scheduler.next_order = next_order;
        core.counters
            .submissions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        scheduler.pending.push(super::scheduler::Pending {
            graph: self.inner.clone(),
            slot: slot_index,
            order,
        });
        drop(slots);
        drop(scheduler);
        core.changed.notify_one();
        Ok(completion)
    }
    /// Number of prepared device regions (GPU regions may contain many kernels).
    pub fn regions(&self) -> usize {
        self.inner.operations.len()
    }
}
