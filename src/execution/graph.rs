use super::{
    Access, BufferView, Completion, Engine, KernelContract, Runtime, RuntimeOwner,
    completion::Signal,
};
use crate::{Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Independent GPU scheduling lanes. Memory hazards apply across every lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum GpuLane {
    /// Host-to-device transfers.
    Upload,
    /// Kernels, fills, and device-to-device copies.
    #[default]
    Compute,
    /// Device-to-host transfers.
    Download,
}

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
    Scoped(Arc<Mutex<ScopedGpu>>, Vec<BufferView>),
    #[cfg(feature = "npu")]
    Npu(crate::npu::NpuKernel, Vec<BufferView>),
}
type ScopedGpu = Box<dyn FnMut(&[crate::gpu::View<'_>]) -> Result<()> + Send>;
struct Entry {
    operation: Description,
    uses: Vec<Use>,
    lane: GpuLane,
}
/// A pipeline builder. Each operation declares placement and actual buffer use.
pub struct Graph {
    runtime: Runtime,
    entries: Vec<Entry>,
    // Private fragment temporaries only. No exposed IO or model weights enter
    // this pool; access hazards order a later fragment's reuse after all readers.
    fragment_scratch: Vec<(bool, BufferView)>,
}
impl Graph {
    pub(crate) fn validate_runtime(&self, runtime: &Runtime) -> Result<()> {
        if !self.runtime.same_domain(runtime) {
            return Err(Error::Message("graph belongs to another runtime".into()));
        }
        Ok(())
    }
    /// Integrate owned native GPU execution into this graph's hazards and
    /// producer dependencies. The callback runs on a worker, never in `poll`.
    ///
    /// # Safety
    /// The callback must retain all private native resources it uses, access
    /// these views only as declared, and synchronize every stream touching them
    /// before returning. It must not let raw pointers/views escape this scope.
    /// Private storage must not be concurrently accessible outside the callback.
    /// On an uncertain failure return an error; captured resources and bindings
    /// are then quarantined together, preserving native lifetimes.
    pub unsafe fn gpu_scoped(
        &mut self,
        bindings: &[super::GpuAccess],
        callback: impl FnMut(&[crate::gpu::View<'_>]) -> Result<()> + Send + 'static,
    ) -> Result<Node> {
        for binding in bindings {
            binding.view.gpu()?;
        }
        self.push(
            Description::Scoped(
                Arc::new(Mutex::new(Box::new(callback))),
                bindings.iter().map(|b| b.view.clone()).collect(),
            ),
            bindings
                .iter()
                .map(|b| Use {
                    view: b.view.clone(),
                    access: b.access,
                })
                .collect(),
        )
    }
    pub(super) fn new(runtime: Runtime) -> Self {
        Self {
            runtime,
            entries: Vec::new(),
            fragment_scratch: Vec::new(),
        }
    }

    pub(crate) fn fragment_scratch(
        &mut self,
        requests: &[(usize, bool)],
    ) -> Result<Vec<BufferView>> {
        let mut used = Vec::with_capacity(requests.len());
        let mut result = Vec::with_capacity(requests.len());
        for &(bytes, host_visible) in requests {
            let best = self
                .fragment_scratch
                .iter()
                .enumerate()
                .filter(|(index, (host, view))| {
                    !used.contains(index) && *host == host_visible && view.len() >= bytes
                })
                .min_by_key(|(_, (_, view))| view.len())
                .map(|(index, _)| index);
            let index = if let Some(index) = best {
                index
            } else {
                let placement = if host_visible {
                    super::MemoryPlacement::HostVisible
                } else {
                    super::MemoryPlacement::GpuLocal
                };
                let view = self.runtime.allocate(bytes, placement)?.view();
                self.fragment_scratch.push((host_visible, view));
                self.fragment_scratch.len() - 1
            };
            used.push(index);
            result.push(self.fragment_scratch[index].1.slice(0..bytes)?);
        }
        Ok(result)
    }
    fn push(&mut self, operation: Description, uses: Vec<Use>) -> Result<Node> {
        for binding in &uses {
            if !Arc::ptr_eq(&binding.view.buffer.storage.runtime, &self.runtime.inner) {
                return Err(Error::Message("buffer belongs to another runtime".into()));
            }
        }
        let node = Node(self.entries.len());
        self.entries.push(Entry {
            operation,
            uses,
            lane: GpuLane::Compute,
        });
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
        let lane = match (
            source.buffer.storage.pointer.is_null(),
            destination.buffer.storage.pointer.is_null(),
        ) {
            (false, true) => GpuLane::Upload,
            (true, false) => GpuLane::Download,
            _ => GpuLane::Compute,
        };
        self.copy_on(destination, source, lane)
    }

    /// Copy on an explicit lane, retaining the same cross-lane hazard checks.
    pub fn copy_on(
        &mut self,
        destination: BufferView,
        source: BufferView,
        lane: GpuLane,
    ) -> Result<Node> {
        destination.gpu()?;
        source.gpu()?;
        if destination.len() != source.len() || destination.overlaps(&source) {
            return Err(Error::Message(
                "copy needs equal, nonoverlapping ranges".into(),
            ));
        }
        let node = self.push(
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
        )?;
        self.entries[node.0].lane = lane;
        Ok(node)
    }
    /// Invoke a trusted GPU kernel; binding extents and alignment are checked.
    /// The selected runtime must expose allocation-address queries (interop ABI 1),
    /// including for GPU-local bindings. No base-pointer alignment is assumed.
    pub fn gpu(&mut self, kernel: &GpuKernel, bindings: &[BufferView]) -> Result<Node> {
        kernel.contract.check(bindings)?;
        // Checked above: no writable aliasing is possible.
        unsafe { self.gpu_aliasing(kernel, bindings) }
    }

    /// Invoke a trusted GPU kernel permitting writable argument aliases.
    /// Extents, alignment, runtime identity and cross-operation hazards remain checked.
    ///
    /// # Safety
    /// The kernel must be valid for the overlapping argument regions supplied;
    /// no inter-thread data races or unsupported in-place accesses may occur.
    pub unsafe fn gpu_aliasing(
        &mut self,
        kernel: &GpuKernel,
        bindings: &[BufferView],
    ) -> Result<Node> {
        if !Arc::ptr_eq(&kernel.runtime, &self.runtime.inner) {
            return Err(Error::Message("kernel belongs to another runtime".into()));
        }
        kernel.contract.check_extents(bindings)?;
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
        for (binding, contract) in bindings.iter().zip(&kernel.contract().bindings) {
            let native = binding
                .buffer
                .storage
                .native
                .as_ref()
                .ok_or_else(|| Error::Unsupported("binding has no native storage".into()))?;
            let address = native.device_address(&kernel.inner.device)?;
            if !(address + binding.offset() as u64).is_multiple_of(contract.alignment as u64) {
                return Err(Error::Message(
                    "NPU allocation does not satisfy kernel alignment".into(),
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
            if let Description::Scoped(callback, bindings) = &self.entries[index].operation {
                operations.push(Operation {
                    lane: GpuLane::Compute,
                    copy_bytes: 0,
                    backend: Backend::Scoped(callback.clone(), bindings.clone()),
                    uses: self.entries[index].uses.clone(),
                    dependencies: Vec::new(),
                });
                index += 1;
                continue;
            }
            #[cfg(feature = "npu")]
            if let Description::Npu(kernel, bindings) = &self.entries[index].operation {
                let arguments = bindings
                    .iter()
                    .map(|view| -> Result<_> {
                        Ok(crate::fabric::XdnaBinding {
                            buffer: view.buffer.storage.native.as_ref().ok_or_else(|| {
                                Error::Unsupported("missing native XDNA backing".into())
                            })?,
                            offset: view.offset(),
                            length: view.len(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                // Native preparation validates the image's own range/usage contracts.
                // Every invocation submits the complete establishing command.
                let run = unsafe {
                    kernel.inner.device.prepare_xdna(
                        &kernel.inner.artifact,
                        kernel.inner.columns,
                        &arguments,
                    )
                }?;
                operations.push(Operation {
                    lane: GpuLane::Compute,
                    copy_bytes: 0,
                    backend: Backend::Npu(Mutex::new(run), kernel.clone()),
                    uses: self.entries[index].uses.clone(),
                    dependencies: Vec::new(),
                });
                index += 1;
                continue;
            }
            let start = index;
            let lane = self.entries[start].lane;
            while index < self.entries.len() {
                if matches!(self.entries[index].operation, Description::Scoped(..)) {
                    break;
                }
                if self.entries[index].lane != lane {
                    break;
                }
                #[cfg(feature = "npu")]
                if matches!(self.entries[index].operation, Description::Npu(..)) {
                    break;
                }
                index += 1;
            }
            // Dynamic tensor-to-slot copies need hazard scheduling and retained
            // owners, but no native executable graph or per-request stream.
            if self.entries[start..index]
                .iter()
                .all(|e| matches!(e.operation, Description::Copy(..)))
            {
                let stream =
                    crate::cached_init(&self.runtime.inner.copy_streams[lane as usize], || {
                        let stream =
                            crate::gpu::Device::open(self.runtime.inner.gpu_index)?.stream()?;
                        self.runtime
                            .inner
                            .core
                            .counters
                            .copy_streams_created
                            .fetch_add(1, Ordering::Relaxed);
                        Ok(Arc::new(Mutex::new(CopyStream {
                            stream,
                            healthy: true,
                        })))
                    })?;
                let copies: Vec<_> = self.entries[start..index]
                    .iter()
                    .map(|entry| {
                        let Description::Copy(dst, src) = &entry.operation else {
                            unreachable!()
                        };
                        (dst.clone(), src.clone())
                    })
                    .collect();
                operations.push(Operation {
                    lane,
                    copy_bytes: copies.iter().map(|(dst, _)| dst.len()).sum(),
                    backend: Backend::Copies(stream, copies),
                    uses: self.entries[start..index]
                        .iter()
                        .flat_map(|e| e.uses.iter().cloned())
                        .collect(),
                    dependencies: Vec::new(),
                });
                continue;
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
                    Description::Scoped(..) => unreachable!(),
                    #[cfg(feature = "npu")]
                    Description::Npu(..) => unreachable!(),
                };
                nodes.push(node);
                uses.extend(entry.uses.iter().cloned());
            }
            let executable = graph.finish()?;
            self.runtime
                .inner
                .core
                .counters
                .native_graphs_prepared
                .fetch_add(1, Ordering::Relaxed);
            operations.push(Operation {
                lane,
                copy_bytes: self.entries[start..index]
                    .iter()
                    .filter_map(|entry| match &entry.operation {
                        Description::Copy(dst, _) => Some(dst.len()),
                        _ => None,
                    })
                    .sum(),
                backend: Backend::Gpu(Box::new(Mutex::new((stream, executable)))),
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
                quarantined: AtomicBool::new(false),
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
pub(super) struct CopyStream {
    stream: crate::gpu::Stream,
    healthy: bool,
}
pub(super) enum Backend {
    Copies(Arc<Mutex<CopyStream>>, Vec<(BufferView, BufferView)>),
    Gpu(Box<Mutex<(crate::gpu::Stream, crate::gpu::GraphExec)>>),
    Scoped(Arc<Mutex<ScopedGpu>>, Vec<BufferView>),
    #[cfg(feature = "npu")]
    Npu(Mutex<crate::fabric::XdnaProgram>, crate::npu::NpuKernel),
    #[cfg(test)]
    Mock(Engine, Arc<dyn Fn() -> Result<()> + Send + Sync>),
}
pub(super) struct Operation {
    pub lane: GpuLane,
    pub copy_bytes: usize,
    pub backend: Backend,
    pub uses: Vec<Use>,
    pub dependencies: Vec<usize>,
}
impl Operation {
    pub fn queue(&self) -> usize {
        match self.engine() {
            Engine::Gpu => self.lane as usize,
            Engine::Npu => 3,
            Engine::Host => 4,
        }
    }
    pub fn engine(&self) -> Engine {
        match &self.backend {
            Backend::Gpu(_) | Backend::Scoped(..) | Backend::Copies(..) => Engine::Gpu,
            #[cfg(feature = "npu")]
            Backend::Npu(..) => Engine::Npu,
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
            Backend::Copies(stream, copies) => {
                let mut stream = stream.lock().unwrap_or_else(|e| e.into_inner());
                if !stream.healthy {
                    return Err(Error::DeviceLost("copy stream is quarantined".into()));
                }
                let result = (|| {
                    for (dst, src) in copies {
                        stream.stream.copy(dst.gpu()?, src.gpu()?)?;
                    }
                    stream.stream.synchronize()
                })();
                if result.is_err() {
                    stream.healthy = false;
                }
                result?;
            }
            Backend::Gpu(pair) => {
                let mut pair = pair.lock().unwrap_or_else(|e| e.into_inner());
                let (stream, graph) = &mut *pair;
                stream.launch(graph)?;
                stream.synchronize()?;
            }
            Backend::Scoped(callback, bindings) => {
                let views = bindings
                    .iter()
                    .map(BufferView::gpu)
                    .collect::<Result<Vec<_>>>()?;
                callback.lock().unwrap_or_else(|e| e.into_inner())(&views)?;
            }
            #[cfg(feature = "npu")]
            Backend::Npu(run, _kernel) => {
                unsafe { run.lock().unwrap_or_else(|e| e.into_inner()).dispatch() }?.wait()?
            }
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
                .lane_copy_bytes[self.lane as usize]
                .fetch_add(self.copy_bytes as u64, std::sync::atomic::Ordering::Relaxed);
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
    pub quarantined: AtomicBool,
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
    pub(crate) fn validate_runtime(&self, runtime: &Runtime) -> Result<()> {
        if !Arc::ptr_eq(&self.inner.runtime, &runtime.inner) {
            return Err(Error::Message("graph belongs to another runtime".into()));
        }
        Ok(())
    }

    pub(crate) fn usable(&self) -> Result<()> {
        if self.inner.quarantined.load(Ordering::Acquire) {
            return Err(Error::DeviceLost(
                "graph quarantined after native failure".into(),
            ));
        }
        for usage in &self.inner.uses {
            if let Some(message) = &usage
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
        }
        Ok(())
    }
    /// Enqueue without waiting for a device. Conflicting prior work is ordered.
    /// `Busy` indicates live host mappings or exhausted pending/completion slots.
    pub fn submit(&self) -> Result<Completion> {
        self.submit_after(&[])
    }

    /// Submit after successful producers in this runtime. Failed or cancelled
    /// producers prevent execution, even when no memory range overlaps.
    pub fn submit_after(&self, dependencies: &[Completion]) -> Result<Completion> {
        self.usable()?;
        let core = &self.inner.runtime.core;
        for dependency in dependencies {
            if let Some(owner) = dependency.core.upgrade() {
                if !Arc::ptr_eq(core, &owner) {
                    return Err(Error::Message(
                        "completion belongs to another runtime".into(),
                    ));
                }
            } else if !dependency.is_complete() {
                return Err(Error::Message("orphaned incomplete producer".into()));
            }
        }
        let mut scheduler = core.state.lock().unwrap_or_else(|e| e.into_inner());
        if scheduler.occupied() >= scheduler.capacity {
            return Err(Error::Busy("runtime submission capacity exhausted".into()));
        }
        for access in &self.inner.uses {
            access.view.buffer.storage.host_conflict(access.access)?;
        }
        let mut slots = self.inner.slots.lock().unwrap_or_else(|e| e.into_inner());
        let (slot_index, slot) = slots
            .iter_mut()
            .enumerate()
            .find(|(index, slot)| {
                let signal = &self.inner.signals[*index];
                !slot.occupied
                    && signal.observers.load(Ordering::Acquire) == 0
                    && signal.state.lock().unwrap_or_else(|e| e.into_inner()).done
            })
            .ok_or_else(|| {
                Error::Busy("prepared run slots are occupied; drop completed observers or reserve more slots".into())
            })?;
        self.inner.signals[slot_index].reset();
        slot.nodes.fill(NodeState::Pending);
        slot.failure = None;
        slot.occupied = true;
        let completion =
            Completion::new(self.inner.signals[slot_index].clone(), Arc::downgrade(core));
        core.counters
            .submissions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        scheduler.enqueue(self.inner.clone(), slot_index, dependencies.to_vec());
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
