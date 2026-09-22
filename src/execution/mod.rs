//! Coordinated execution for GPU and NPU kernels.
//!
//! Device placement is explicit. Contracts describe memory access; prepared
//! graphs infer dependencies and retain all allocations through completion.
//! The low-level GPU API remains available in [`crate::gpu`].
mod buffer;
mod completion;
mod contract;
mod dependencies;
mod graph;
mod handoff;
mod native_session;
mod scheduler;
mod statistics;
mod trace;
use crate::{Error, Result};
pub use buffer::{Buffer, BufferView, MemoryPlacement, ReadGuard, WriteGuard};
use buffer::{Engine, HostState, Storage, Visibility};
pub use completion::{Completion, CompletionProfile};
pub use contract::{Access, BindingContract, KernelContract};
pub use graph::{ExecutableGraph, GpuKernel, GpuLane, Graph, Node};
pub use native_session::NativeSession;
use scheduler::Core;
pub use statistics::Statistics;
use std::sync::{Arc, Mutex};
pub use trace::{ExecutionTrace, TraceEvent};

/// A buffer region and the access performed during an external GPU handoff.
#[derive(Clone)]
pub struct GpuAccess {
    /// Region retained and reserved until the supplied stream completes.
    pub view: BufferView,
    /// Reads and writes performed by the callback.
    pub access: Access,
}

pub(super) struct RuntimeOwner {
    core: Arc<Core>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    gpu_index: i32,
    copy_streams: [Mutex<Option<Arc<Mutex<graph::CopyStream>>>>; 3],
    allocation_stream: Mutex<Option<crate::gpu::Stream>>,
}
impl Drop for RuntimeOwner {
    fn drop(&mut self) {
        self.core
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .shutdown = true;
        self.core.changed.notify_all();
        for worker in self
            .workers
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
    }
}
/// Runtime limits fixed before workers or prepared run slots are created.
#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    /// Optional shared byte ceiling for tracked allocations, including weights,
    /// private scratch and transfer staging. Native allocations made outside
    /// this runtime are not covered; adoption charges them at the boundary.
    pub memory_budget: Option<crate::residency::MemoryBudget>,
    /// Physical GPU index; zero selects the integrated GPU on the tested host.
    pub gpu_index: i32,
    /// Maximum simultaneous submissions. Exhaustion returns `Busy`.
    /// This bounds queued work; it does not increase per-engine concurrency.
    pub max_submissions: usize,
    /// Preallocated completion slots per prepared graph.
    pub graph_slots: usize,
}
impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            memory_budget: None,
            gpu_index: 0,
            max_submissions: 64,
            graph_slots: 2,
        }
    }
}
/// Shared scheduler and allocation domain. Clones use the same dependency state.
///
/// Upload, compute, download, and NPU regions have independent scheduling lanes.
/// Each lane runs one region at a time; conflicting memory accesses are ordered
/// across all lanes and submissions. Actual device overlap depends on hardware.
#[derive(Clone)]
pub struct Runtime {
    pub(super) inner: Arc<RuntimeOwner>,
    options: RuntimeOptions,
}
/// A selected GPU and its verified compilation target.
#[derive(Clone)]
pub struct GpuDevice {
    index: i32,
    target: crate::Target,
}
impl GpuDevice {
    /// Physical GPU ordinal.
    pub fn index(&self) -> i32 {
        self.index
    }
    /// Target for compiling a kernel on this GPU.
    pub fn target(&self) -> &crate::Target {
        &self.target
    }
}
/// An explicitly selected NPU ordinal.
#[cfg(feature = "npu")]
#[derive(Clone)]
pub struct NpuDevice {
    native: crate::fabric::Device,
    index: i32,
    budget: Option<crate::residency::MemoryBudget>,
}
#[cfg(feature = "npu")]
impl NpuDevice {
    /// Physical NPU ordinal.
    pub fn index(&self) -> i32 {
        self.index
    }
    /// Exact XDNA deployment target for offline compilation.
    pub fn target(&self) -> &crate::Target {
        self.native.target()
    }
    /// Native device domain, independent of any loaded program.
    pub fn native(&self) -> &crate::fabric::Device {
        &self.native
    }
    /// Admit a canonical XDNA artifact and its external binding contract.
    ///
    /// # Safety
    /// The native code must be trusted and satisfy the supplied access contract.
    pub unsafe fn load_artifact(
        &self,
        artifact: &crate::loom::Artifact,
        columns: u16,
        contract: KernelContract,
    ) -> Result<crate::npu::NpuKernel> {
        unsafe {
            crate::npu::NpuKernel::load(
                self.native.clone(),
                artifact,
                columns,
                contract,
                self.budget.as_ref(),
            )
        }
    }
}
impl Runtime {
    /// Begin an opt-in bounded trace of host-observed native-region latencies.
    pub fn start_trace(&self, event_capacity: usize) -> Result<()> {
        self.inner.core.tracer.start(event_capacity)
    }
    /// Stop capture without waiting for pending work. Wait for its completions
    /// first when the capture should include the entire request.
    pub fn finish_trace(&self) -> Option<ExecutionTrace> {
        self.inner.core.tracer.finish()
    }
    /// Whether a view belongs to this runtime's allocation and scheduling domain.
    pub fn owns(&self, view: &BufferView) -> bool {
        Arc::ptr_eq(&self.inner, &view.buffer.storage.runtime)
    }

    /// Whether two handles share the same allocation and scheduling domain.
    pub fn same_domain(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Shared allocation ceiling, if selected at construction. Native clients
    /// can attach it to their streams before loading weights or workspace.
    pub fn memory_budget(&self) -> Option<&crate::residency::MemoryBudget> {
        self.options.memory_budget.as_ref()
    }

    /// Create a runtime. Native devices are opened lazily when requested.
    pub fn new() -> Result<Self> {
        Self::with_options(RuntimeOptions::default())
    }
    /// Create a runtime with bounded pending submission and graph capacities.
    pub fn with_options(options: RuntimeOptions) -> Result<Self> {
        if options.max_submissions == 0 || options.graph_slots == 0 || options.gpu_index < 0 {
            return Err(Error::Message(
                "runtime capacities must be nonzero and GPU index nonnegative".into(),
            ));
        }
        let core = Arc::new(Core::new(options.max_submissions));
        let owner = Arc::new(RuntimeOwner {
            core: core.clone(),
            workers: Mutex::new(Vec::new()),
            gpu_index: options.gpu_index,
            copy_streams: std::array::from_fn(|_| Mutex::new(None)),
            allocation_stream: Mutex::new(None),
        });
        for name in ["hrx-upload", "hrx-compute", "hrx-download", "hrx-npu"] {
            let core = core.clone();
            let worker = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || scheduler::worker(core))?;
            owner
                .workers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(worker);
        }
        Ok(Self {
            inner: owner,
            options,
        })
    }
    /// Open the selected GPU and return its actual target.
    pub fn gpu(&self) -> Result<GpuDevice> {
        let device = crate::gpu::Device::open(self.inner.gpu_index)?;
        Ok(GpuDevice {
            index: self.inner.gpu_index,
            target: device.target().clone(),
        })
    }
    /// Select an NPU. Opening its program validates actual device access.
    #[cfg(feature = "npu")]
    pub fn npu(&self, index: i32) -> Result<NpuDevice> {
        if index < 0 {
            return Err(Error::Message("NPU index must be nonnegative".into()));
        }
        Ok(NpuDevice {
            native: crate::fabric::Device::open(crate::fabric::Engine::Xdna, index as usize)?,
            index,
            budget: self.options.memory_budget.clone(),
        })
    }
    // Buffers retain their allocating stream. Creating one per allocation
    // therefore consumes one native queue per live tensor, even after the
    // temporary Stream handle drops. Serialize allocation/initialization on a
    // single lazy stream; execution graphs retain their independent queues.
    fn allocation_stream(&self) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::Stream>>> {
        let mut stream = self
            .inner
            .allocation_stream
            .lock()
            .map_err(|_| Error::DeviceLost("allocation stream poisoned".into()))?;
        if stream.is_none() {
            *stream = Some(crate::gpu::Device::open(self.inner.gpu_index)?.stream()?);
        }
        Ok(stream)
    }

    /// Allocate initialized storage. Shared memory is exported and imported once.
    pub fn allocate(&self, bytes: usize, placement: MemoryPlacement) -> Result<Buffer> {
        self.allocate_inner(bytes, placement, None)
    }

    /// Transfer an initialized allocation into this runtime, optionally importing
    /// it into an NPU program once. Only GpuLocal and Shared placement are accepted.
    ///
    /// # Safety
    /// All bytes must be initialized and prior uses complete. No old pointer,
    /// recorded graph or other external alias may access the allocation after
    /// adoption, except through this runtime's scoped GPU handoff.
    pub unsafe fn adopt_gpu_buffer(
        &self,
        buffer: crate::gpu::Buffer,
        placement: MemoryPlacement,
    ) -> Result<Buffer> {
        let mut allocation = self.allocation_stream()?;
        let stream = allocation.as_mut().unwrap();
        if buffer.device_id() != stream.device_id() {
            return Err(Error::Message("buffer belongs to another device".into()));
        }
        if matches!(placement, MemoryPlacement::HostVisible) || matches_npu_local(&placement) {
            return Err(Error::Unsupported(
                "adoption requires GpuLocal or Shared placement".into(),
            ));
        }
        drop(allocation);
        self.allocate_inner(buffer.bytes(), placement, Some(buffer))
    }

    fn allocate_inner(
        &self,
        bytes: usize,
        placement: MemoryPlacement,
        mut adopted: Option<crate::gpu::Buffer>,
    ) -> Result<Buffer> {
        if bytes == 0 || bytes > isize::MAX as usize {
            return Err(Error::Message(
                "allocation size must be in 1..=isize::MAX".into(),
            ));
        }
        let mut storage = Storage {
            _reservation: self
                .options
                .memory_budget
                .as_ref()
                .filter(|budget| {
                    !adopted
                        .as_ref()
                        .is_some_and(|buffer| buffer.charged_to(budget))
                })
                .map(|budget| budget.reserve(bytes))
                .transpose()?,
            native: None,
            accounted: false,
            gpu: None,
            pointer: std::ptr::null_mut(),
            bytes,
            #[cfg(test)]
            _test_memory: None,
            shared: false,
            host: Mutex::new(HostState::default()),
            visibility: Mutex::new(Visibility::new()),
            runtime: self.inner.clone(),
        };
        match &placement {
            MemoryPlacement::GpuLocal => {
                let mut allocation = self.allocation_stream()?;
                let stream = allocation.as_mut().unwrap();
                let buffer = if let Some(buffer) = adopted.take() {
                    buffer
                } else {
                    let buffer = stream.allocate(bytes)?;
                    stream.fill(buffer.binding(), 0)?;
                    stream.synchronize()?;
                    buffer
                };
                storage.native = Some(buffer.native.clone());
                storage.gpu = Some(buffer);
                storage.visibility.get_mut().unwrap().wrote(Engine::Gpu);
            }
            MemoryPlacement::HostVisible => {
                let mut allocation = self.allocation_stream()?;
                let stream = allocation.as_mut().unwrap();
                let buffer = stream.allocate_shared(bytes)?;
                let pointer = buffer.device_ptr()?.cast::<u8>();
                // Coherent host-local allocation, with no aliases or device uses.
                unsafe {
                    std::ptr::write_bytes(pointer, 0, bytes);
                }
                storage.pointer = pointer;
                storage.native = Some(buffer.native.clone());
                storage.gpu = Some(buffer);
            }
            #[cfg(feature = "npu")]
            MemoryPlacement::Shared(device) | MemoryPlacement::NpuLocal(device) => {
                let native = if matches!(placement, MemoryPlacement::Shared(_)) {
                    let mut allocation = self.allocation_stream()?;
                    let stream = allocation.as_mut().unwrap();
                    let gpu = if let Some(buffer) = adopted.take() {
                        // Register the same owned backing; its original native owner
                        // remains retained through the shared attachment's lifetime.
                        buffer.share_with(device.native())?
                    } else {
                        stream.allocate_for(
                            bytes,
                            &[
                                crate::gpu::Device::open(self.inner.gpu_index)?
                                    .native()
                                    .clone(),
                                device.native().clone(),
                            ],
                        )?
                    };
                    let native = gpu.native.clone();
                    storage.gpu = Some(gpu);
                    storage.shared = true;
                    native
                } else {
                    device
                        .native()
                        .fabric()
                        .allocate(bytes, std::slice::from_ref(device.native()))?
                };
                storage.pointer = native.host_pointer();
                storage.native = Some(native);
            }
        }

        use std::sync::atomic::Ordering;
        let counters = &self.inner.core.counters;
        counters.allocations.fetch_add(1, Ordering::Relaxed);
        if storage.shared {
            counters.imports.fetch_add(1, Ordering::Relaxed);
        }
        let live = counters
            .live_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed)
            + bytes as u64;
        counters.peak_bytes.fetch_max(live, Ordering::Relaxed);
        storage.accounted = true;
        Ok(Buffer {
            storage: Arc::new(storage),
        })
    }
    /// Observe allocation, transfer, and execution counters.
    pub fn statistics(&self) -> Statistics {
        self.inner.core.counters.snapshot()
    }

    /// Start describing a reusable pipeline in this allocation domain.
    pub fn graph(&self) -> Graph {
        Graph::new(self.clone())
    }
    /// Load a fixed GPU specialization with its trusted memory contract.
    /// # Safety
    /// The code must obey the contract for this grid/block and every accepted
    /// binding. Kernel memory access is not sandboxed by native binding lengths.
    pub unsafe fn load_gpu_kernel(
        &self,
        path: impl AsRef<std::path::Path>,
        symbol: &str,
        grid: [u32; 3],
        block: [u32; 3],
        contract: KernelContract,
    ) -> Result<GpuKernel> {
        let mut allocation = self.allocation_stream()?;
        let stream = allocation.as_mut().unwrap();
        let raw = unsafe { stream.load(path.as_ref(), symbol) }?;
        drop(allocation);
        unsafe { self.adopt_gpu_kernel(raw, grid, block, contract) }
    }

    /// Retain a loaded executable without loading or copying its code again.
    /// # Safety
    /// The kernel must obey the declared memory contract for this launch and
    /// every accepted binding, as for [`Self::load_gpu_kernel`].
    pub unsafe fn adopt_gpu_kernel(
        &self,
        raw: crate::gpu::Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        contract: KernelContract,
    ) -> Result<GpuKernel> {
        contract.validate()?;
        crate::runtime::validate_export_launch(raw.info(), grid, block)?;
        let mut allocation = self.allocation_stream()?;
        let stream = allocation.as_mut().unwrap();
        if raw.device_id() != stream.device_id() {
            return Err(Error::Message("kernel belongs to another device".into()));
        }
        if raw.info().binding_count as usize != contract.bindings.len()
            || raw.info().constant_byte_length as usize != contract.constants.len()
        {
            return Err(Error::Message(
                "GPU metadata does not match contract".into(),
            ));
        }
        Ok(GpuKernel {
            raw,
            contract,
            grid,
            block,
            runtime: self.inner.clone(),
        })
    }
}

fn matches_npu_local(placement: &MemoryPlacement) -> bool {
    #[cfg(feature = "npu")]
    {
        matches!(placement, MemoryPlacement::NpuLocal(_))
    }
    #[cfg(not(feature = "npu"))]
    {
        let _ = placement;
        false
    }
}

#[cfg(test)]
mod tests;
