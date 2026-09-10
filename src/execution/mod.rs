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
mod scheduler;
mod statistics;
use crate::{Error, Result};
pub use buffer::{Buffer, BufferView, MemoryPlacement, ReadGuard, WriteGuard};
use buffer::{Engine, HostState, Storage, Visibility};
pub use completion::Completion;
pub use contract::{Access, BindingContract, KernelContract};
pub use graph::{ExecutableGraph, GpuKernel, Graph, Node};
use scheduler::Core;
pub use statistics::Statistics;
use std::sync::{Arc, Mutex};

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
    /// Physical GPU index; zero selects the integrated GPU on the tested host.
    pub gpu_index: i32,
    /// Maximum simultaneous submissions. Exhaustion returns `Busy`.
    pub max_submissions: usize,
    /// Preallocated completion slots per prepared graph.
    pub graph_slots: usize,
}
impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            gpu_index: 0,
            max_submissions: 64,
            graph_slots: 2,
        }
    }
}
/// Shared scheduler and allocation domain. Clones use the same dependency state.
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
#[derive(Clone, Copy)]
pub struct NpuDevice {
    index: i32,
}
#[cfg(feature = "npu")]
impl NpuDevice {
    /// Physical NPU ordinal.
    pub fn index(&self) -> i32 {
        self.index
    }
    /// Load trusted code on this device.
    /// # Safety
    /// The image must satisfy [`crate::npu::NpuProgram::load`]'s contract.
    pub unsafe fn load_program(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<crate::npu::NpuProgram> {
        unsafe { crate::npu::NpuProgram::load(self.index, path) }
    }
}
impl Runtime {
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
        });
        for (name, engine) in [("hrx-gpu", Engine::Gpu), ("hrx-npu", Engine::Npu)] {
            let core = core.clone();
            let worker = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || scheduler::worker(core, engine))?;
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
        Ok(NpuDevice { index })
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
        let stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
        if buffer.device_id() != stream.device_id() {
            return Err(Error::Message("buffer belongs to another device".into()));
        }
        if matches!(placement, MemoryPlacement::HostVisible) || matches_npu_local(&placement) {
            return Err(Error::Unsupported(
                "adoption requires GpuLocal or Shared placement".into(),
            ));
        }
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
            #[cfg(feature = "npu")]
            bo: None,
            #[cfg(feature = "npu")]
            descriptor: None,
            accounted: false,
            gpu: None,
            pointer: std::ptr::null_mut(),
            bytes,
            #[cfg(feature = "npu")]
            npu_device: None,
            #[cfg(feature = "npu")]
            group: None,
            #[cfg(test)]
            _test_memory: None,
            shared: false,
            host: Mutex::new(HostState::default()),
            visibility: Mutex::new(Visibility::new()),
            runtime: self.inner.clone(),
        };
        match &placement {
            MemoryPlacement::GpuLocal => {
                let mut stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
                let buffer = if let Some(buffer) = adopted.take() {
                    buffer
                } else {
                    let buffer = stream.allocate(bytes)?;
                    stream.fill(buffer.binding(), 0)?;
                    stream.synchronize()?;
                    buffer
                };
                storage.gpu = Some(buffer);
                storage.visibility.get_mut().unwrap().wrote(Engine::Gpu);
            }
            MemoryPlacement::HostVisible => {
                let stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
                let buffer = stream.allocate_shared(bytes)?;
                let pointer = buffer.device_ptr()?.cast::<u8>();
                // Coherent host-local allocation, with no aliases or device uses.
                unsafe {
                    std::ptr::write_bytes(pointer, 0, bytes);
                }
                storage.pointer = pointer;
                storage.gpu = Some(buffer);
            }
            #[cfg(feature = "npu")]
            MemoryPlacement::Shared(program) | MemoryPlacement::NpuLocal(program) => {
                let context = &program.inner.context;
                let group = context.group_id(3).map_err(Error::Message)?;
                let bo = if matches!(placement, MemoryPlacement::Shared(_)) {
                    use std::os::fd::AsRawFd;
                    let mut stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
                    let gpu = if let Some(buffer) = adopted.take() {
                        buffer
                    } else {
                        let buffer = stream.allocate(bytes)?;
                        stream.fill(buffer.binding(), 0)?;
                        stream.synchronize()?;
                        buffer
                    };
                    let (fd, offset) = gpu.export_dmabuf()?;
                    let offset = usize::try_from(offset).map_err(|_| {
                        Error::Unsupported("dma-buf offset exceeds address space".into())
                    })?;
                    // Only the owned subregion is initialized; an HSA pool may
                    // export a larger root containing unrelated allocations.
                    let bo = unsafe { context.import_dmabuf_region(fd.as_raw_fd(), offset, bytes) }
                        .map_err(Error::Message)?;
                    storage.gpu = Some(gpu);
                    storage.descriptor = Some(fd);
                    storage.shared = true;
                    bo
                } else {
                    context
                        .alloc_bo(bytes, crate::npu::raw::BoKind::HostOnly, group)
                        .map_err(Error::Message)?
                };
                let pointer = bo.map().map_err(Error::Message)?;
                // No aliases or device submissions exist during initialization.
                // Shared GPU allocations are already initialized, including
                // adopted data. Only freshly allocated NPU-local bytes need zeroing.
                if !storage.shared {
                    unsafe {
                        std::ptr::write_bytes(pointer, 0, bytes);
                    }
                } else {
                    storage.visibility.get_mut().unwrap().wrote(Engine::Gpu);
                }
                storage.pointer = pointer;
                storage.bo = Some(bo);
                storage.npu_device = Some(program.inner.device);
                storage.group = Some(group);
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
        let stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
        let raw = unsafe { stream.load(path.as_ref(), symbol) }?;
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
        let stream = crate::gpu::Device::open(self.inner.gpu_index)?.stream()?;
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
