//! GPU conveniences implemented over the owned native fabric.
use crate::{Error, Result, Target, fabric};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};

/// Native entry's checked argument and workgroup layout.
#[derive(Clone, Debug, Default)]
pub struct ExportInfo {
    /// Scalar bytes packed in declaration order, excluding alignment padding.
    pub constant_byte_length: u32,
    /// Number of explicit buffer arguments.
    pub binding_count: u32,
    /// Number of explicit native arguments.
    pub parameter_count: u32,
    /// Required workgroup dimensions; zero means unconstrained.
    pub workgroup_size: [u32; 3],
}
struct Inner {
    device: fabric::Device,
    queue: fabric::Queue,
    last: Mutex<Option<fabric::Completion>>,
    free: Mutex<Vec<fabric::Buffer>>,
    dispatch_cache: Mutex<Vec<CachedDispatch>>,
    transfer_cache: Mutex<Vec<CachedTransfer>>,
}
impl Inner {
    fn submit(&self, command: &fabric::PreparedGpu) -> Result<()> {
        let done = match unsafe { command.dispatch() } {
            Ok(done) => done,
            Err(Error::Busy(_)) => {
                self.wait()?;
                unsafe { command.dispatch() }?
            }
            Err(error) => return Err(error),
        };
        *self
            .last
            .lock()
            .map_err(|_| Error::DeviceLost("stream timeline poisoned".into()))? = Some(done);
        Ok(())
    }
    fn drain_for_drop(&self) -> bool {
        self.last.lock().ok().is_some_and(|last| {
            last.as_ref().is_none_or(|done| {
                done.wait_timeout(std::time::Duration::from_secs(10))
                    .unwrap_or(false)
            })
        })
    }
    fn wait(&self) -> Result<()> {
        if let Some(done) = self
            .last
            .lock()
            .map_err(|_| Error::DeviceLost("stream timeline poisoned".into()))?
            .as_ref()
        {
            done.wait()?;
        }
        Ok(())
    }
}
/// An owned GPU address and execution domain.
#[derive(Clone)]
pub struct Device {
    native: fabric::Device,
}
impl Device {
    /// Activate an exact GPU ordinal in the native provider.
    pub fn open(index: i32) -> Result<Self> {
        let index = usize::try_from(index)
            .map_err(|_| Error::Message("GPU index must be nonnegative".into()))?;
        Ok(Self {
            native: fabric::Device::open(fabric::Engine::Gpu, index)?,
        })
    }
    /// Require the workload's exact target before creating execution resources.
    pub fn open_for(index: i32, target: &str) -> Result<Self> {
        let device = Self::open(index)?;
        if device.target().as_str() != target {
            return Err(Error::Unsupported(format!(
                "workload requires {target}, found {}",
                device.target().as_str()
            )));
        }
        Ok(device)
    }
    /// Exact compiler deployment key.
    pub fn target(&self) -> &Target {
        self.native.target()
    }
    /// Create an independent ordered native command queue.
    pub fn stream(&self) -> Result<Stream> {
        Ok(Stream {
            inner: Arc::new(Inner {
                device: self.native.clone(),
                queue: self.native.queue()?,
                last: Mutex::new(None),
                free: Mutex::new(Vec::with_capacity(16)),
                dispatch_cache: Mutex::new(Vec::with_capacity(64)),
                transfer_cache: Mutex::new(Vec::with_capacity(128)),
            }),
            staging: Vec::new(),
            staging_pool: Vec::new(),
            scratch: BTreeMap::new(),
            scratch_bytes: 0,
            scratch_limit: 256 * 1024 * 1024,
            budget: None,
            budget_uses: RefCell::new(BudgetUses::default()),
        })
    }
    /// Access the owned native domain for explicit cross-engine allocations.
    pub fn native(&self) -> &fabric::Device {
        &self.native
    }
}
/// Device backing with allocation-budget ownership.
pub struct Buffer {
    pub(crate) native: fabric::Buffer,
    bytes: usize,
    owner: Arc<Inner>,
    reservation: Option<Arc<crate::residency::MemoryReservation>>,
    poolable: bool,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // Cached commands are only useful while the application owns their
        // bindings. Eviction also keeps pooled or budgeted backing reclaimable.
        if let Ok(mut cache) = self.owner.dispatch_cache.lock() {
            cache.retain(|entry| {
                !entry
                    .bindings
                    .iter()
                    .any(|(buffer, _, _)| buffer.same_backing(&self.native))
            });
        }
        if let Ok(mut cache) = self.owner.transfer_cache.lock() {
            cache.retain(|entry| {
                !entry.destination.same_backing(&self.native)
                    && !entry
                        .source
                        .as_ref()
                        .is_some_and(|(buffer, _)| buffer.same_backing(&self.native))
            });
        }
        if self.poolable
            && self.reservation.is_none()
            && self.native.exclusively_owned()
            && self.bytes <= 64 * 1024 * 1024
            && let Ok(mut pool) = self.owner.free.lock()
        {
            let used: usize = pool.iter().map(fabric::Buffer::len).sum();
            if pool.len() < 16 && self.bytes <= (64 * 1024 * 1024usize).saturating_sub(used) {
                pool.push(self.native.clone());
            }
        }
    }
}
/// Checked borrowed logical range in an owned buffer.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    owner: &'a Buffer,
    offset: usize,
    length: usize,
}
impl<'a> View<'a> {
    /// Select a checked range relative to this view.
    pub fn slice(self, offset: usize, length: usize) -> Result<Self> {
        checked_span(offset, length, self.length)?;
        Ok(Self {
            owner: self.owner,
            offset: self.offset + offset,
            length,
        })
    }
    /// Owning allocation.
    pub fn owner(self) -> &'a Buffer {
        self.owner
    }
    /// Absolute offset in the allocation.
    pub fn offset(self) -> usize {
        self.offset
    }
    /// Logical byte length.
    pub fn len(self) -> usize {
        self.length
    }
    /// Whether the range is empty.
    pub fn is_empty(self) -> bool {
        self.length == 0
    }
}
impl Buffer {
    // Tracked storage accounts for its own lifetime; its backing must not move
    // into an uncharged stream pool when that owner releases its reservation.
    pub(crate) fn into_unpooled(mut self) -> Self {
        self.poolable = false;
        self
    }
    #[cfg(feature = "npu")]
    pub(crate) fn share_with(mut self, device: &fabric::Device) -> Result<Self> {
        self.poolable = false;
        if self.native.device_address(device).is_err() {
            self.native = self
                .owner
                .device
                .fabric()
                .share_owned(&self.native, &[self.owner.device.clone(), device.clone()])?;
        }
        Ok(self)
    }
    pub(crate) fn charged_to(&self, budget: &crate::residency::MemoryBudget) -> bool {
        self.reservation
            .as_ref()
            .is_some_and(|reservation| budget.contains(reservation))
    }
    pub(crate) fn device_id(&self) -> usize {
        self.owner.device.id()
    }
    /// Allocation's exposed byte length.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Whole allocation as a logical binding.
    pub fn binding(&self) -> View<'_> {
        View {
            owner: self,
            offset: 0,
            length: self.bytes,
        }
    }
    /// Checked subrange.
    pub fn try_slice(&self, offset: usize, length: usize) -> Result<View<'_>> {
        self.binding().slice(offset, length)
    }
    /// Checked subrange, panicking on invalid bounds.
    pub fn slice(&self, offset: usize, length: usize) -> View<'_> {
        self.try_slice(offset, length)
            .expect("slice exceeds allocation")
    }
    /// Mapped host address. Dereferencing requires external synchronization.
    /// Ordinary allocations additionally require `cache_control` before host
    /// reads (including partial cache-line updates) and after host writes;
    /// `allocate_shared` omits that requirement.
    pub fn device_ptr(&self) -> Result<*mut std::ffi::c_void> {
        Ok(self.native.host_pointer().cast())
    }
    /// Publish host writes (`flush = true`) or acquire device writes (`false`).
    /// Stream upload/read operations perform these transitions automatically.
    ///
    /// # Safety
    /// The caller must exclude conflicting host/device access for the entire
    /// transition and retain the allocation until all accesses have completed.
    pub unsafe fn cache_control(&self, flush: bool, offset: usize, length: usize) -> Result<()> {
        unsafe { self.native.cache_control(flush, offset, length) }
    }
    pub(crate) fn allocation_address(&self) -> Result<u64> {
        self.native.device_address(&self.owner.device)
    }
}
/// Trusted native GPU entry and its declaration-order argument layout.
#[derive(Clone)]
pub struct Kernel {
    native: fabric::Kernel,
    info: ExportInfo,
    layout: Arc<[(u32, usize)]>,
    symbol: Arc<str>,
}
impl Kernel {
    pub(crate) fn device_id(&self) -> usize {
        self.native.device().id()
    }
    /// Checked argument and launch metadata.
    pub fn info(&self) -> &ExportInfo {
        &self.info
    }
    /// Selected native entry name.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    fn from_native(native: fabric::Kernel, symbol: &str) -> Result<Self> {
        let layout = native.argument_layout()?;
        let mut info = ExportInfo {
            workgroup_size: native.workgroup_size(),
            parameter_count: layout.len() as u32,
            ..Default::default()
        };
        for &(kind, size) in &layout {
            match kind {
                1 => {
                    info.constant_byte_length =
                        info.constant_byte_length
                            .checked_add(size as u32)
                            .ok_or_else(|| Error::Message("constant layout overflow".into()))?
                }
                2 => info.binding_count += 1,
                _ => {
                    return Err(Error::Unsupported(
                        "native argument kind is unsupported".into(),
                    ));
                }
            }
        }
        Ok(Self {
            native,
            info,
            layout: layout.into(),
            symbol: symbol.into(),
        })
    }
}
#[derive(Default)]
struct BudgetUses(std::collections::HashMap<usize, Arc<crate::residency::MemoryReservation>>);
impl BudgetUses {
    fn retain(&mut self, views: &[View<'_>]) {
        for view in views {
            if let Some(reservation) = &view.owner.reservation {
                self.0
                    .entry(Arc::as_ptr(reservation) as usize)
                    .or_insert_with(|| reservation.clone());
            }
        }
    }
    fn clear(&mut self) {
        self.0.clear();
    }
}
struct CachedDispatch {
    kernel: Kernel,
    grid: [u32; 3],
    block: [u32; 3],
    constants: Constants,
    bindings: Vec<(fabric::Buffer, usize, usize)>,
    command: fabric::PreparedGpu,
}
impl CachedDispatch {
    fn matches(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'_>],
    ) -> bool {
        self.kernel.native.same_entry(&kernel.native)
            && self.grid == grid
            && self.block == block
            && self.constants.as_bytes() == constants.as_bytes()
            && self.bindings.len() == bindings.len()
            && self
                .bindings
                .iter()
                .zip(bindings)
                .all(|((buffer, offset, length), view)| {
                    buffer.same_backing(&view.owner.native)
                        && *offset == view.offset
                        && *length == view.length
                })
    }
}
struct CachedTransfer {
    destination: fabric::Buffer,
    destination_offset: usize,
    length: usize,
    source: Option<(fabric::Buffer, usize)>,
    value: u8,
    command: fabric::PreparedGpu,
}
/// An ordered GPU stream with explicit prepared commands and bounded pools.
pub struct Stream {
    inner: Arc<Inner>,
    staging: Vec<Buffer>,
    staging_pool: Vec<Buffer>,
    scratch: BTreeMap<usize, Vec<Buffer>>,
    scratch_bytes: usize,
    scratch_limit: usize,
    budget: Option<crate::residency::MemoryBudget>,
    budget_uses: RefCell<BudgetUses>,
}
impl Stream {
    /// Open the first qualified GPU.
    pub fn open() -> Result<Self> {
        Device::open(0)?.stream()
    }
    /// Stable native device identity.
    pub fn device_id(&self) -> usize {
        self.inner.device.id()
    }
    /// Stream identity, unique while it remains live.
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
    /// Compiler deployment key.
    pub fn target(&self) -> &Target {
        self.inner.device.target()
    }
    /// Attach a shared allocation ceiling.
    pub fn with_memory_budget(mut self, budget: crate::residency::MemoryBudget) -> Self {
        self.budget = Some(budget);
        self
    }
    /// Selected allocation ceiling.
    pub fn memory_budget(&self) -> Option<&crate::residency::MemoryBudget> {
        self.budget.as_ref()
    }
    fn reserve(&self, bytes: usize) -> Result<Option<Arc<crate::residency::MemoryReservation>>> {
        self.budget
            .as_ref()
            .map(|budget| budget.reserve(bytes.max(1)).map(Arc::new))
            .transpose()
    }
    fn owns(&self, buffer: &Buffer) -> Result<()> {
        owns(&self.inner, buffer)
    }
    /// Allocate initialized native GPU-visible storage with unspecified contents.
    /// Empty requests round to one.
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(1);
        let reservation = self.reserve(bytes)?;
        let reused = if reservation.is_none() {
            let mut pool = self
                .inner
                .free
                .lock()
                .map_err(|_| Error::DeviceLost("allocation pool poisoned".into()))?;
            pool.iter()
                .position(|buffer| buffer.len() == bytes)
                .map(|index| pool.swap_remove(index))
        } else {
            None
        };
        let native = match reused {
            Some(native) => native,
            None => self
                .inner
                .device
                .fabric()
                .allocate(bytes, std::slice::from_ref(&self.inner.device))?,
        };
        Ok(Buffer {
            native,
            bytes,
            owner: self.inner.clone(),
            reservation,
            poolable: true,
        })
    }
    /// Allocate initialized storage containing zero bytes.
    pub fn allocate_zeroed(&self, bytes: usize) -> Result<Buffer> {
        let buffer = self.allocate(bytes)?;
        buffer.native.zero()?;
        Ok(buffer)
    }
    /// Allocate GPU-coherent host storage for direct, synchronized host access.
    pub fn allocate_shared(&self, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(1);
        let reservation = self.reserve(bytes)?;
        let native = self
            .inner
            .device
            .fabric()
            .allocate_shared(bytes, std::slice::from_ref(&self.inner.device))?;
        Ok(Buffer {
            native,
            bytes,
            owner: self.inner.clone(),
            reservation,
            poolable: false,
        })
    }
    #[cfg(feature = "npu")]
    pub(crate) fn allocate_for(&self, bytes: usize, devices: &[fabric::Device]) -> Result<Buffer> {
        let reservation = self.reserve(bytes)?;
        let native = self.inner.device.fabric().allocate(bytes, devices)?;
        native.device_address(&self.inner.device)?;
        Ok(Buffer {
            native,
            bytes,
            owner: self.inner.clone(),
            reservation,
            poolable: false,
        })
    }
    /// Wait for all preceding native work and reclaim transfer staging.
    pub fn synchronize(&mut self) -> Result<()> {
        self.inner.wait()?;
        self.reclaim_staging();
        self.budget_uses.get_mut().clear();
        Ok(())
    }
    /// Snapshot preceding work without a host wait.
    pub fn record_event(&mut self) -> Result<Event> {
        Ok(Event {
            device: self.device_id(),
            done: self
                .inner
                .last
                .lock()
                .map_err(|_| Error::DeviceLost("stream timeline poisoned".into()))?
                .clone(),
        })
    }
    /// Order subsequent stream work after an immutable event.
    pub fn wait_event(&mut self, event: &Event) -> Result<()> {
        if event.device != self.device_id() {
            return Err(Error::Message("event belongs to another device".into()));
        }
        if let Some(done) = &event.done {
            self.inner.submit(&self.inner.queue.prepare_wait(done)?)?;
        }
        Ok(())
    }
    /// Observe the current native submission prefix.
    pub fn submit(&mut self) -> Result<Submission<'_>> {
        Ok(Submission { stream: self })
    }
    /// Write a checked range after draining this stream.
    pub fn upload_blocking(&mut self, dst: View<'_>, bytes: &[u8]) -> Result<()> {
        self.owns(dst.owner)?;
        checked_span(0, bytes.len(), dst.len())?;
        self.synchronize()?;
        dst.owner.native.write(dst.offset, bytes)
    }
    /// Write at an allocation-relative byte offset.
    pub fn upload_blocking_at(&mut self, dst: &Buffer, offset: usize, bytes: &[u8]) -> Result<()> {
        self.upload_blocking(dst.try_slice(offset, bytes.len())?, bytes)
    }
    /// Read a checked range after draining this stream.
    pub fn read_blocking(&mut self, src: View<'_>, bytes: &mut [u8]) -> Result<()> {
        self.owns(src.owner)?;
        checked_span(0, bytes.len(), src.len())?;
        self.synchronize()?;
        src.owner.native.read(src.offset, bytes)
    }
    /// Read at an allocation-relative byte offset.
    pub fn read_blocking_at(
        &mut self,
        src: &Buffer,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<()> {
        self.read_blocking(src.try_slice(offset, bytes.len())?, bytes)
    }
    fn prepare_fill(&self, dst: View<'_>, value: u8) -> Result<fabric::PreparedGpu> {
        self.owns(dst.owner)?;
        let mut cache = self
            .inner
            .transfer_cache
            .lock()
            .map_err(|_| Error::DeviceLost("transfer cache poisoned".into()))?;
        if let Some(entry) = cache.iter().find(|entry| {
            entry.destination.same_backing(&dst.owner.native)
                && entry.destination_offset == dst.offset
                && entry.length == dst.length
                && entry.source.is_none()
                && entry.value == value
        }) {
            return Ok(entry.command.clone());
        }
        let command =
            self.inner
                .queue
                .prepare_fill(&dst.owner.native, dst.offset, dst.length, value)?;
        // Other streams may use this device-scoped allocation. Only cache when
        // its owning stream can evict the entry when the public buffer drops.
        if Arc::ptr_eq(&dst.owner.owner, &self.inner) && dst.owner.reservation.is_none() {
            if cache.len() == 128 {
                cache.remove(0);
            }
            cache.push(CachedTransfer {
                destination: dst.owner.native.clone(),
                destination_offset: dst.offset,
                length: dst.length,
                source: None,
                value,
                command: command.clone(),
            });
        }
        Ok(command)
    }
    fn prepare_copy(&self, dst: View<'_>, src: View<'_>) -> Result<fabric::PreparedGpu> {
        self.owns(dst.owner)?;
        self.owns(src.owner)?;
        if dst.len() != src.len() {
            return Err(Error::Message("copy requires equal spans".into()));
        }
        let mut cache = self
            .inner
            .transfer_cache
            .lock()
            .map_err(|_| Error::DeviceLost("transfer cache poisoned".into()))?;
        if let Some(entry) = cache.iter().find(|entry| {
            entry.destination.same_backing(&dst.owner.native)
                && entry.destination_offset == dst.offset
                && entry.length == dst.length
                && entry.source.as_ref().is_some_and(|(buffer, offset)| {
                    buffer.same_backing(&src.owner.native) && *offset == src.offset
                })
        }) {
            return Ok(entry.command.clone());
        }
        let command = self.inner.queue.prepare_copy(
            &dst.owner.native,
            dst.offset,
            &src.owner.native,
            src.offset,
            src.length,
        )?;
        if Arc::ptr_eq(&dst.owner.owner, &self.inner)
            && Arc::ptr_eq(&src.owner.owner, &self.inner)
            && dst.owner.reservation.is_none()
            && src.owner.reservation.is_none()
        {
            if cache.len() == 128 {
                cache.remove(0);
            }
            cache.push(CachedTransfer {
                destination: dst.owner.native.clone(),
                destination_offset: dst.offset,
                length: dst.length,
                source: Some((src.owner.native.clone(), src.offset)),
                value: 0,
                command: command.clone(),
            });
        }
        Ok(command)
    }
    /// Enqueue a native byte-pattern fill.
    pub fn fill(&self, dst: View<'_>, value: u8) -> Result<()> {
        self.budget_uses.borrow_mut().retain(&[dst]);
        self.inner.submit(&self.prepare_fill(dst, value)?)
    }
    /// Enqueue a native copy between equal, non-overlapping ranges.
    pub fn copy(&self, dst: View<'_>, src: View<'_>) -> Result<()> {
        self.budget_uses.borrow_mut().retain(&[dst, src]);
        self.inner.submit(&self.prepare_copy(dst, src)?)
    }
    /// Retain an owned staging copy of host bytes and enqueue a GPU transfer.
    pub fn upload(&mut self, dst: View<'_>, bytes: &[u8]) -> Result<()> {
        self.owns(dst.owner)?;
        checked_span(0, bytes.len(), dst.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        if self.staging.len() >= 8
            || self
                .staging
                .iter()
                .map(|b| b.bytes)
                .sum::<usize>()
                .saturating_add(bytes.len())
                > STAGING_LIMIT
        {
            self.synchronize()?;
        }
        let staging = if let Some(index) = self
            .staging_pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bytes >= bytes.len())
            .min_by_key(|(_, b)| b.bytes)
            .map(|(index, _)| index)
        {
            self.staging_pool.swap_remove(index)
        } else {
            self.allocate(bytes.len())?
        };
        staging.native.write(0, bytes)?;
        self.copy(
            dst.slice(0, bytes.len())?,
            staging.try_slice(0, bytes.len())?,
        )?;
        self.staging.push(staging);
        Ok(())
    }
    /// Enqueue a checked upload at an allocation-relative offset.
    pub fn upload_at(&mut self, dst: &Buffer, offset: usize, bytes: &[u8]) -> Result<()> {
        self.upload(dst.try_slice(offset, bytes.len())?, bytes)
    }
    fn reclaim_staging(&mut self) {
        let mut cached = self.staging_pool.iter().map(|b| b.bytes).sum::<usize>();
        for buffer in self.staging.drain(..) {
            if buffer.bytes > STAGING_LIMIT {
                continue;
            }
            while self.staging_pool.len() >= 8
                || buffer.bytes > STAGING_LIMIT.saturating_sub(cached)
            {
                let index = self
                    .staging_pool
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, b)| b.bytes)
                    .map(|(i, _)| i)
                    .unwrap();
                cached -= self.staging_pool.swap_remove(index).bytes;
            }
            cached += buffer.bytes;
            self.staging_pool.push(buffer);
        }
    }
    /// Load a trusted native code object from disk.
    ///
    /// # Safety
    /// Native code must obey its declared memory and argument contract.
    pub unsafe fn load(&self, path: &Path, symbol: &str) -> Result<Kernel> {
        Kernel::from_native(
            unsafe { self.inner.device.load_bytes(&std::fs::read(path)?, symbol) }?,
            symbol,
        )
    }
    /// Load a compiler artifact for this exact device target.
    ///
    /// # Safety
    /// Native code must obey its declared memory and argument contract.
    pub unsafe fn load_artifact(&self, artifact: &crate::loom::Artifact) -> Result<Kernel> {
        Kernel::from_native(
            unsafe { self.inner.device.load(artifact) }?,
            artifact.symbol(),
        )
    }
    unsafe fn prepare_dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'_>],
    ) -> Result<fabric::PreparedGpu> {
        if kernel.device_id() != self.device_id() {
            return Err(Error::Message("kernel belongs to another device".into()));
        }
        validate_export_launch(&kernel.info, grid, block)?;
        if constants.len != kernel.info.constant_byte_length as usize
            || bindings.len() != kernel.info.binding_count as usize
        {
            return Err(Error::Message(
                "kernel binding or constant byte count mismatch".into(),
            ));
        }
        for view in bindings {
            self.owns(view.owner)?;
        }
        let mut args = Vec::with_capacity(kernel.layout.len());
        let mut scalar = 0;
        let mut binding = 0;
        for &(kind, size) in kernel.layout.iter() {
            if kind == 1 {
                args.push(fabric::Argument::Value(
                    &constants.bytes[scalar..scalar + size],
                ));
                scalar += size;
            } else {
                let view = bindings[binding];
                args.push(fabric::Argument::Buffer(&view.owner.native, view.offset));
                binding += 1;
            }
        }
        unsafe {
            self.inner
                .queue
                .prepare(&kernel.native, grid, block.map(|v| v as u16), &args)
        }
    }
    /// Enqueue a kernel invocation with declaration-order constants and buffers.
    ///
    /// # Safety
    /// Dimensions, constants, and actual kernel accesses must obey each binding's range.
    pub unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'_>],
    ) -> Result<()> {
        self.budget_uses.borrow_mut().retain(bindings);
        if self.budget.is_none()
            && bindings.iter().all(|view| {
                Arc::ptr_eq(&view.owner.owner, &self.inner) && view.owner.reservation.is_none()
            })
        {
            let mut cache = self
                .inner
                .dispatch_cache
                .lock()
                .map_err(|_| Error::DeviceLost("dispatch cache poisoned".into()))?;
            if let Some(entry) = cache
                .iter()
                .find(|entry| entry.matches(kernel, grid, block, constants, bindings))
            {
                return self.inner.submit(&entry.command);
            }
            let command =
                unsafe { self.prepare_dispatch(kernel, grid, block, constants, bindings) }?;
            self.inner.submit(&command)?;
            if cache.len() == 64 {
                cache.remove(0);
            }
            cache.push(CachedDispatch {
                kernel: kernel.clone(),
                grid,
                block,
                constants: constants.clone(),
                bindings: bindings
                    .iter()
                    .map(|view| (view.owner.native.clone(), view.offset, view.length))
                    .collect(),
                command,
            });
            Ok(())
        } else {
            self.inner.submit(&unsafe {
                self.prepare_dispatch(kernel, grid, block, constants, bindings)
            }?)
        }
    }
    /// Reuse scratch storage under this queue's execution order.
    pub fn scratch(&mut self, bytes: usize) -> Result<Buffer> {
        let choice = self
            .scratch
            .range(bytes.max(1)..=bytes.saturating_mul(2).max(1))
            .next()
            .map(|(&size, _)| size);
        if let Some(size) = choice {
            let list = self.scratch.get_mut(&size).unwrap();
            let buffer = list.pop().unwrap();
            if list.is_empty() {
                self.scratch.remove(&size);
            }
            self.scratch_bytes -= buffer.bytes;
            Ok(buffer)
        } else {
            self.allocate(bytes)
        }
    }
    /// Return this stream's allocation after recording its last use.
    pub fn recycle(&mut self, buffer: Buffer) -> Result<()> {
        if !Arc::ptr_eq(&buffer.owner, &self.inner) {
            return Err(Error::Message("scratch belongs to another stream".into()));
        }
        if buffer.bytes <= self.scratch_limit.saturating_sub(self.scratch_bytes) {
            self.scratch_bytes += buffer.bytes;
            self.scratch.entry(buffer.bytes).or_default().push(buffer);
        }
        Ok(())
    }
    /// Bound retained scratch storage, evicting large entries first.
    pub fn set_scratch_limit(&mut self, bytes: usize) {
        self.scratch_limit = bytes;
        while self.scratch_bytes > bytes {
            if let Some((_, list)) = self.scratch.pop_last() {
                self.scratch_bytes -= list.iter().map(|b| b.bytes).sum::<usize>();
            } else {
                break;
            }
        }
    }
    /// Queue an owned readback.
    pub fn read(&mut self, source: View<'_>) -> Result<Readback> {
        let buffer = self.allocate(source.len())?;
        if !source.is_empty() {
            self.copy(buffer.try_slice(0, source.len())?, source)?;
        }
        Ok(Readback {
            buffer,
            length: source.len(),
        })
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        if !self.inner.drain_for_drop() {
            std::mem::forget(std::mem::take(&mut self.staging));
            std::mem::forget(std::mem::take(self.budget_uses.get_mut()));
        }
    }
}
/// Immutable native completion point.
#[derive(Clone)]
pub struct Event {
    device: usize,
    done: Option<fabric::Completion>,
}
impl Event {
    /// Poll terminal completion.
    pub fn is_complete(&self) -> Result<bool> {
        self.done
            .as_ref()
            .map_or(Ok(true), fabric::Completion::is_complete)
    }
    /// Wait on the host for preceding work.
    pub fn synchronize(&self) -> Result<()> {
        self.done.as_ref().map_or(Ok(()), fabric::Completion::wait)
    }
}
/// Borrowed completion observer that also reclaims completed staging.
#[must_use]
pub struct Submission<'a> {
    stream: &'a mut Stream,
}
impl Submission<'_> {
    /// Poll submitted work and reclaim its completed staging.
    pub fn is_complete(&mut self) -> Result<bool> {
        let done = self
            .stream
            .inner
            .last
            .lock()
            .map_err(|_| Error::DeviceLost("stream timeline poisoned".into()))?
            .as_ref()
            .map_or(Ok(true), fabric::Completion::is_complete)?;
        if done {
            self.stream.reclaim_staging();
            self.stream.budget_uses.get_mut().clear();
        }
        Ok(done)
    }
    /// Wait and reclaim staging.
    pub fn wait(self) -> Result<()> {
        self.stream.synchronize()
    }
}
/// Owned destination retained through native readback completion.
#[must_use]
pub struct Readback {
    buffer: Buffer,
    length: usize,
}
impl Readback {
    /// Wait on the originating stream and return initialized bytes.
    pub fn wait(self, stream: &mut Stream) -> Result<Vec<u8>> {
        if !Arc::ptr_eq(&self.buffer.owner, &stream.inner) {
            return Err(Error::Message("readback belongs to another stream".into()));
        }
        stream.synchronize()?;
        let mut bytes = vec![0; self.length];
        self.buffer.native.read(0, &mut bytes)?;
        Ok(bytes)
    }
}
fn owns(inner: &Arc<Inner>, buffer: &Buffer) -> Result<()> {
    if inner.device.id() == buffer.device_id() {
        Ok(())
    } else {
        Err(Error::Message("buffer belongs to another device".into()))
    }
}
pub(crate) fn checked_span(offset: usize, length: usize, capacity: usize) -> Result<()> {
    if offset.checked_add(length).is_none_or(|end| end > capacity) {
        Err(Error::Message(format!(
            "span {offset}+{length} exceeds {capacity} bytes"
        )))
    } else {
        Ok(())
    }
}
pub(crate) fn validate_launch(grid: [u32; 3], block: [u32; 3]) -> Result<()> {
    if grid.contains(&0)
        || block.contains(&0)
        || block
            .iter()
            .try_fold(1u32, |n, b| n.checked_mul(*b))
            .is_none_or(|n| n > 1024)
    {
        Err(Error::Message("invalid GPU launch dimensions".into()))
    } else {
        Ok(())
    }
}
pub(crate) fn validate_export_launch(
    info: &ExportInfo,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<()> {
    validate_launch(grid, block)?;
    if info
        .workgroup_size
        .iter()
        .zip(block)
        .any(|(&expected, actual)| expected != 0 && expected != actual)
    {
        Err(Error::Message(format!(
            "workgroup size {block:?} disagrees with compiled size {:?}",
            info.workgroup_size
        )))
    } else {
        Ok(())
    }
}
const STAGING_LIMIT: usize = 64 * 1024 * 1024;

/// Explicit scalar widths, packed in declaration order for HRX binding dispatch.
/// Unlike direct kernargs this format has no implicit alignment or pointer slots.
/// Export metadata gives only the total byte count. Mixed scalar widths must
/// come from the kernel's argument contract.
#[derive(Clone, Debug)]
pub struct Constants {
    bytes: [u8; 256],
    len: usize,
}
impl Default for Constants {
    fn default() -> Self {
        Self {
            bytes: [0; 256],
            len: 0,
        }
    }
}
impl Constants {
    /// Pack an entry point whose scalar arguments are all unsigned Loom indices.
    ///
    /// Loom lowers a homogeneous index list to 32- or 64-bit slots depending on
    /// its range analysis. This adapter is only for that model contract; mixed
    /// scalar types must use `push` with their explicit native widths.
    ///
    /// Neither [`ExportInfo`] nor the compiler's resource report carries per-slot
    /// scalar types — only `constant_byte_length` and `parameter_count` — so no
    /// API here can build a mixed-width constant block by construction. A width
    /// inferred by dividing the byte count is wrong for a mixed `u32`/`u64`
    /// signature, and wrong quietly. Take each width from the source that
    /// declared it.
    pub fn indices(kernel: &Kernel, values: &[u32]) -> Result<Self> {
        let size = kernel.info.constant_byte_length as usize;
        let mut constants = Self::new();
        if values.is_empty() {
            if size != 0 {
                return Err(Error::Message("missing index constants".into()));
            }
            return Ok(constants);
        }
        if size == values.len() * 4 {
            for &value in values {
                constants.push(value)?;
            }
        } else if size == values.len() * 8 {
            for &value in values {
                constants.push(u64::from(value))?;
            }
        } else {
            return Err(Error::Message(
                "export is not a homogeneous 32/64-bit index list".into(),
            ));
        }
        Ok(constants)
    }
    /// Create an empty constant block.
    pub fn new() -> Self {
        Self::default()
    }
    #[must_use = "constant packing can fail when capacity is exceeded"]
    /// Append a scalar at its explicit width; fails if the 256-byte capacity is exceeded.
    pub fn push<T: Scalar>(&mut self, value: T) -> Result<()> {
        let bytes = value.bytes();
        checked_span(self.len, bytes.as_ref().len(), self.bytes.len())?;
        self.bytes[self.len..self.len + bytes.as_ref().len()].copy_from_slice(bytes.as_ref());
        self.len += bytes.as_ref().len();
        Ok(())
    }
    /// Construct an explicitly contracted scalar block from packed bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut result = Self::new();
        checked_span(0, bytes.len(), result.bytes.len())?;
        result.bytes[..bytes.len()].copy_from_slice(bytes);
        result.len = bytes.len();
        Ok(result)
    }
    /// The packed constant bytes in declaration order.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}
mod sealed {
    pub trait Sealed {}
}
/// A supported fixed-width scalar for little-endian binding constants.
pub trait Scalar: sealed::Sealed {
    /// The scalar's fixed-size byte representation.
    type Bytes: AsRef<[u8]>;
    /// Encode this value in little-endian order.
    fn bytes(self) -> Self::Bytes;
}
macro_rules! scalars {
    ($($ty:ty => $n:literal),*) => {
        $(
            impl sealed::Sealed for $ty {}
            impl Scalar for $ty {
                type Bytes = [u8; $n];
                fn bytes(self) -> Self::Bytes { self.to_le_bytes() }
            }
        )*
    };
}
scalars!(u32 => 4, i32 => 4, f32 => 4, u64 => 8, i64 => 8, f64 => 8);

/// A dependency node branded with its recording's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    graph: u64,
    index: u32,
}
static NEXT_GRAPH_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Owned native commands being recorded under checked dependency edges.
pub struct Graph<'a> {
    stream: &'a Stream,
    id: u64,
    nodes: Vec<(Recorded<'a>, Option<u32>)>,
    budget_uses: BudgetUses,
}
enum Recorded<'a> {
    Fill(View<'a>, u8),
    Copy(View<'a>, View<'a>),
    Prepared(fabric::PreparedGpu),
    Join,
}
/// Reusable prepared GPU graph and its last immutable completion point.
pub struct GraphExec {
    inner: Arc<Inner>,
    commands: Option<fabric::PreparedGpu>,
    last: Option<fabric::Completion>,
    failed: bool,
    budget_uses: BudgetUses,
}
impl Stream {
    /// Begin a dependency-checked recording borrowing this stream.
    pub fn graph(&self) -> Result<Graph<'_>> {
        Ok(Graph {
            stream: self,
            id: NEXT_GRAPH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            nodes: Vec::new(),
            budget_uses: BudgetUses::default(),
        })
    }
    /// Replay fixed native commands on the originating stream.
    pub fn launch(&mut self, graph: &mut GraphExec) -> Result<()> {
        if !Arc::ptr_eq(&graph.inner, &self.inner) {
            return Err(Error::Message("graph belongs to another stream".into()));
        }
        if graph.failed {
            return Err(Error::DeviceLost(
                "graph has an unretired failed launch".into(),
            ));
        }
        graph.failed = true;
        if let Some(command) = &graph.commands {
            self.inner.submit(command)?;
        }
        graph.last = self
            .inner
            .last
            .lock()
            .map_err(|_| Error::DeviceLost("stream timeline poisoned".into()))?
            .clone();
        graph.failed = false;
        Ok(())
    }
}
impl<'a> Graph<'a> {
    fn validate_dependencies(&self, after: &[Node]) -> Result<()> {
        for node in after {
            if node.graph != self.id || node.index as usize >= self.nodes.len() {
                return Err(Error::Message(
                    "dependency node belongs to another graph".into(),
                ));
            }
        }
        if self.nodes.len() >= u32::MAX as usize {
            return Err(Error::Message("graph node capacity exceeded".into()));
        }
        Ok(())
    }
    fn record(&mut self, command: Recorded<'a>, after: &[Node]) -> Node {
        let index = self.nodes.len() as u32;
        self.nodes
            .push((command, after.iter().map(|node| node.index).max()));
        Node {
            graph: self.id,
            index,
        }
    }
    /// Record a byte fill after checked predecessor nodes.
    pub fn fill(&mut self, after: &[Node], dst: View<'a>, pattern: u8) -> Result<Node> {
        self.validate_dependencies(after)?;
        self.stream.owns(dst.owner)?;
        if dst.is_empty() {
            return Err(Error::Message("empty graph fill".into()));
        }
        self.budget_uses.retain(&[dst]);
        Ok(self.record(Recorded::Fill(dst, pattern), after))
    }
    /// Record a non-overlapping byte copy after checked predecessor nodes.
    pub fn copy(&mut self, after: &[Node], dst: View<'a>, src: View<'a>) -> Result<Node> {
        self.validate_dependencies(after)?;
        self.stream.owns(dst.owner)?;
        self.stream.owns(src.owner)?;
        if dst.len() != src.len() || dst.is_empty() {
            return Err(Error::Message(
                "graph copy requires equal nonempty spans".into(),
            ));
        }
        self.budget_uses.retain(&[dst, src]);
        Ok(self.record(Recorded::Copy(dst, src), after))
    }
    /// Record a fixed native kernel invocation.
    ///
    /// # Safety
    /// Kernel accesses must obey bindings and constants; dependency edges must
    /// order conflicting accesses for every replay.
    pub unsafe fn dispatch(
        &mut self,
        after: &[Node],
        kernel: &'a Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        constants: &Constants,
        bindings: &[View<'a>],
    ) -> Result<Node> {
        self.validate_dependencies(after)?;
        let command = unsafe {
            self.stream
                .prepare_dispatch(kernel, grid, block, constants, bindings)
        }?;
        self.budget_uses.retain(bindings);
        Ok(self.record(Recorded::Prepared(command), after))
    }
    /// Record a dependency join with no native payload.
    pub fn join(&mut self, after: &[Node]) -> Result<Node> {
        self.validate_dependencies(after)?;
        Ok(self.record(Recorded::Join, after))
    }
    /// Transfer recorded commands and allocation charges into owned replay state.
    pub fn finish(self) -> Result<GraphExec> {
        let mut ordered = Vec::new();
        let mut completed = 0usize;
        for (index, (node, dependency)) in self.nodes.into_iter().enumerate() {
            let command = match node {
                Recorded::Fill(dst, value) => self.stream.prepare_fill(dst, value)?,
                Recorded::Copy(dst, src) => self.stream.prepare_copy(dst, src)?,
                Recorded::Prepared(command) => command,
                Recorded::Join => continue,
            };
            let barrier = dependency.is_some_and(|node| node as usize >= completed);
            if barrier {
                completed = index;
            }
            ordered.push((command, barrier));
        }
        let commands = if ordered.is_empty() {
            None
        } else {
            // Graph edges order conflicting accesses; all binding backing is retained.
            Some(unsafe { self.stream.inner.queue.prepare_batch(&ordered) }?)
        };
        Ok(GraphExec {
            inner: self.stream.inner.clone(),
            commands,
            last: None,
            failed: false,
            budget_uses: self.budget_uses,
        })
    }
}
impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.failed
            || self.last.as_ref().is_some_and(|done| {
                !done
                    .wait_timeout(std::time::Duration::from_secs(10))
                    .unwrap_or(false)
            })
        {
            std::mem::forget(std::mem::take(&mut self.budget_uses));
        }
    }
}
impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("target", self.target())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("symbol", &self.symbol)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Graph<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.nodes.len())
            .finish_non_exhaustive()
    }
}
impl std::fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExec").finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Readback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Readback").finish_non_exhaustive()
    }
}
impl std::fmt::Debug for Submission<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn compiled_workgroup_dimensions_are_enforced() {
        let info = super::ExportInfo {
            workgroup_size: [64, 2, 1],
            ..Default::default()
        };
        assert!(super::validate_export_launch(&info, [1; 3], [64, 2, 1]).is_ok());
        assert!(super::validate_export_launch(&info, [1; 3], [128, 1, 1]).is_err());
        assert!(super::validate_export_launch(&info, [0, 1, 1], [64, 2, 1]).is_err());
        let dynamic = super::ExportInfo {
            workgroup_size: [0, 2, 1],
            ..Default::default()
        };
        assert!(super::validate_export_launch(&dynamic, [1; 3], [32, 2, 1]).is_ok());
        assert!(super::validate_export_launch(&dynamic, [1; 3], [32, 1, 1]).is_err());
    }
    /// The bounds arithmetic, without a device: `offset + length` must not wrap into a pass.
    #[test]
    fn a_slice_past_the_allocation_is_refused_even_when_the_sum_wraps() {
        let checked = |offset, length, bytes| super::checked_span(offset, length, bytes).is_ok();
        assert!(checked(0, 8, 8));
        assert!(checked(4, 4, 8));
        assert!(!checked(4, 5, 8));
        // the wrapping case: 8 bytes must not accept an offset near the top of the address space
        assert!(!checked(usize::MAX, 2, 8));
        assert!(!checked(usize::MAX - 1, 4, 8));
    }
}

#[cfg(test)]
mod staging_tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151"]
    fn a_scratch_pool_only_accepts_the_stream_that_allocated_the_buffer() -> Result<()> {
        let device = Device::open(0)?;
        let mut owner = device.stream()?;
        let mut other = device.stream()?;
        let private = other.scratch(4096)?;
        let private_raw = private.allocation_address()?;
        other.recycle(private)?;

        // Buffers are device-scoped, so another stream may read and write this one.
        let buffer = owner.allocate(4096)?;
        owner.fill(buffer.binding(), 0x5a)?;
        let ready = owner.record_event()?;
        other.wait_event(&ready)?;
        let mut seen = [0u8; 16];
        other.read_blocking(buffer.binding().slice(0, 16)?, &mut seen)?;
        assert_eq!(seen, [0x5a; 16], "a foreign stream can use the allocation");

        // A scratch pool is still one stream's private free list, not shared state.
        assert!(
            other
                .recycle(buffer)
                .unwrap_err()
                .to_string()
                .contains("another stream")
        );
        assert!(owner.scratch.is_empty());
        assert_eq!(other.scratch_bytes, 4096);
        assert_eq!(other.scratch(4096)?.allocation_address()?, private_raw);
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn completed_staging_is_reclaimed_and_reused_on_every_completion_path() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(1024)?;
        stream.upload(buffer.binding(), &[7; 1024])?;
        let original = stream.staging[0].allocation_address()?;
        let mut output = [0; 1024];
        stream.read_blocking(buffer.binding(), &mut output)?;
        assert_eq!(output, [7; 1024]);
        assert!(stream.staging.is_empty());
        stream.upload(buffer.binding(), &[9; 1024])?;
        assert_eq!(stream.staging[0].allocation_address()?, original);
        stream.upload_blocking(buffer.binding(), &[11; 1024])?;
        assert!(stream.staging.is_empty());
        stream.upload(buffer.binding(), &[12; 512])?;
        assert_eq!(stream.staging[0].allocation_address()?, original);
        stream.synchronize()?;
        stream.upload(buffer.binding(), &[13; 1024])?;
        stream.inner.wait()?; // Establish completion without the cleanup being tested.
        assert!(stream.submit()?.is_complete()?);
        assert!(stream.staging.is_empty());
        for _ in 0..32 {
            stream.upload(buffer.binding(), &[1; 1024])?;
            assert!(stream.staging.len() <= 8);
        }
        stream.synchronize()?;
        assert!(stream.staging_pool.len() <= 8);
        assert!(stream.staging_pool.iter().map(|b| b.bytes).sum::<usize>() <= STAGING_LIMIT);
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn uploads_batch_until_pressure_and_reuse_the_smallest_capacity() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(4096)?;
        for size in [1024, 2048, 4096] {
            stream.upload(buffer.binding(), &vec![7; size])?;
        }
        assert_eq!(stream.staging.len(), 3);
        let medium = stream.staging[1].allocation_address()?;
        stream.synchronize()?;
        stream.upload(buffer.binding(), &[9; 1500])?;
        assert_eq!(stream.staging[0].allocation_address()?, medium);
        for _ in 0..7 {
            stream.upload(buffer.binding(), &[11; 1024])?;
        }
        assert_eq!(stream.staging.len(), 8);
        stream.upload(buffer.binding(), &[13; 1024])?;
        assert_eq!(stream.staging.len(), 1);
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn larger_uploads_replace_small_completed_staging() -> Result<()> {
        let mut stream = Stream::open()?;
        let buffer = stream.allocate(16 << 20)?;
        for _ in 0..8 {
            stream.upload(buffer.binding(), &[7; 1024])?;
        }
        stream.synchronize()?;
        assert_eq!(stream.staging_pool.len(), 8);

        let source = vec![0x35; 1 << 20];
        stream.upload(buffer.binding(), &source)?;
        let large = stream.staging[0].allocation_address()?;
        stream.synchronize()?;
        let replacement = vec![0xa9; source.len()];
        stream.upload(buffer.binding(), &replacement)?;
        assert_eq!(stream.staging[0].allocation_address()?, large);
        let mut output = vec![0; source.len()];
        stream.read_blocking(buffer.try_slice(0, output.len())?, &mut output)?;
        assert_eq!(output, replacement);
        assert!(stream.staging_pool.len() <= 8);
        assert!(stream.staging_pool.iter().map(|b| b.bytes).sum::<usize>() <= STAGING_LIMIT);

        // Larger packets also exercise eviction at the byte ceiling, before
        // the cache reaches its eight-entry ceiling.
        let source = vec![0x63; 16 << 20];
        for _ in 0..9 {
            stream.upload(buffer.binding(), &source)?;
        }
        let mut output = vec![0; source.len()];
        stream.read_blocking(buffer.binding(), &mut output)?;
        assert_eq!(output, source);
        assert!(stream.staging_pool.len() <= 8);
        assert!(stream.staging_pool.iter().map(|b| b.bytes).sum::<usize>() <= STAGING_LIMIT);
        Ok(())
    }
}

#[cfg(test)]
mod dag_probe {
    use super::*;

    /// Declared edges cost scheduling time, so the graph API must be able to omit
    /// the ones a workload does not need. Latency-bound by construction: 64 tiny
    /// fills, so bandwidth cannot confound the comparison.
    #[test]
    #[ignore = "requires gfx1151"]
    fn independent_nodes_beat_a_serial_chain() -> Result<()> {
        const NODES: usize = 64;
        let mut stream = Stream::open()?;
        let buffers: Vec<Buffer> = (0..NODES)
            .map(|_| stream.allocate(4096))
            .collect::<Result<_>>()?;
        let mut timings = Vec::new();
        for chained in [true, false] {
            let mut graph = stream.graph()?;
            let mut previous: Option<Node> = None;
            for buffer in &buffers {
                let after: &[Node] = match (chained, &previous) {
                    (true, Some(node)) => std::slice::from_ref(node),
                    _ => &[],
                };
                previous = Some(graph.fill(after, buffer.binding(), 0x3c)?);
            }
            let mut exec = graph.finish()?;
            for _ in 0..3 {
                stream.launch(&mut exec)?;
                stream.synchronize()?;
            }
            let mut samples = Vec::new();
            for _ in 0..9 {
                let start = std::time::Instant::now();
                stream.launch(&mut exec)?;
                stream.synchronize()?;
                samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{:<12} {NODES} nodes: {:8.1} us total, {:6.2} us/node",
                if chained { "serial" } else { "independent" },
                samples[4],
                samples[4] / NODES as f64
            );
            timings.push(samples[4]);
            let mut seen = [0u8; 4096];
            stream.read_blocking(buffers[NODES - 1].binding(), &mut seen)?;
            assert_eq!(seen, [0x3c; 4096], "every node ran");
        }
        // A margin rather than a strict inequality: the property under test is
        // that omitting edges still reaches the runtime, not that a shared CI
        // runner is quiet. The measured gap is ~1.7x, so 1.2x is a wide floor.
        assert!(
            timings[1] * 1.2 < timings[0],
            "omitting edges should be materially cheaper: {timings:?}"
        );
        Ok(())
    }
}
