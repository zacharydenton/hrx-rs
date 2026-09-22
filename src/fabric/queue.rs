use super::*;
use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

/// A serialized host producer for one native GPU PM4 queue.
#[derive(Clone)]
pub struct Queue(Arc<QueueInner>);
struct QueueInner {
    device: Device,
    raw: *mut amdf_user_queue_t,
    mapping: *mut amdf_user_queue_mapping_t,
    info: amdf_user_queue_mapping_info_t,
    state: Mutex<QueueState>,
    transfers: Mutex<Option<[Kernel; 4]>>,
}
struct QueueState {
    write: u64,
    pending: VecDeque<Pending>,
}
struct Pending {
    work: Arc<Work>,
    value: u32,
}
struct Work {
    fence: Buffer,
    _kernels: Vec<Kernel>,
    buffers: Vec<Buffer>,
    leases: Mutex<Vec<memory::DeviceUse>>,
    retired: AtomicU32,
    active: AtomicU32,
    dependencies: Vec<(Arc<Work>, u32)>,
}
// Command publication has one mutex-protected producer. Status observation is
// documented as concurrent; mapped native pointers remain valid until drop.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

/// Native execution completion retaining all indirect resources.
#[derive(Clone)]
pub struct Completion {
    queue: Queue,
    work: Arc<Work>,
    value: u32,
}
impl Completion {
    /// Observe a completion fence, never just the queue's consumed index.
    pub fn is_complete(&self) -> Result<bool> {
        if !self.queue.0.poll(&self.work, self.value)? {
            return Ok(false);
        }
        // A completed prefix must release every earlier submission's leases,
        // even when callers observed only the last completion point.
        let mut state = self
            .queue
            .0
            .state
            .lock()
            .map_err(|_| Error::DeviceLost("queue producer poisoned".into()))?;
        while let Some(work) = state.pending.front() {
            if !self.queue.0.poll(&work.work, work.value)? {
                break;
            }
            state.pending.pop_front();
        }
        Ok(true)
    }
    /// Wait without cancelling or releasing pending resources on timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let start = Instant::now();
        loop {
            if self.is_complete()? {
                return Ok(true);
            }
            if start.elapsed() >= timeout {
                return Ok(false);
            }
            std::thread::yield_now();
        }
    }
    /// Wait until execution retires or the provider reports failure.
    pub fn wait(&self) -> Result<()> {
        while !self.is_complete()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}
impl Work {
    fn retire(&self, value: u32) -> Result<bool> {
        if self.retired.load(Ordering::Acquire) >= value {
            return Ok(true);
        }
        let mut leases = self
            .leases
            .lock()
            .map_err(|_| Error::DeviceLost("submission ownership poisoned".into()))?;
        if self.retired.load(Ordering::Acquire) >= value {
            return Ok(true);
        }
        let completed = self.fence.fence_value()?;
        if completed >= value {
            for (dependency, point) in &self.dependencies {
                if !dependency.retire(*point)? {
                    return Ok(false);
                }
            }
            if completed >= self.active.load(Ordering::Acquire) {
                leases.clear();
            }
            self.retired.store(completed, Ordering::Release);
            return Ok(true);
        }
        Ok(false)
    }
}
impl QueueInner {
    fn poll(&self, work: &Work, value: u32) -> Result<bool> {
        if work.retire(value)? {
            return Ok(true);
        }
        let api = self.device.0.endpoint.0.instance.api.core;
        let mut status = amdf_user_queue_status_t {
            type_: AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
            structure_size: size_of::<amdf_user_queue_status_t>() as u32,
            ..Default::default()
        };
        check("user_queue_query_status", unsafe {
            entry!(api, user_queue_query_status)(self.raw, &mut status)
        })?;
        if status.state != AMDF_QUEUE_STATE_ACTIVE {
            return Err(Error::DeviceLost(format!(
                "PM4 queue failed with native status {}",
                status.terminal_status
            )));
        }
        Ok(false)
    }
}
impl Drop for QueueInner {
    fn drop(&mut self) {
        let pending = std::mem::take(
            &mut self
                .state
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .pending,
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        for work in &pending {
            loop {
                match self.poll(&work.work, work.value) {
                    Ok(true) => break,
                    Ok(false) if Instant::now() < deadline => std::thread::yield_now(),
                    Ok(false) | Err(_) => {
                        // Failure is not retirement. Leak the entire native
                        // ownership chain instead of unmapping reachable memory.
                        std::mem::forget(pending);
                        std::mem::forget(self.device.clone());
                        return;
                    }
                }
            }
        }
        let api = self.device.0.endpoint.0.instance.api.core;
        unsafe {
            if !self.mapping.is_null() && api.user_queue_mapping_destroy.unwrap()(self.mapping) != 0
            {
                std::mem::forget(self.device.clone());
                return;
            }
            if !self.raw.is_null() && api.user_queue_destroy.unwrap()(self.raw) != 0 {
                std::mem::forget(self.device.clone());
            }
        }
    }
}

fn header(opcode: u32, count: usize) -> u32 {
    (3 << 30) | (((count - 2) as u32) << 16) | (opcode << 8)
}
fn barrier(words: &mut Vec<u32>) {
    // Native PM4 v1/GCR system release and acquire, matching libamdf CTS.
    words.extend([
        header(0x46, 2),
        7 | (4 << 8),
        header(0x58, 8),
        0,
        u32::MAX,
        0xff,
        0,
        0,
        0x0a,
        3 | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15),
    ]);
}
fn pad(words: &mut Vec<u32>) {
    let mut count = 8 - words.len() % 8;
    if count == 1 {
        count += 8;
    }
    words.push(header(0x10, count));
    words.resize(words.len() + count - 1, 0);
}

impl Device {
    /// Create a native PM4 queue with one serialized host producer.
    pub fn queue(&self) -> Result<Queue> {
        if self.endpoint().engine() != Engine::Gpu {
            return Err(Error::Unsupported(
                "PM4 queues require an AMDGPU device".into(),
            ));
        }
        let api = &self.0.endpoint.0.instance.api;
        let _ = entry!(api.core, user_queue_destroy);
        let _ = entry!(api.core, user_queue_mapping_destroy);
        let mut endpoint = amdf_endpoint_info_t {
            type_: AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
            structure_size: size_of::<amdf_endpoint_info_t>() as u32,
            ..Default::default()
        };
        check("endpoint_query_info", unsafe {
            entry!(api.core, endpoint_query_info)(self.0.endpoint.0.raw, &mut endpoint)
        })?;
        for ordinal in 0..endpoint.queue_family_count {
            let mut family = amdf_queue_family_info_t {
                type_: AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
                structure_size: size_of::<amdf_queue_family_info_t>() as u32,
                ..Default::default()
            };
            check("endpoint_query_queue_family_info", unsafe {
                entry!(api.core, endpoint_query_queue_family_info)(
                    self.0.endpoint.0.raw,
                    ordinal,
                    &mut family,
                )
            })?;
            if family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                || family.format_version != AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1
                || family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER == 0
                || family.format_features & AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR as u64 == 0
            {
                continue;
            }
            let options = amdf_gpu_user_queue_create_info_t {
                type_: HRX_AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
                structure_size: size_of::<amdf_gpu_user_queue_create_info_t>() as u32,
                queue_family_ordinal: ordinal,
                priority: AMDF_QUEUE_PRIORITY_NORMAL,
                producer_mode: AMDF_QUEUE_PRODUCER_MODE_SINGLE,
                required_capabilities: AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER as u64,
                ring_byte_length: family.minimum_ring_byte_length,
                ..Default::default()
            };
            let mut raw = ptr::null_mut();
            check("gpu.user_queue_create", unsafe {
                entry!(api.gpu, user_queue_create)(self.0.raw, &options, &mut raw)
            })?;
            if raw.is_null() {
                return Err(missing("created queue"));
            }
            let mut queue = QueueInner {
                device: self.clone(),
                raw,
                mapping: ptr::null_mut(),
                info: amdf_user_queue_mapping_info_t::default(),
                transfers: Mutex::new(None),
                state: Mutex::new(QueueState {
                    write: 0,
                    pending: VecDeque::with_capacity(4096),
                }),
            };
            check("user_queue_map", unsafe {
                entry!(api.core, user_queue_map)(raw, ptr::null_mut(), &mut queue.mapping)
            })?;
            queue.info.type_ = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
            queue.info.structure_size = size_of::<amdf_user_queue_mapping_info_t>() as u32;
            check("user_queue_mapping_query_info", unsafe {
                entry!(api.core, user_queue_mapping_query_info)(queue.mapping, &mut queue.info)
            })?;
            let info = &queue.info;
            if info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                || info.format_version != 1
                || !info.ring_byte_length.is_power_of_two()
                || info.ring_byte_length < 4096
                || info.index_bits != 64
                || info.doorbell_bits != 64
                || [
                    info.ring_address,
                    info.read_index_address,
                    info.write_index_address,
                    info.doorbell_address,
                ]
                .iter()
                .any(|address| *address == 0 || address % 8 != 0)
            {
                return Err(Error::Unsupported(
                    "unexpected native PM4 mapping contract".into(),
                ));
            }
            return Ok(Queue(Arc::new(queue)));
        }
        Err(Error::Unsupported(
            "no qualified native PM4 queue family".into(),
        ))
    }
}

impl Queue {
    /// Prepare immutable commands and backing for repeated native execution.
    ///
    /// # Safety
    /// Arguments and launch dimensions must satisfy the kernel's memory
    /// contract. The caller orders conflicting accesses across other queues.
    pub unsafe fn prepare(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u16; 3],
        arguments: &[Argument<'_>],
    ) -> Result<PreparedGpu> {
        if !Arc::ptr_eq(&kernel.device().0, &self.0.device.0) {
            return Err(Error::Message(
                "kernel belongs to another native device".into(),
            ));
        }
        let fabric = Fabric(self.0.device.0.endpoint.0.instance.clone());
        let (packed, mut buffers) = kernel.arguments(arguments)?;
        let argument_buffer =
            fabric.allocate(packed.len().max(64), std::slice::from_ref(&self.0.device))?;
        argument_buffer.write(0, &packed)?;
        let fence = fabric.allocate(64, std::slice::from_ref(&self.0.device))?;
        let address = fence.device_address(&self.0.device)?;
        let scratch = if kernel.0.info.private_bytes != 0 {
            let mut info = amdf_gpu_endpoint_info_t {
                type_: HRX_AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
                structure_size: size_of::<amdf_gpu_endpoint_info_t>() as u32,
                ..Default::default()
            };
            let api = &self.0.device.0.endpoint.0.instance.api;
            check("GPU scratch topology", unsafe {
                entry!(api.gpu, endpoint_query_info)(self.0.device.0.endpoint.0.raw, &mut info)
            })?;
            if info.topology.xcc_count != 1 {
                return Err(Error::Unsupported(
                    "scratch requires the qualified single-XCC topology".into(),
                ));
            }
            let waves = info
                .compute
                .compute_unit_count
                .checked_mul(info.compute.maximum_scratch_wave_count_per_compute_unit)
                .ok_or_else(|| Error::Message("scratch topology overflow".into()))?;
            let per_wave = (u64::from(kernel.0.info.private_bytes)
                * u64::from(kernel.0.info.wave_size))
            .div_ceil(256)
                * 256;
            let size = per_wave
                .checked_mul(u64::from(waves))
                .filter(|size| *size <= 512 * 1024 * 1024)
                .ok_or_else(|| {
                    Error::Unsupported("kernel scratch exceeds 512 MiB admission limit".into())
                })?;
            let buffer = fabric.allocate_access(
                size as usize,
                std::slice::from_ref(&self.0.device),
                AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                256,
            )?;
            Some((buffer, waves, info.topology.shader_engine_count_per_xcc))
        } else {
            None
        };
        let mut words = Vec::with_capacity(160);
        barrier(&mut words);
        let dispatch_start = words.len();
        words.extend(
            kernel.dispatch_words(
                grid,
                block,
                &packed,
                argument_buffer.device_address(&self.0.device)?,
                scratch
                    .as_ref()
                    .map(|(buffer, waves, engines)| (buffer, *waves, *engines)),
            )?,
        );
        let dispatch_end = words.len();
        barrier(&mut words);
        let fence_word = words.len() + 4;
        words.extend([
            header(0x37, 5),
            (2 << 8) | (1 << 20),
            address as u32,
            (address >> 32) as u32,
            1,
        ]);
        pad(&mut words);
        buffers.push(argument_buffer.clone());
        if let Some((scratch, _, _)) = scratch {
            buffers.push(scratch);
        }
        buffers.sort_by_key(|buffer| Arc::as_ptr(&buffer.0) as usize);
        buffers.dedup_by(|a, b| Arc::ptr_eq(&a.0, &b.0));
        let leases = Mutex::new(Vec::with_capacity(buffers.len()));
        Ok(PreparedGpu {
            queue: self.clone(),
            inner: Arc::new(PreparedInner {
                work: Arc::new(Work {
                    _kernels: vec![kernel.clone()],
                    buffers,
                    fence,
                    leases,
                    retired: AtomicU32::new(0),
                    active: AtomicU32::new(0),
                    dependencies: Vec::new(),
                }),
                words,
                dispatch_range: dispatch_start..dispatch_end,
                fence_word,
                previous: Mutex::new(0),
            }),
        })
    }
    /// Combine prepared dispatches into one immutable indirect command buffer.
    /// A true flag establishes a system-memory dependency on preceding nodes.
    ///
    /// # Safety
    /// The supplied barriers must order every conflicting access. Commands must
    /// obey their kernel contracts and must not use this batch's storage elsewhere.
    pub unsafe fn prepare_batch(&self, commands: &[(PreparedGpu, bool)]) -> Result<PreparedGpu> {
        if commands.is_empty() {
            return Err(Error::Message("empty native command batch".into()));
        }
        let mut indirect = Vec::new();
        let mut buffers = Vec::new();
        let mut kernels = Vec::new();
        let mut dependencies = Vec::new();
        for (command, dependency) in commands {
            if !Arc::ptr_eq(&command.queue.0, &self.0) {
                return Err(Error::Message(
                    "batch command belongs to another queue".into(),
                ));
            }
            if *dependency {
                barrier(&mut indirect);
            }
            indirect.extend_from_slice(&command.inner.words[command.inner.dispatch_range.clone()]);
            buffers.extend(command.inner.work.buffers.iter().cloned());
            kernels.extend(command.inner.work._kernels.iter().cloned());
            dependencies.extend(command.inner.work.dependencies.iter().cloned());
        }
        pad(&mut indirect);
        if indirect.len() >= (1 << 20) {
            return Err(Error::Unsupported(
                "native batch exceeds indirect command capacity".into(),
            ));
        }
        let fabric = self.device().fabric();
        let storage = fabric.allocate_access(
            indirect.len() * 4,
            std::slice::from_ref(self.device()),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | AMDF_MEMORY_ACCESS_EXECUTE,
            256,
        )?;
        // Native PM4 words are little endian on this supported x86_64 host.
        storage.write(0, unsafe {
            std::slice::from_raw_parts(indirect.as_ptr().cast(), indirect.len() * 4)
        })?;
        let ib_address = storage.device_address(self.device())?;
        buffers.push(storage);
        buffers.sort_by_key(|buffer| Arc::as_ptr(&buffer.0) as usize);
        buffers.dedup_by(|a, b| Arc::ptr_eq(&a.0, &b.0));
        let fence = fabric.allocate(64, std::slice::from_ref(self.device()))?;
        let address = fence.device_address(self.device())?;
        let mut words = Vec::new();
        barrier(&mut words);
        let dispatch_start = words.len();
        words.extend([
            header(0x3f, 4),
            ib_address as u32,
            (ib_address >> 32) as u32,
            indirect.len() as u32 | (1 << 23),
        ]);
        let dispatch_end = words.len();
        barrier(&mut words);
        let fence_word = words.len() + 4;
        words.extend([
            header(0x37, 5),
            (2 << 8) | (1 << 20),
            address as u32,
            (address >> 32) as u32,
            1,
        ]);
        pad(&mut words);
        let leases = Mutex::new(Vec::with_capacity(buffers.len()));
        Ok(PreparedGpu {
            queue: self.clone(),
            inner: Arc::new(PreparedInner {
                work: Arc::new(Work {
                    fence,
                    _kernels: kernels,
                    buffers,
                    leases,
                    retired: AtomicU32::new(0),
                    active: AtomicU32::new(0),
                    dependencies,
                }),
                words,
                dispatch_range: dispatch_start..dispatch_end,
                fence_word,
                previous: Mutex::new(0),
            }),
        })
    }
    /// Prepare a device-side wait for an immutable completion point.
    /// The source fence and all indirect resources remain owned through retirement.
    pub fn prepare_wait(&self, completion: &Completion) -> Result<PreparedGpu> {
        if self.device().id() != completion.queue.device().id() {
            return Err(Error::Message("event belongs to another device".into()));
        }
        let address = completion.work.fence.device_address(self.device())?;
        let fence = self
            .device()
            .fabric()
            .allocate(64, std::slice::from_ref(self.device()))?;
        let output = fence.device_address(self.device())?;
        let mut words = Vec::with_capacity(40);
        // WAIT_REG_MEM32, memory space, unsigned >= comparison, MEC polling.
        words.extend([
            header(0x3c, 7),
            5 | (1 << 4),
            address as u32,
            (address >> 32) as u32,
            completion.value,
            u32::MAX,
            4,
        ]);
        barrier(&mut words);
        let dispatch_end = words.len();
        let fence_word = words.len() + 4;
        words.extend([
            header(0x37, 5),
            (2 << 8) | (1 << 20),
            output as u32,
            (output >> 32) as u32,
            1,
        ]);
        pad(&mut words);
        Ok(PreparedGpu {
            queue: self.clone(),
            inner: Arc::new(PreparedInner {
                work: Arc::new(Work {
                    fence,
                    _kernels: Vec::new(),
                    buffers: Vec::new(),
                    leases: Mutex::new(Vec::new()),
                    retired: AtomicU32::new(0),
                    active: AtomicU32::new(0),
                    dependencies: vec![(completion.work.clone(), completion.value)],
                }),
                words,
                dispatch_range: 0..dispatch_end,
                fence_word,
                previous: Mutex::new(0),
            }),
        })
    }
    /// Prepare and publish a single invocation.
    ///
    /// # Safety
    /// The kernel and its arguments must obey the declared memory contract;
    /// callers order conflicting accesses on other queues.
    pub unsafe fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u16; 3],
        arguments: &[Argument<'_>],
    ) -> Result<Completion> {
        unsafe { self.prepare(kernel, grid, block, arguments)?.dispatch() }
    }
}

/// Immutable bound GPU commands with reusable argument, scratch and fence storage.
/// Replays remain ordered on the originating queue, including scratch accesses.
#[derive(Clone)]
pub struct PreparedGpu {
    queue: Queue,
    inner: Arc<PreparedInner>,
}
struct PreparedInner {
    work: Arc<Work>,
    words: Vec<u32>,
    dispatch_range: std::ops::Range<usize>,
    fence_word: usize,
    previous: Mutex<u32>,
}
impl PreparedGpu {
    /// Submit previously prepared commands without creating native resources.
    ///
    /// # Safety
    /// Callers order conflicting accesses on other queues and obey the native
    /// executable's binding contracts.
    pub unsafe fn dispatch(&self) -> Result<Completion> {
        let mut previous = self
            .inner
            .previous
            .lock()
            .map_err(|_| Error::DeviceLost("prepared command poisoned".into()))?;
        let value = previous
            .checked_add(1)
            .ok_or_else(|| Error::DeviceLost("prepared completion timeline exhausted".into()))?;
        let words = &self.inner.words;
        let mut state = self
            .queue
            .0
            .state
            .lock()
            .map_err(|_| Error::DeviceLost("queue producer poisoned".into()))?;
        // Reap completed submissions even if their public completion was dropped.
        while let Some(work) = state.pending.front() {
            if !self.queue.0.poll(&work.work, work.value)? {
                break;
            }
            state.pending.pop_front();
        }
        if state.pending.len() >= 4096 {
            return Err(Error::Busy("native queue submission capacity".into()));
        }
        let capacity = self.queue.0.info.ring_byte_length / 4;
        let cursor = state.write & (capacity - 1);
        let wrap = if words.len() as u64 > capacity - cursor {
            capacity - cursor
        } else {
            0
        };
        let required = wrap + words.len() as u64;
        let read = unsafe {
            (&*(self.queue.0.info.read_index_address as *const AtomicU64)).load(Ordering::Acquire)
        };
        let outstanding = state.write.wrapping_sub(read) & (capacity - 1);
        if required >= capacity - outstanding {
            return Err(Error::Busy("native command ring is full".into()));
        }
        let published = state
            .write
            .checked_add(required)
            .ok_or_else(|| Error::DeviceLost("native queue index exhausted".into()))?;
        let work = self.inner.work.clone();
        let mut leases = work
            .leases
            .lock()
            .map_err(|_| Error::DeviceLost("submission ownership poisoned".into()))?;
        if leases.is_empty() {
            for buffer in &work.buffers {
                match buffer.retain_use() {
                    Ok(lease) => leases.push(lease),
                    Err(error) => {
                        leases.clear();
                        return Err(error);
                    }
                }
            }
        }
        work.active.store(value, Ordering::Release);
        // Fixed capacity and reusable ownership storage are allocated at prepare.
        state.pending.push_back(Pending {
            work: work.clone(),
            value,
        });
        drop(leases);
        unsafe {
            let ring = self.queue.0.info.ring_address as *mut u32;
            if wrap != 0 {
                ptr::write(ring.add(cursor as usize), header(0x10, wrap as usize));
                ptr::write_bytes(ring.add(cursor as usize + 1), 0, wrap as usize - 1);
            }
            let cursor = if wrap == 0 { cursor as usize } else { 0 };
            ptr::copy_nonoverlapping(words.as_ptr(), ring.add(cursor), words.len());
            ptr::write(ring.add(cursor + self.inner.fence_word), value);
            (&*(self.queue.0.info.write_index_address as *const AtomicU64))
                .store(published, Ordering::Release);
            std::sync::atomic::fence(Ordering::SeqCst);
            ptr::write_volatile(self.queue.0.info.doorbell_address as *mut u64, published);
        }
        state.write = published;
        *previous = value;
        Ok(Completion {
            queue: self.queue.clone(),
            work,
            value,
        })
    }
}

impl Queue {
    /// Device whose address domain this queue executes in.
    pub fn device(&self) -> &Device {
        &self.0.device
    }
    fn transfer_kernels(&self) -> Result<[Kernel; 4]> {
        let mut kernels = self
            .0
            .transfers
            .lock()
            .map_err(|_| Error::DeviceLost("transfer compiler poisoned".into()))?;
        if kernels.is_none() {
            let compiler = crate::loom::Compiler::for_target(None, self.device().target())?;
            let source =
                crate::loom::CxxSource::new("hrx-transfers.cpp", include_str!("transfers.cpp"));
            let module = compiler.import_cxx(source)?;
            // Compile an upper bound for blockIdx; these kernels never read
            // gridDim and guard every byte with the runtime length. Dispatching
            // the checked prefix below therefore preserves their memory contract.
            let request = |symbol: &str| {
                crate::loom::Specialization::new(symbol)
                    .with_config(format!("{symbol}.workgroup_count.x"), "16777216")
                    .with_config(format!("{symbol}.workgroup_count.y"), "1")
                    .with_config(format!("{symbol}.workgroup_count.z"), "1")
            };
            let mut entries = Vec::new();
            for symbol in ["hrx_copy", "hrx_fill", "hrx_copy_words", "hrx_fill_words"] {
                let artifact = module.compile(&request(symbol))?;
                // Crate-owned bounded kernels; prepared calls validate ranges and alignment.
                entries.push(unsafe { self.device().load(&artifact) }?);
            }
            *kernels = Some(entries.try_into().unwrap_or_else(|_| unreachable!()));
        }
        Ok(kernels.as_ref().unwrap().clone())
    }
    /// Prepare a byte copy between non-overlapping logical ranges.
    pub fn prepare_copy(
        &self,
        destination: &Buffer,
        destination_offset: usize,
        source: &Buffer,
        source_offset: usize,
        length: usize,
    ) -> Result<PreparedGpu> {
        transfer_range(destination, destination_offset, length)?;
        transfer_range(source, source_offset, length)?;
        if Arc::ptr_eq(&destination.0, &source.0)
            && destination_offset < source_offset + length
            && source_offset < destination_offset + length
        {
            return Err(Error::Message("native copy ranges overlap".into()));
        }
        let kernels = self.transfer_kernels()?;
        let wide = (destination_offset | source_offset | length) & 7 == 0;
        let copy = &kernels[if wide { 2 } else { 0 }];
        let elements = if wide { length / 8 } else { length };
        let count = (elements as u64).to_le_bytes();
        unsafe {
            self.prepare(
                copy,
                [elements.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                &[
                    Argument::Buffer(destination, destination_offset),
                    Argument::Buffer(source, source_offset),
                    Argument::Value(&count),
                ],
            )
        }
    }
    /// Prepare a repeated byte fill over one logical range.
    pub fn prepare_fill(
        &self,
        destination: &Buffer,
        offset: usize,
        length: usize,
        value: u8,
    ) -> Result<PreparedGpu> {
        transfer_range(destination, offset, length)?;
        let kernels = self.transfer_kernels()?;
        let wide = (offset | length) & 7 == 0;
        let fill = &kernels[if wide { 3 } else { 1 }];
        let elements = if wide { length / 8 } else { length };
        let count = (elements as u64).to_le_bytes();
        let value = u32::from(value).to_le_bytes();
        unsafe {
            self.prepare(
                fill,
                [elements.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                &[
                    Argument::Buffer(destination, offset),
                    Argument::Value(&count),
                    Argument::Value(&value),
                ],
            )
        }
    }
}
fn transfer_range(buffer: &Buffer, offset: usize, length: usize) -> Result<()> {
    if length == 0
        || length > u32::MAX as usize
        || offset
            .checked_add(length)
            .is_none_or(|end| end > buffer.len())
    {
        return Err(Error::Message(
            "native transfer range must be nonempty, in bounds, and at most 4 GiB".into(),
        ));
    }
    Ok(())
}
