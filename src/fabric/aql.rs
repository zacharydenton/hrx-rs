//! Owned AQL v1 dispatch using libamdf's native packet and signal contracts.
use super::*;
use std::{
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

/// A serialized AQL producer with fixed scratch capacity.
#[derive(Clone)]
pub struct AqlQueue(Arc<QueueState>);
struct QueueState {
    device: Device,
    raw: *mut amdf_user_queue_t,
    mapping: *mut amdf_user_queue_mapping_t,
    info: amdf_user_queue_mapping_info_t,
    scratch: Option<Buffer>,
    maximum_private_bytes: u32,
    budget: Option<crate::residency::MemoryBudget>,
    producer: Mutex<Producer>,
}
struct Producer {
    write: u64,
    pending: VecDeque<(Arc<Work>, u64)>,
}
// One host publisher owns the ring under producer. Mapping and scratch remain
// alive through queue destruction; completion observers only read owned signals.
unsafe impl Send for QueueState {}
unsafe impl Sync for QueueState {}
struct Work {
    signal: Buffer,
    buffers: Vec<Buffer>,
    _kernel: Kernel,
    leases: Mutex<Vec<memory::DeviceUse>>,
    active: AtomicU64,
    retired: AtomicU64,
}
impl Work {
    fn refresh(&self, point: u64) -> Result<bool> {
        if self.retired.load(Ordering::Acquire) >= point {
            return Ok(true);
        }
        let value = unsafe {
            (&*self.signal.host_pointer().add(8).cast::<AtomicI64>()).load(Ordering::Acquire)
        };
        let completed = i64::MAX
            .checked_sub(value)
            .filter(|v| *v >= 0)
            .ok_or_else(|| Error::DeviceLost("invalid AQL completion signal".into()))?
            as u64;
        if completed < point {
            return Ok(false);
        }
        let mut leases = self
            .leases
            .lock()
            .map_err(|_| Error::DeviceLost("AQL leases poisoned".into()))?;
        if completed >= self.active.load(Ordering::Acquire) {
            leases.clear();
        }
        self.retired.fetch_max(completed, Ordering::Release);
        Ok(true)
    }
}
impl Drop for QueueState {
    fn drop(&mut self) {
        let pending = std::mem::take(
            &mut self
                .producer
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .pending,
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        for (work, point) in &pending {
            loop {
                match work.refresh(*point) {
                    Ok(true) => break,
                    Ok(false) if Instant::now() < deadline => std::thread::yield_now(),
                    _ => {
                        std::mem::forget(pending);
                        std::mem::forget(self.device.clone());
                        std::mem::forget(self.scratch.take());
                        return;
                    }
                }
            }
        }
        let api = self.device.0.endpoint.0.instance.api.core;
        unsafe {
            if (!self.mapping.is_null()
                && api.user_queue_mapping_destroy.unwrap()(self.mapping) != 0)
                || api.user_queue_destroy.unwrap()(self.raw) != 0
            {
                std::mem::forget(self.device.clone());
                std::mem::forget(self.scratch.take());
            }
        }
    }
}
/// Immutable AQL packets and bindings for repeated ordered dispatch.
#[derive(Clone)]
pub struct PreparedAql {
    queue: AqlQueue,
    work: Arc<Work>,
    packets: Vec<[u32; 16]>,
    next: Arc<Mutex<u64>>,
}
/// A native AQL execution completion retaining code, arguments and payloads.
#[derive(Clone)]
pub struct AqlCompletion {
    queue: AqlQueue,
    work: Arc<Work>,
    point: u64,
}
impl AqlCompletion {
    /// Cached retirement only; no native query, lock or progress work.
    pub fn is_complete(&self) -> bool {
        self.work.retired.load(Ordering::Acquire) >= self.point
    }
    /// Observe native execution completion and release completed leases.
    pub fn refresh(&self) -> Result<bool> {
        if !self.work.refresh(self.point)? {
            let api = self.queue.0.device.0.endpoint.0.instance.api.core;
            let mut info = amdf_user_queue_status_t {
                type_: AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
                structure_size: size_of::<amdf_user_queue_status_t>() as u32,
                ..Default::default()
            };
            check("AQL queue status", unsafe {
                entry!(api, user_queue_query_status)(self.queue.0.raw, &mut info)
            })?;
            if info.state != AMDF_QUEUE_STATE_ACTIVE {
                return Err(Error::DeviceLost(format!(
                    "AQL queue status {}",
                    info.terminal_status
                )));
            }
            return Ok(false);
        }
        let mut producer = self
            .queue
            .0
            .producer
            .lock()
            .map_err(|_| Error::DeviceLost("AQL producer poisoned".into()))?;
        while let Some((work, point)) = producer.pending.front() {
            if !work.refresh(*point)? {
                break;
            }
            producer.pending.pop_front();
        }
        Ok(true)
    }
    /// Wait under a host deadline. Timeout does not cancel or release work.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let start = Instant::now();
        loop {
            if self.refresh()? {
                return Ok(true);
            }
            if start.elapsed() >= timeout {
                return Ok(false);
            }
            std::thread::yield_now();
        }
    }
    /// Wait for execution completion, preserving resources on failure.
    pub fn wait(&self) -> Result<()> {
        while !self.refresh()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}
impl Device {
    /// Create an AQL queue with fixed per-workitem scratch capacity.
    /// Zero admits kernels with no private segment. All physical scratch slots
    /// are allocated before returning; dispatch never grows the pool.
    pub fn aql_queue(&self, maximum_private_bytes: u32) -> Result<AqlQueue> {
        self.aql_queue_budgeted(maximum_private_bytes, None)
    }
    /// Create a queue whose scratch and prepared dispatch storage share a byte ceiling.
    pub fn aql_queue_budgeted(
        &self,
        maximum_private_bytes: u32,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<AqlQueue> {
        if self.endpoint().engine() != Engine::Gpu {
            return Err(Error::Unsupported("AQL requires a GPU".into()));
        }
        let family = self
            .endpoint()
            .queue_capabilities()?
            .into_iter()
            .find(|f| {
                f.command == QueueCommand::Aql
                    && f.format_version == 1
                    && f.user_publication
                    && f.system_release
                    && f.system_acquire
            })
            .ok_or_else(|| Error::Unsupported("no native AQL v1 family".into()))?;
        let api = &self.0.endpoint.0.instance.api;
        let _ = entry!(api.core, user_queue_destroy);
        let _ = entry!(api.core, user_queue_mapping_destroy);
        let mut scratch_info = amdf_gpu_queue_scratch_t::default();
        let scratch = if maximum_private_bytes != 0 {
            let mut endpoint = amdf_gpu_endpoint_info_t {
                type_: HRX_AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
                structure_size: size_of::<amdf_gpu_endpoint_info_t>() as u32,
                ..Default::default()
            };
            check("AQL scratch topology", unsafe {
                entry!(api.gpu, endpoint_query_info)(self.0.endpoint.0.raw, &mut endpoint)
            })?;
            let engines = u64::from(endpoint.topology.xcc_count)
                * u64::from(endpoint.topology.shader_engine_count_per_xcc);
            if engines == 0 {
                return Err(Error::Unsupported("empty AQL scratch topology".into()));
            }
            let slots = u64::from(endpoint.compute.compute_unit_count).div_ceil(engines)
                * engines
                * u64::from(endpoint.compute.maximum_scratch_wave_count_per_compute_unit);
            let bytes = (u64::from(maximum_private_bytes) * 64)
                .next_multiple_of(1024)
                .checked_mul(slots)
                .filter(|bytes| {
                    *bytes != 0 && *bytes <= 512 * 1024 * 1024 && slots <= u64::from(u32::MAX)
                })
                .ok_or_else(|| {
                    Error::Unsupported("AQL scratch exceeds the 512 MiB admission limit".into())
                })?;
            let buffer = self.fabric().allocate_owned(
                bytes as usize,
                self,
                AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                4096,
                false,
                budget,
            )?;
            scratch_info = amdf_gpu_queue_scratch_t {
                memory: buffer.0.raw,
                byte_length: bytes,
                maximum_private_segment_byte_length: maximum_private_bytes,
                maximum_wave_count: slots as u32,
                ..Default::default()
            };
            Some(buffer)
        } else {
            None
        };
        let options = amdf_gpu_user_queue_create_info_t {
            type_: HRX_AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
            structure_size: size_of::<amdf_gpu_user_queue_create_info_t>() as u32,
            queue_family_ordinal: family.ordinal,
            priority: AMDF_QUEUE_PRIORITY_NORMAL,
            producer_mode: AMDF_QUEUE_PRODUCER_MODE_SINGLE,
            required_capabilities: AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER as u64,
            scratch: scratch_info,
            ..Default::default()
        };
        let mut raw = ptr::null_mut();
        check("AQL queue create", unsafe {
            entry!(api.gpu, user_queue_create)(self.0.raw, &options, &mut raw)
        })?;
        if raw.is_null() {
            return Err(missing("AQL queue"));
        }
        let mut queue = QueueState {
            device: self.clone(),
            raw,
            mapping: ptr::null_mut(),
            info: Default::default(),
            scratch,
            maximum_private_bytes,
            budget: budget.cloned(),
            producer: Mutex::new(Producer {
                write: 0,
                pending: VecDeque::with_capacity(4096),
            }),
        };
        check("AQL queue map", unsafe {
            entry!(api.core, user_queue_map)(raw, ptr::null_mut(), &mut queue.mapping)
        })?;
        queue.info.type_ = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
        queue.info.structure_size = size_of::<amdf_user_queue_mapping_info_t>() as u32;
        check("AQL mapping info", unsafe {
            entry!(api.core, user_queue_mapping_query_info)(queue.mapping, &mut queue.info)
        })?;
        let info = &queue.info;
        if info.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            || info.format_version != 1
            || info.index_bits != 64
            || info.doorbell_bits != 64
            || !info.ring_byte_length.is_power_of_two()
            || info.ring_byte_length < 128
            || [
                info.ring_address,
                info.read_index_address,
                info.write_index_address,
                info.doorbell_address,
            ]
            .iter()
            .any(|address| *address == 0 || address % 8 != 0)
        {
            return Err(Error::Unsupported("unexpected AQL mapping contract".into()));
        }
        Ok(AqlQueue(Arc::new(queue)))
    }
}
impl AqlQueue {
    /// Device owning this queue and its fixed scratch.
    pub fn device(&self) -> &Device {
        &self.0.device
    }
    /// Prepare a dispatch and its instruction-cache publication packets.
    /// # Safety
    /// Arguments and geometry must satisfy the trusted kernel's memory contract.
    /// Caller orders conflicting accesses by other queues.
    pub unsafe fn prepare(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u16; 3],
        arguments: &[Argument<'_>],
    ) -> Result<PreparedAql> {
        if !Arc::ptr_eq(&kernel.device().0, &self.device().0) {
            return Err(Error::Message(
                "AQL kernel belongs to another device".into(),
            ));
        }
        if kernel.0.info.private_bytes > self.0.maximum_private_bytes {
            return Err(Error::Unsupported(
                "kernel exceeds fixed AQL scratch capacity".into(),
            ));
        }
        let mut extents = [0; 3];
        for axis in 0..3 {
            if block[axis] == 0
                || grid[axis] == 0
                || (kernel.workgroup_size()[axis] != 0
                    && kernel.workgroup_size()[axis] != u32::from(block[axis]))
            {
                return Err(Error::Message(
                    "AQL launch geometry does not match kernel".into(),
                ));
            }
            extents[axis] = grid[axis]
                .checked_mul(u32::from(block[axis]))
                .ok_or_else(|| Error::Message("AQL grid extent overflow".into()))?;
        }
        if block.iter().map(|v| u64::from(*v)).product::<u64>() > 1024 {
            return Err(Error::Message(
                "AQL workgroup exceeds gfx1151 capacity".into(),
            ));
        }
        let specialized = kernel.for_aql(
            grid,
            self.0.info.ring_address,
            self.0.info.ring_byte_length / 64 - 1,
        )?;
        let clear = if let Some(state) = &specialized.0.race_state {
            let (utility, clear_grid) = state.clear_kernel(self.device(), kernel.race_budget())?;
            // Utility is uninstrumented, so preparation does not recurse again.
            Some(unsafe {
                self.prepare(
                    &utility,
                    clear_grid,
                    [64, 1, 1],
                    &[Argument::Buffer(&state.shadow, 0)],
                )
            }?)
        } else {
            None
        };
        let kernel = &specialized;
        let (packed, mut buffers) = kernel.arguments(arguments)?;
        let fabric = self.device().fabric();
        let kernarg = fabric.allocate_owned(
            packed.len().max(64),
            self.device(),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            true,
            self.0.budget.as_ref(),
        )?;
        kernarg.write(0, &packed)?;
        let kernarg_address = kernarg.device_address(self.device())?;
        let descriptor = kernel
            .0
            .code
            .device_address(self.device())?
            .checked_add(kernel.0.info.descriptor_offset)
            .ok_or_else(|| Error::Message("AQL descriptor address overflow".into()))?;
        // GFX11 GCR includes GLI_INV_ALL. The native system fences on dispatch
        // order data; this explicit publication also handles reused code addresses.
        let invalidation = [
            0xc0065800u32,
            0,
            u32::MAX,
            0xff,
            0,
            0,
            0x0a,
            1 | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15),
        ];
        let publication = fabric.allocate_owned(
            4096,
            self.device(),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
            4096,
            false,
            self.0.budget.as_ref(),
        )?;
        publication.write(
            0,
            &invalidation
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let ib = publication.device_address(self.device())?;
        if ib >= (1 << 48) || ib + 32 > (1 << 48) {
            return Err(Error::Unsupported(
                "AQL publication IB is outside the 48-bit address domain".into(),
            ));
        }
        let signal = fabric.allocate_owned(
            64,
            self.device(),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            true,
            self.0.budget.as_ref(),
        )?;
        signal.write(0, &1i64.to_le_bytes())?;
        signal.write(8, &i64::MAX.to_le_bytes())?;
        let signal_address = signal.device_address(self.device())?;
        let mut publish = [0; 16];
        publish[0] = (1 << 16) | (1 << 8);
        publish[1] = 0xc0023f00;
        publish[2] = ib as u32;
        publish[3] = (ib >> 32) as u32;
        publish[4] = (1 << 23) | 8;
        publish[5] = 10;
        let mut dispatch = [0; 16];
        dispatch[0] = 2 | (1 << 8) | (2 << 9) | (2 << 11) | (3 << 16);
        dispatch[1] = u32::from(block[0]) | (u32::from(block[1]) << 16);
        dispatch[2] = u32::from(block[2]);
        dispatch[3..6].copy_from_slice(&extents);
        dispatch[6] = kernel.0.info.private_bytes;
        dispatch[7] = kernel.0.info.local_bytes;
        dispatch[8] = descriptor as u32;
        dispatch[9] = (descriptor >> 32) as u32;
        dispatch[10] = kernarg_address as u32;
        dispatch[11] = (kernarg_address >> 32) as u32;
        dispatch[14] = signal_address as u32;
        dispatch[15] = (signal_address >> 32) as u32;
        let mut packets = vec![publish];
        if let Some(clear) = clear {
            let mut packet = *clear.packets.last().unwrap();
            packet[14..16].fill(0);
            packets.push(packet);
            // The outer completion covers the clear kernel and all of its
            // bindings. The private utility completion signal is never submitted.
            buffers.extend(clear.work.buffers.iter().cloned());
        }
        packets.push(dispatch);
        buffers.extend([kernarg, publication]);
        buffers.sort_by_key(Buffer::identity);
        buffers.dedup_by(|a, b| a.same_backing(b));
        let leases = Mutex::new(Vec::with_capacity(buffers.len()));
        Ok(PreparedAql {
            queue: self.clone(),
            work: Arc::new(Work {
                signal,
                buffers,
                _kernel: kernel.clone(),
                leases,
                active: AtomicU64::new(0),
                retired: AtomicU64::new(0),
            }),
            packets,
            next: Arc::new(Mutex::new(0)),
        })
    }
}
impl PreparedAql {
    /// Publish prepared packets without allocating or growing scratch.
    /// # Safety
    /// Caller orders conflicting accesses on other queues and obeys the kernel contract.
    pub unsafe fn dispatch(&self) -> Result<AqlCompletion> {
        let mut next = self
            .next
            .lock()
            .map_err(|_| Error::DeviceLost("AQL dispatch poisoned".into()))?;
        let point = next
            .checked_add(1)
            .filter(|point| *point <= i64::MAX as u64)
            .ok_or_else(|| Error::DeviceLost("AQL timeline exhausted".into()))?;
        let mut producer = self
            .queue
            .0
            .producer
            .lock()
            .map_err(|_| Error::DeviceLost("AQL producer poisoned".into()))?;
        while let Some((work, point)) = producer.pending.front() {
            if !work.refresh(*point)? {
                break;
            }
            producer.pending.pop_front();
        }
        let info = &self.queue.0.info;
        let capacity = info.ring_byte_length / 64;
        let read =
            unsafe { (&*(info.read_index_address as *const AtomicU64)).load(Ordering::Acquire) };
        let end = producer
            .write
            .checked_add(self.packets.len() as u64)
            .ok_or_else(|| Error::DeviceLost("AQL ring index exhausted".into()))?;
        if producer.pending.len() == 4096 || end.saturating_sub(read) > capacity {
            return Err(Error::Busy("AQL ring or pending capacity exhausted".into()));
        }
        let mut leases = self
            .work
            .leases
            .lock()
            .map_err(|_| Error::DeviceLost("AQL leases poisoned".into()))?;
        if leases.is_empty() {
            for buffer in &self.work.buffers {
                match buffer.retain_use() {
                    Ok(lease) => leases.push(lease),
                    Err(error) => {
                        leases.clear();
                        return Err(error);
                    }
                }
            }
        }
        self.work.active.store(point, Ordering::Release);
        producer.pending.push_back((self.work.clone(), point));
        unsafe {
            (&*(info.write_index_address as *const AtomicU64)).store(end, Ordering::Release);
            for (offset, packet) in self.packets.iter().enumerate() {
                let index = producer.write + offset as u64;
                let slot =
                    (info.ring_address as *mut u32).add(((index & (capacity - 1)) * 16) as usize);
                ptr::copy_nonoverlapping(packet.as_ptr().add(1), slot.add(1), 15);
                (&*slot.cast::<AtomicU32>()).store(packet[0], Ordering::Release);
                std::sync::atomic::fence(Ordering::SeqCst);
                ptr::write_volatile(info.doorbell_address as *mut u64, index);
            }
        }
        producer.write = end;
        *next = point;
        Ok(AqlCompletion {
            queue: self.queue.clone(),
            work: self.work.clone(),
            point,
        })
    }
}
