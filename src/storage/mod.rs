//! Bounded GPU-authored file I/O over registered system memory.
//!
//! This is not a peer-to-peer VRAM or durable-checkpoint API. Files, native
//! commands and payload slots remain owned through I/O and downstream copies.
use crate::{Error, Result, fabric};
pub use fabric::StorageProgress;
use std::{
    collections::VecDeque,
    fs::File,
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

/// Explicit filesystem I/O policy. Unsupported direct I/O never becomes buffered I/O.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StorageMode {
    /// Ordinary filesystem I/O, including the page cache.
    #[default]
    Buffered,
    /// Filesystem-aligned direct I/O into registered system pages.
    Direct,
}
/// Capacity fixed before native resources or a service thread are created.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    /// Filesystem route.
    pub mode: StorageMode,
    /// Native kernel progress service.
    pub progress: StorageProgress,
    /// Number of simultaneously retained payloads (1 through 64).
    pub slots: usize,
    /// Maximum physical I/O bytes per slot, including alignment padding.
    pub slot_bytes: usize,
    /// Collect host-observed timing and service counters.
    pub statistics: bool,
}
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            mode: StorageMode::Buffered,
            progress: StorageProgress::Sqpoll,
            slots: 4,
            slot_bytes: 16 << 20,
            statistics: false,
        }
    }
}
/// Measured session counters. Durations use host clocks, not GPU timestamps.
#[derive(Clone, Debug, Default)]
pub struct StorageStatistics {
    /// Accepted logical requests, including shared reads.
    pub logical_requests: u64,
    /// Native requests submitted to the kernel.
    pub physical_requests: u64,
    /// Logical reads sharing an existing retained read.
    pub shared_reads: u64,
    /// Bytes from successful requests, including alignment padding.
    pub completed_bytes: u64,
    /// Distinct requests ending in an error; shared tickets count once.
    pub errors: u64,
    /// Maximum simultaneously retained slots.
    pub peak_slots: usize,
    /// Largest submitted batch (not NVMe hardware queue depth).
    pub peak_outstanding: usize,
    /// Accumulated queue delay before GPU publication, when enabled.
    pub admission_time: Option<Duration>,
    /// Accumulated submit-to-observed-completion time, when enabled.
    pub ready_time: Option<Duration>,
    /// Service thread CPU time, when supported and enabled.
    pub service_cpu_time: Option<Duration>,
    /// Kernel enter calls: SQPOLL wakes or deferred-work service.
    pub service_calls: u64,
    /// Bounded eventfd waits.
    pub wait_calls: u64,
}
struct RegisteredFile {
    file: File,
    bytes: AtomicU64,
    alignment: usize,
    writable: bool,
    readable: bool,
    identity: (u64, u64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    file: usize,
    offset: u64,
    bytes: usize,
    writing: bool,
}
#[derive(Clone, Copy, Debug)]
struct Physical {
    offset: u64,
    bytes: usize,
    skip: usize,
}
fn physical(extent: Extent, size: u64, alignment: usize, capacity: usize) -> Result<Physical> {
    let end = extent
        .offset
        .checked_add(extent.bytes as u64)
        .ok_or_else(|| Error::Message("file extent overflow".into()))?;
    if extent.bytes == 0 || extent.bytes > capacity || (!extent.writing && end > size) {
        return Err(Error::Message(
            "file extent exceeds its file or payload slot".into(),
        ));
    }
    let mask = alignment as u64 - 1;
    if extent.writing && (extent.offset & mask != 0 || extent.bytes & (alignment - 1) != 0) {
        return Err(Error::Message(
            "direct writes require aligned file offsets and lengths".into(),
        ));
    }
    let offset = extent.offset & !mask;
    let end = end
        .checked_add(mask)
        .ok_or_else(|| Error::Message("aligned extent overflow".into()))?
        & !mask;
    let bytes = usize::try_from(end - offset)
        .map_err(|_| Error::Message("file extent too large".into()))?;
    if bytes > capacity {
        return Err(Error::Message("aligned read exceeds payload slot".into()));
    }
    Ok(Physical {
        offset,
        bytes,
        skip: (extent.offset - offset) as usize,
    })
}
fn register(file: &File, mode: StorageMode) -> Result<RegisteredFile> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(Error::Unsupported("storage requires a regular file".into()));
    }
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let access = flags & libc::O_ACCMODE;
    if flags & libc::O_APPEND != 0 {
        return Err(Error::Unsupported(
            "storage does not accept append-only descriptors".into(),
        ));
    }
    let file = if mode == StorageMode::Direct || flags & libc::O_DIRECT != 0 {
        let opened = std::fs::OpenOptions::new()
            .read(access != libc::O_WRONLY)
            .write(access != libc::O_RDONLY)
            .custom_flags(
                libc::O_CLOEXEC
                    | (flags & (libc::O_SYNC | libc::O_DSYNC))
                    | if mode == StorageMode::Direct {
                        libc::O_DIRECT
                    } else {
                        0
                    },
            )
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
        let actual = opened.metadata()?;
        if (actual.dev(), actual.ino()) != (metadata.dev(), metadata.ino()) {
            return Err(Error::Message("storage file identity changed".into()));
        }
        opened
    } else {
        file.try_clone()?
    };
    let alignment = if mode == StorageMode::Direct {
        let mut info: libc::statx = unsafe { std::mem::zeroed() };
        let result = unsafe {
            libc::statx(
                file.as_raw_fd(),
                c"".as_ptr(),
                libc::AT_EMPTY_PATH,
                libc::STATX_DIOALIGN,
                &mut info,
            )
        };
        if result != 0
            || info.stx_mask & libc::STATX_DIOALIGN == 0
            || info.stx_dio_mem_align == 0
            || info.stx_dio_offset_align == 0
        {
            return Err(Error::Unsupported(
                "filesystem does not report direct-I/O alignment".into(),
            ));
        }
        let alignment = info.stx_dio_mem_align.max(info.stx_dio_offset_align) as usize;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        if !info.stx_dio_mem_align.is_power_of_two()
            || !info.stx_dio_offset_align.is_power_of_two()
            || alignment > page
        {
            return Err(Error::Unsupported(
                "direct-I/O alignment exceeds registered-page alignment".into(),
            ));
        }
        alignment
    } else {
        1
    };
    Ok(RegisteredFile {
        file,
        bytes: AtomicU64::new(metadata.len()),
        alignment,
        writable: access != libc::O_RDONLY,
        readable: access != libc::O_WRONLY && flags & libc::O_PATH == 0,
        identity: (metadata.dev(), metadata.ino()),
    })
}
type Outcome = std::result::Result<usize, Arc<Error>>;
struct Request {
    shared: Arc<Shared>,
    extent: Extent,
    physical: Physical,
    slot: usize,
    submitted: Instant,
    outcome: Mutex<Option<Outcome>>,
    ready: Condvar,
}
struct State {
    pending: VecDeque<Arc<Request>>,
    slots: Vec<Weak<Request>>,
    stop: bool,
    failed: Option<Arc<Error>>,
    statistics: StorageStatistics,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    payload: fabric::Buffer,
    files: Vec<RegisteredFile>,
    config: StorageConfig,
    device: fabric::Device,
}
fn stop_after_failure(shared: &Shared, error: Error) -> Error {
    let error = Arc::new(error);
    let mut state = shared.state.lock().unwrap();
    state.failed.get_or_insert_with(|| error.clone());
    state.stop = true;
    for request in &state.pending {
        *request.outcome.lock().unwrap() = Some(Err(error.clone()));
        request.ready.notify_all();
    }
    state.statistics.errors += state.pending.len() as u64;
    state.pending.clear();
    shared.changed.notify_all();
    Error::Execution { source: error }
}
/// Reusable storage queue with bounded payload ownership and one service thread.
/// Opening a session requires a stream in the process-lifetime native domain.
pub struct StorageSession {
    shared: Arc<Shared>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl StorageSession {
    /// Register files and allocate all payload slots before accepting requests.
    pub fn new(stream: &crate::Stream, files: &[File], config: StorageConfig) -> Result<Self> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(Error::Unsupported("host page size unavailable".into()));
        }
        if config.slots == 0
            || config.slots > 64
            || config.slot_bytes == 0
            || config.slot_bytes > 64 << 20
            || !config.slot_bytes.is_multiple_of(page as usize)
        {
            return Err(Error::Message(
                "storage needs 1..=64 slots of page-aligned size, at most 64 MiB each".into(),
            ));
        }
        let files = files
            .iter()
            .map(|f| register(f, config.mode))
            .collect::<Result<Vec<_>>>()?;
        if files.is_empty() || files.len() > 1024 {
            return Err(Error::Message("storage needs 1..=1024 files".into()));
        }
        let device = stream.native_device().clone();
        let payload = device.fabric().allocate_registered(
            config
                .slot_bytes
                .checked_mul(config.slots)
                .ok_or_else(|| Error::Message("storage capacity overflow".into()))?,
            std::slice::from_ref(&device),
            stream.memory_budget(),
        )?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                pending: VecDeque::with_capacity(config.slots),
                slots: (0..config.slots).map(|_| Weak::new()).collect(),
                stop: false,
                failed: None,
                statistics: StorageStatistics::default(),
            }),
            changed: Condvar::new(),
            payload,
            files,
            config,
            device,
        });
        let budget = stream.memory_budget().cloned();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let context = shared.clone();
        let worker = std::thread::Builder::new()
            .name("hrx-storage".into())
            .spawn(move || {
                let prepared = Worker::new(&context, budget);
                match prepared {
                    Ok(mut worker) => {
                        let _ = ready_tx.send(Ok(()));
                        worker.run(context);
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            })?;
        match ready_rx
            .recv()
            .map_err(|_| Error::DeviceLost("storage setup worker stopped".into()))?
        {
            Ok(()) => Ok(Self {
                shared,
                worker: Some(worker),
            }),
            Err(error) => {
                let _ = worker.join();
                Err(error)
            }
        }
    }
    /// Maximum logical read bytes guaranteed to fit despite direct-I/O padding.
    pub fn read_chunk_bytes(&self) -> usize {
        self.shared.config.slot_bytes.saturating_sub(
            self.shared
                .files
                .iter()
                .map(|f| f.alignment)
                .max()
                .unwrap_or(1)
                - 1,
        )
    }
    /// Read an exact immutable extent. Identical retained reads share one slot.
    /// Files must not be written through another handle while a shared read lives.
    pub fn read(&self, file: usize, offset: u64, bytes: usize) -> Result<ReadTicket> {
        Ok(ReadTicket {
            request: self.submit(
                Extent {
                    file,
                    offset,
                    bytes,
                    writing: false,
                },
                None,
            )?,
        })
    }
    /// Copy host bytes into an owned slot and submit a positioned write.
    /// Completion reports transferred bytes, not power-loss durability.
    /// Returns Busy when another slot has active device use: host writes obey
    /// the registered allocation's whole-buffer access exclusion.
    pub fn write(&self, file: usize, offset: u64, bytes: &[u8]) -> Result<ReadTicket> {
        Ok(ReadTicket {
            request: self.submit(
                Extent {
                    file,
                    offset,
                    bytes: bytes.len(),
                    writing: true,
                },
                Some(bytes),
            )?,
        })
    }
    fn submit(&self, extent: Extent, data: Option<&[u8]>) -> Result<Arc<Request>> {
        let file = self
            .shared
            .files
            .get(extent.file)
            .ok_or_else(|| Error::Message("unregistered file index".into()))?;
        if (extent.writing && !file.writable) || (!extent.writing && !file.readable) {
            return Err(std::io::Error::from_raw_os_error(libc::EBADF).into());
        }
        let physical = physical(
            extent,
            file.bytes.load(Ordering::Acquire),
            file.alignment,
            self.shared.config.slot_bytes,
        )?;
        let mut state = self.shared.state.lock().unwrap();
        if let Some(error) = &state.failed {
            return Err(Error::Execution {
                source: error.clone(),
            });
        }
        if state.stop {
            return Err(Error::Cancelled);
        }
        for request in state.slots.iter().filter_map(Weak::upgrade) {
            if self.shared.files[request.extent.file].identity == file.identity {
                let other = request.extent;
                if !extent.writing
                    && !other.writing
                    && other.offset == extent.offset
                    && other.bytes == extent.bytes
                    && !matches!(*request.outcome.lock().unwrap(), Some(Err(_)))
                {
                    state.statistics.logical_requests += 1;
                    state.statistics.shared_reads += 1;
                    return Ok(request);
                }
                if (extent.writing || other.writing)
                    && extent.offset < other.offset + other.bytes as u64
                    && other.offset < extent.offset + extent.bytes as u64
                {
                    return Err(Error::Busy(
                        "overlapping storage write is still retained".into(),
                    ));
                }
            }
        }
        let slot = state
            .slots
            .iter()
            .position(|s| s.strong_count() == 0)
            .ok_or_else(|| Error::Busy("all storage payload slots are retained".into()))?;
        if let Some(data) = data {
            self.shared
                .payload
                .write(slot * self.shared.config.slot_bytes, data)?;
        }
        let request = Arc::new(Request {
            shared: self.shared.clone(),
            extent,
            physical,
            slot,
            submitted: Instant::now(),
            outcome: Mutex::new(None),
            ready: Condvar::new(),
        });
        state.slots[slot] = Arc::downgrade(&request);
        state.pending.push_back(request.clone());
        state.statistics.logical_requests += 1;
        state.statistics.peak_slots = state
            .statistics
            .peak_slots
            .max(state.slots.iter().filter(|s| s.strong_count() > 0).count());
        self.shared.changed.notify_one();
        Ok(request)
    }
    /// Snapshot counters without waiting for outstanding operations.
    pub fn statistics(&self) -> StorageStatistics {
        self.shared.state.lock().unwrap().statistics.clone()
    }
}
impl Drop for StorageSession {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().stop = true;
        self.shared.changed.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
/// A retained asynchronous read or write. Dropping it never cancels accepted I/O.
#[derive(Clone)]
pub struct ReadTicket {
    request: Arc<Request>,
}
impl ReadTicket {
    /// Observe completion without releasing the payload slot.
    pub fn is_complete(&self) -> bool {
        self.request.outcome.lock().unwrap().is_some()
    }
    /// Wait under one deadline. A timeout keeps the request and slot alive.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<Option<ReadLease>> {
        let state = self.request.outcome.lock().unwrap();
        let (state, _) = self
            .request
            .ready
            .wait_timeout_while(state, timeout, |s| s.is_none())
            .unwrap();
        match state.as_ref() {
            None => Ok(None),
            Some(Err(error)) => Err(Error::Execution {
                source: error.clone(),
            }),
            Some(Ok(_)) => Ok(Some(ReadLease {
                request: self.request.clone(),
            })),
        }
    }
    /// Wait for a completed exact extent.
    pub fn wait(&self) -> Result<ReadLease> {
        loop {
            if let Some(lease) = self.wait_timeout(Duration::from_secs(1))? {
                return Ok(lease);
            }
        }
    }
}
/// Completed payload, retained until every reader and copy ticket releases it.
#[derive(Clone)]
pub struct ReadLease {
    request: Arc<Request>,
}
impl ReadLease {
    /// Logical extent, excluding direct-I/O padding.
    pub fn len(&self) -> usize {
        self.request.extent.bytes
    }
    /// Whether this extent is empty (zero-length submissions are rejected).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copy completed bytes to a caller's host slice. May return Busy while
    /// another slot in the registered payload has outstanding device use.
    pub fn read(&self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() != self.len() {
            return Err(Error::Message("storage readback length mismatch".into()));
        }
        self.request.shared.payload.read(self.offset(), bytes)
    }
    fn offset(&self) -> usize {
        self.request.slot * self.request.shared.config.slot_bytes + self.request.physical.skip
    }
    /// Copy into this stream's allocation, retaining the slot through completion.
    pub fn copy_to(
        self,
        stream: &mut crate::Stream,
        destination: crate::View<'_>,
    ) -> Result<StorageTransfer> {
        let destination = destination.slice(0, self.len())?;
        // SAFETY: only an immediate copy is submitted; no source binding escapes.
        unsafe { self.enqueue(stream, |stream, source| stream.copy(destination, source)) }
    }
    /// Submit an immediate consumer of this payload on the supplied stream.
    /// The returned transfer retains the slot until all submitted work retires.
    ///
    /// # Safety
    /// The operation must only read the source, must submit all source uses on
    /// this stream before returning, and must not retain source bindings in a
    /// graph or another queue for later execution.
    pub unsafe fn enqueue(
        self,
        stream: &mut crate::Stream,
        operation: impl FnOnce(&mut crate::Stream, crate::View<'_>) -> Result<()>,
    ) -> Result<StorageTransfer> {
        if stream.device_id() != self.request.shared.device.id() {
            return Err(Error::Message(
                "storage and stream belong to different native domains".into(),
            ));
        }
        let buffer = stream.storage_buffer(self.request.shared.payload.clone())?;
        let source = buffer.try_slice(self.offset(), self.len())?;
        let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            operation(stream, source).and_then(|()| stream.record_event())
        }));
        let submitted = match submitted {
            Ok(result) => result,
            Err(panic) => {
                if let Err(failure) = stream.synchronize() {
                    stop_after_failure(&self.request.shared, failure);
                    std::mem::forget((self, buffer));
                }
                std::panic::resume_unwind(panic);
            }
        };
        let event = match submitted {
            Ok(event) => event,
            Err(error) => {
                if let Err(failure) = stream.synchronize() {
                    stop_after_failure(&self.request.shared, failure);
                    std::mem::forget((self, buffer));
                }
                return Err(error);
            }
        };
        Ok(StorageTransfer {
            retained: Some((self, buffer)),
            event,
        })
    }
}
/// Source ownership for an asynchronous stream copy.
pub struct StorageTransfer {
    retained: Option<(ReadLease, crate::Buffer)>,
    event: crate::Event,
}
impl StorageTransfer {
    fn fail(&self, error: Error) -> Error {
        match &self.retained {
            Some((lease, _)) => stop_after_failure(&lease.request.shared, error),
            None => error,
        }
    }
    /// Reclaim the source only after the stream copy has completed.
    pub fn is_complete(&mut self) -> Result<bool> {
        let complete = self.event.is_complete().map_err(|error| self.fail(error))?;
        if complete {
            self.retained.take();
            Ok(true)
        } else {
            Ok(false)
        }
    }
    /// Wait and release source capacity; failures keep it retained.
    pub fn wait(&mut self) -> Result<()> {
        self.event.synchronize().map_err(|error| self.fail(error))?;
        self.retained.take();
        Ok(())
    }
}
impl Drop for StorageTransfer {
    fn drop(&mut self) {
        if self.retained.is_some() && self.wait().is_err() {
            std::mem::forget(self.retained.take());
        }
    }
}
struct Worker {
    ring: Option<fabric::StorageRing>,
    descriptors: fabric::Buffer,
    results: fabric::Buffer,
    kernel: fabric::Kernel,
    queue: fabric::Queue,
    budget: Option<crate::residency::MemoryBudget>,
    requests: u64,
}
impl Worker {
    fn new(shared: &Shared, budget: Option<crate::residency::MemoryBudget>) -> Result<Self> {
        let files = shared
            .files
            .iter()
            .map(|f| f.file.try_clone())
            .collect::<std::io::Result<Vec<_>>>()?;
        let ring = fabric::StorageRing::new(
            &shared.device,
            &files,
            &shared.payload,
            fabric::StorageOptions {
                entries: 64,
                progress: shared.config.progress,
                memory_budget: budget.clone(),
                ..Default::default()
            },
        )?;
        let allocate = |bytes| {
            shared.device.fabric().allocate_registered(
                bytes,
                std::slice::from_ref(&shared.device),
                budget.as_ref(),
            )
        };
        let descriptors = allocate(64 * 32)?;
        let results = allocate(64 * 4)?;
        let artifact = crate::loom::Compiler::for_target(None, shared.device.target())?
            .module(include_str!("batch.loom"))
            .compile(&crate::loom::Specialization::new("storage_batch"))?;
        let kernel = unsafe { shared.device.load(&artifact) }?;
        let queue = shared.device.queue()?;
        Ok(Self {
            ring: Some(ring),
            descriptors,
            results,
            kernel,
            queue,
            budget,
            requests: 0,
        })
    }
    fn run(&mut self, shared: Arc<Shared>) {
        loop {
            let jobs = {
                let mut state = shared.state.lock().unwrap();
                while state.pending.is_empty() && !state.stop {
                    state = shared.changed.wait(state).unwrap();
                }
                if state.pending.is_empty() {
                    return;
                }
                state.pending.drain(..).collect::<Vec<_>>()
            };
            let cpu_before = thread_cpu();
            let started = Instant::now();
            let outcome = self.execute(&shared, &jobs);
            let mut state = shared.state.lock().unwrap();
            state.statistics.physical_requests = self.requests;
            state.statistics.peak_outstanding = state.statistics.peak_outstanding.max(jobs.len());
            if shared.config.statistics {
                let add = |v: &mut Option<Duration>, d| *v = Some(v.unwrap_or_default() + d);
                add(
                    &mut state.statistics.admission_time,
                    jobs.iter()
                        .map(|j| started.duration_since(j.submitted))
                        .sum(),
                );
                add(
                    &mut state.statistics.ready_time,
                    jobs.iter().map(|j| j.submitted.elapsed()).sum(),
                );
                if let (Some(a), Some(b)) = (cpu_before, thread_cpu()) {
                    add(&mut state.statistics.service_cpu_time, b.saturating_sub(a));
                }
            }
            if let Some(ring) = &self.ring {
                let (service, wait) = ring.service_counts();
                state.statistics.service_calls = service as u64;
                state.statistics.wait_calls = wait as u64;
            }
            match outcome {
                Ok(results) => {
                    for (job, result) in jobs.iter().zip(results) {
                        let required = job.physical.skip + job.extent.bytes;
                        let outcome = if result < 0 {
                            Err(Arc::new(Error::Io(std::io::Error::from_raw_os_error(
                                -result,
                            ))))
                        } else if (result as usize) < required {
                            Err(Arc::new(Error::Io(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "incomplete storage extent",
                            ))))
                        } else {
                            state.statistics.completed_bytes += result as u64;
                            if job.extent.writing {
                                let identity = shared.files[job.extent.file].identity;
                                for file in
                                    shared.files.iter().filter(|file| file.identity == identity)
                                {
                                    file.bytes.fetch_max(
                                        job.extent.offset + job.extent.bytes as u64,
                                        Ordering::Release,
                                    );
                                }
                            }
                            Ok(result as usize)
                        };
                        if let Err(error) = &outcome {
                            state.statistics.errors += 1;
                            state.failed.get_or_insert_with(|| error.clone());
                            state.stop = true;
                        }
                        *job.outcome.lock().unwrap() = Some(outcome);
                        job.ready.notify_all();
                    }
                }
                Err(error) => {
                    let error = Arc::new(error);
                    state.failed = Some(error.clone());
                    state.stop = true;
                    for job in jobs.iter().chain(state.pending.iter()) {
                        *job.outcome.lock().unwrap() = Some(Err(error.clone()));
                        job.ready.notify_all();
                    }
                    state.statistics.errors += (jobs.len() + state.pending.len()) as u64;
                    state.pending.clear();
                    return;
                }
            }
            if let Some(error) = state.failed.clone() {
                for job in &state.pending {
                    *job.outcome.lock().unwrap() = Some(Err(error.clone()));
                    job.ready.notify_all();
                }
                state.statistics.errors += state.pending.len() as u64;
                state.pending.clear();
                return;
            }
            // Completion wakes consumers before this scope ends. Release service
            // references under the admission lock so an immediately following
            // request sees only real tickets/leases, not a retiring batch owner.
            drop(jobs);
            drop(state);
        }
    }
    fn execute(&mut self, shared: &Shared, jobs: &[Arc<Request>]) -> Result<Vec<i32>> {
        let mut completed = vec![0usize; jobs.len()];
        let mut results = vec![0i32; jobs.len()];
        let mut remaining: Vec<usize> = (0..jobs.len()).collect();
        while !remaining.is_empty() {
            let attempts: Vec<_> = remaining
                .iter()
                .map(|&i| (&jobs[i], completed[i]))
                .collect();
            let returned = self.execute_batch(shared, &attempts)?;
            let mut retry = Vec::new();
            let mut failed = false;
            for (i, result) in remaining.into_iter().zip(returned) {
                let job = &jobs[i];
                if result < 0 {
                    results[i] = result;
                    failed = true;
                    continue;
                }
                completed[i] += result as usize;
                results[i] = completed[i] as i32;
                if completed[i] >= job.physical.skip + job.extent.bytes {
                    continue;
                }
                // A short direct completion can only be retried at a legal boundary.
                if result == 0
                    || !completed[i].is_multiple_of(shared.files[job.extent.file].alignment)
                {
                    results[i] = if job.extent.writing {
                        -libc::EIO
                    } else {
                        results[i]
                    };
                    failed = true;
                } else {
                    retry.push(i);
                }
            }
            // All accepted SQEs have retired. Do not issue retries after any failure.
            if failed {
                break;
            }
            remaining = retry;
        }
        Ok(results)
    }
    fn execute_batch(
        &mut self,
        shared: &Shared,
        jobs: &[(&Arc<Request>, usize)],
    ) -> Result<Vec<i32>> {
        let mut descriptors = vec![0u8; jobs.len() * 32];
        for (bytes, (job, completed)) in descriptors.as_chunks_mut::<32>().0.iter_mut().zip(jobs) {
            bytes[0..8].copy_from_slice(&(job.physical.offset + *completed as u64).to_le_bytes());
            bytes[8..16].copy_from_slice(
                &((job.slot * shared.config.slot_bytes + completed) as u64).to_le_bytes(),
            );
            bytes[16..20].copy_from_slice(&(job.extent.file as u32).to_le_bytes());
            bytes[20..24]
                .copy_from_slice(&(if job.extent.writing { 5u32 } else { 4u32 }).to_le_bytes());
            bytes[24..28].copy_from_slice(&((job.physical.bytes - completed) as u32).to_le_bytes());
        }
        self.descriptors.write(0, &descriptors)?;
        let ring = self.ring.as_ref().unwrap();
        let layout = ring.layout();
        let host = layout.host_payload.to_le_bytes();
        let sq = layout.submission_mask.to_le_bytes();
        let cq = layout.completion_mask.to_le_bytes();
        let count = (jobs.len() as u32).to_le_bytes();
        use fabric::Argument::{Buffer, Value};
        let args = [
            Buffer(ring.memory(), layout.submission_entries),
            Buffer(ring.memory(), layout.submission_tail),
            Buffer(ring.memory(), layout.completion_entries),
            Buffer(ring.memory(), layout.completion_head),
            Buffer(ring.memory(), layout.completion_tail),
            Buffer(&self.descriptors, 0),
            Buffer(&self.results, 0),
            Value(&host),
            Value(&sq),
            Value(&cq),
            Value(&count),
        ];
        let command = unsafe { self.queue.prepare(&self.kernel, [1; 3], [1; 3], &args) }?;
        let charge = self
            .budget
            .as_ref()
            .map(|b| b.reserve(command.storage_bytes()))
            .transpose()?;
        let mut execution = unsafe { self.ring.take().unwrap().dispatch(command) }?;
        self.requests += jobs.len() as u64;
        match execution.wait_timeout(Duration::from_secs(30)) {
            Ok(true) => {}
            outcome => {
                // Unretired native allocations and their accounting stay together.
                std::mem::forget(charge);
                std::mem::forget(execution);
                return Err(outcome.err().unwrap_or_else(|| {
                    Error::DeviceLost("storage did not retire within 30 seconds".into())
                }));
            }
        }
        self.ring = Some(execution.into_ring()?);
        let mut bytes = vec![0; jobs.len() * 4];
        self.results.read(0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| i32::from_le_bytes(*v))
            .collect())
    }
}
fn thread_cpu() -> Option<Duration> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    (unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut value) } == 0)
        .then(|| Duration::new(value.tv_sec as u64, value.tv_nsec as u32))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligned_extents_preserve_logical_boundaries() {
        let read = Extent {
            file: 0,
            offset: 4103,
            bytes: 700,
            writing: false,
        };
        let p = physical(read, 9000, 4096, 8192).unwrap();
        assert_eq!((p.offset, p.bytes, p.skip), (4096, 4096, 7));
        assert!(physical(read, 4200, 4096, 8192).is_err());
        assert!(
            physical(
                Extent {
                    offset: u64::MAX,
                    ..read
                },
                u64::MAX,
                4096,
                8192
            )
            .is_err()
        );
        assert!(
            physical(
                Extent {
                    writing: true,
                    ..read
                },
                9000,
                4096,
                8192
            )
            .is_err()
        );
        assert!(
            physical(
                Extent {
                    offset: 1,
                    bytes: 8192,
                    ..read
                },
                9000,
                4096,
                8192
            )
            .is_err()
        );
    }
}
