//! GPU-published Linux io_uring requests over retained, registered host pages.
use super::{memory::DeviceUse, *};
use std::{
    fs::File,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::atomic::{AtomicU32, Ordering},
    time::{Duration, Instant},
};
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    missing_docs,
    clippy::all
)]
#[path = "storage_ffi.rs"]
mod ffi;
use ffi::*;

// Stable Linux UAPI operations; the generated layouts carry size/offset asserts.
const REGISTER_BUFFERS: u32 = 0;
const REGISTER_FILES: u32 = 2;
const REGISTER_RESTRICTIONS: u32 = 11;
const ENABLE_RINGS: u32 = 12;

/// Bounded caller-owned SQ/CQ storage and its kernel SQPOLL service.
#[derive(Clone, Debug)]
pub struct StorageOptions {
    /// Power-of-two SQ capacity, 2 through 4096. CQ capacity is twice this value.
    pub entries: u32,
    /// Milliseconds before an idle SQPOLL worker sleeps; waits issue idle wakes.
    pub idle_milliseconds: u32,
    /// Shared residency ceiling for ring pages. Payload pages carry their own charge.
    pub memory_budget: Option<crate::residency::MemoryBudget>,
}
impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            entries: 8,
            idle_milliseconds: 1,
            memory_budget: None,
        }
    }
}
/// Returned native ring offsets. GPU addresses are obtained from [`StorageRing::memory`].
/// `host_payload` is a CPU virtual address for Linux fixed-buffer I/O, never a GPU VA.
#[derive(Clone, Copy, Debug)]
pub struct StorageLayout {
    /// Offset of the native 64-byte SQE array.
    pub submission_entries: usize,
    /// Offset of the 32-bit SQ publisher tail.
    pub submission_tail: usize,
    /// Offset of the native 16-byte CQE array.
    pub completion_entries: usize,
    /// Offset of the 32-bit CQ consumer head.
    pub completion_head: usize,
    /// Offset of the 32-bit CQ producer tail.
    pub completion_tail: usize,
    /// Native SQ entry mask.
    pub submission_mask: u32,
    /// Native CQ entry mask.
    pub completion_mask: u32,
    /// CPU address of fixed buffer zero.
    pub host_payload: u64,
}
/// One fixed-file table, one fixed payload buffer, and a single GPU SQ/CQ owner.
/// Ring restrictions permit only fixed-file READ_FIXED and WRITE_FIXED requests.
/// They are not a sandbox for untrusted native programs.
pub struct StorageRing {
    fd: OwnedFd,
    memory: Buffer,
    // SQPOLL may update ring controls even before the GPU is submitted. Block
    // ordinary host buffer access for the entire enabled kernel ring lifetime.
    _kernel_use: DeviceUse,
    payload: Buffer,
    _files: Vec<File>,
    parameters: io_uring_params,
    control: usize,
    layout: StorageLayout,
}
fn syscall_result(result: libc::c_long) -> Result<libc::c_long> {
    if result < 0 {
        Err(std::io::Error::last_os_error().into())
    } else {
        Ok(result)
    }
}
impl StorageRing {
    /// Register regular files and a payload allocated by `allocate_registered`.
    /// File order defines fixed-file indices; payload is fixed-buffer index zero.
    /// Requires NO_MMAP, NO_SQARRAY, restrictions and SQPOLL support. Admission
    /// failure is explicit; no alternate transport or host request relay is used.
    pub fn new(
        device: &Device,
        files: &[File],
        payload: &Buffer,
        options: StorageOptions,
    ) -> Result<Self> {
        if device.endpoint().engine() != Engine::Gpu
            || files.is_empty()
            || files.len() > 1024
            || !options.entries.is_power_of_two()
            || !(2..=4096).contains(&options.entries)
            || options.idle_milliseconds == 0
        {
            return Err(Error::Message(
                "invalid GPU storage ring configuration".into(),
            ));
        }
        if !payload.owns_registered_pages() {
            return Err(Error::Unsupported(
                "storage payload requires owned registered host pages".into(),
            ));
        }
        payload.device_address(device)?;
        let files = files
            .iter()
            .map(|file| {
                if !file.metadata()?.is_file() {
                    return Err(Error::Unsupported("storage requires regular files".into()));
                }
                Ok(file.try_clone()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(Error::Unsupported("host page size unavailable".into()));
        }
        let page = page as usize;
        let round = |n: usize| n.div_ceil(page) * page;
        let control = round(options.entries as usize * 64);
        let control_bytes = page + round(options.entries as usize * 2 * 16);
        let memory = device.fabric().allocate_registered(
            control + control_bytes,
            std::slice::from_ref(device),
            options.memory_budget.as_ref(),
        )?;
        let mut parameters = io_uring_params {
            flags: IORING_SETUP_NO_MMAP
                | IORING_SETUP_NO_SQARRAY
                | IORING_SETUP_R_DISABLED
                | IORING_SETUP_SQPOLL,
            sq_thread_idle: options.idle_milliseconds,
            ..Default::default()
        };
        parameters.sq_off.user_addr = memory.host_pointer() as u64;
        parameters.cq_off.user_addr = parameters.sq_off.user_addr + control as u64;
        let fd =
            unsafe { libc::syscall(libc::SYS_io_uring_setup, options.entries, &mut parameters) };
        if fd < 0 {
            let error = std::io::Error::last_os_error();
            return if matches!(
                error.raw_os_error(),
                Some(libc::ENOSYS | libc::EINVAL | libc::EPERM | libc::EACCES)
            ) {
                Err(Error::Unsupported(format!(
                    "caller-owned SQPOLL ring unavailable: {error}"
                )))
            } else {
                Err(error.into())
            };
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd as i32) };
        if parameters.sq_entries != options.entries
            || parameters.cq_entries != options.entries * 2
            || parameters.cq_off.cqes as usize + parameters.cq_entries as usize * 16 > control_bytes
        {
            return Err(Error::Unsupported(
                "unexpected io_uring storage requirements".into(),
            ));
        }
        for offset in [
            parameters.sq_off.head,
            parameters.sq_off.tail,
            parameters.sq_off.flags,
            parameters.sq_off.dropped,
            parameters.cq_off.head,
            parameters.cq_off.tail,
            parameters.cq_off.overflow,
        ] {
            if offset % 4 != 0 || offset as usize + 4 > control_bytes {
                return Err(Error::Unsupported("invalid native ring offsets".into()));
            }
        }
        let layout = StorageLayout {
            submission_entries: 0,
            submission_tail: control + parameters.sq_off.tail as usize,
            completion_entries: control + parameters.cq_off.cqes as usize,
            completion_head: control + parameters.cq_off.head as usize,
            completion_tail: control + parameters.cq_off.tail as usize,
            submission_mask: parameters.sq_entries - 1,
            completion_mask: parameters.cq_entries - 1,
            host_payload: payload.host_pointer() as u64,
        };
        let kernel_use = memory.retain_use()?;
        let ring = Self {
            fd,
            memory,
            _kernel_use: kernel_use,
            payload: payload.clone(),
            _files: files,
            parameters,
            control,
            layout,
        };
        let iov = libc::iovec {
            iov_base: payload.host_pointer().cast(),
            iov_len: payload.len(),
        };
        ring.register(REGISTER_BUFFERS, &iov as *const _ as *const libc::c_void, 1)?;
        let descriptors: Vec<i32> = ring._files.iter().map(AsRawFd::as_raw_fd).collect();
        ring.register(
            REGISTER_FILES,
            descriptors.as_ptr().cast(),
            descriptors.len() as u32,
        )?;
        // restriction opcode: register-op=0, SQE-op=1, flags-allowed=2, flags-required=3.
        let restrictions =
            [(1, 4), (1, 5), (2, 1), (3, 1), (0, ENABLE_RINGS as u8)].map(|(opcode, value)| {
                io_uring_restriction {
                    opcode,
                    __bindgen_anon_1: io_uring_restriction__bindgen_ty_1 { register_op: value },
                    ..Default::default()
                }
            });
        ring.register(
            REGISTER_RESTRICTIONS,
            restrictions.as_ptr().cast(),
            restrictions.len() as u32,
        )?;
        ring.register(ENABLE_RINGS, ptr::null(), 0)?;
        Ok(ring)
    }
    fn register(&self, operation: u32, data: *const libc::c_void, count: u32) -> Result<()> {
        syscall_result(unsafe {
            libc::syscall(
                libc::SYS_io_uring_register,
                self.fd.as_raw_fd(),
                operation,
                data,
                count,
            )
        })?;
        Ok(())
    }
    /// Ring memory for preparing trusted GPU arguments. Ordinary host reads and
    /// writes return Busy for the entire ring lifetime: SQPOLL owns its controls.
    pub fn memory(&self) -> &Buffer {
        &self.memory
    }
    /// Retained registered payload backing.
    pub fn payload(&self) -> &Buffer {
        &self.payload
    }
    /// Native offsets, capacities and fixed-buffer CPU address.
    pub fn layout(&self) -> StorageLayout {
        self.layout
    }
    fn word(&self, offset: u32) -> u32 {
        unsafe {
            (&*self
                .memory
                .host_pointer()
                .add(self.control + offset as usize)
                .cast::<AtomicU32>())
                .load(Ordering::Acquire)
        }
    }
    fn wake(&self) -> Result<()> {
        if self.word(self.parameters.sq_off.flags) & IORING_SQ_NEED_WAKEUP != 0
            && self.word(self.parameters.sq_off.tail) != self.word(self.parameters.sq_off.head)
        {
            let result = unsafe {
                libc::syscall(
                    libc::SYS_io_uring_enter,
                    self.fd.as_raw_fd(),
                    0u32,
                    0u32,
                    IORING_ENTER_SQ_WAKEUP,
                    ptr::null::<u8>(),
                    0usize,
                )
            };
            if result < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                return Ok(());
            }
            syscall_result(result)?;
        }
        Ok(())
    }
    fn drained(&self) -> bool {
        let tail = self.word(self.parameters.sq_off.tail);
        tail == self.word(self.parameters.sq_off.head)
            && tail == self.word(self.parameters.cq_off.tail)
    }
    /// Submit one GPU owner of this initially empty ring and retain all resources.
    /// # Safety
    /// `command` must be the sole SQ publisher and CQ consumer, obey native ring
    /// capacity/release/acquire rules, handle short/error CQEs, and drain accepted
    /// I/O before returning. All fixed-file accesses must be authorized and ordered
    /// against external users. GPU and CPU payload addresses must not be confused.
    pub unsafe fn dispatch(self, command: PreparedGpu) -> Result<StorageExecution> {
        let uses = vec![self.memory.retain_use()?, self.payload.retain_use()?];
        let done = unsafe { command.dispatch() }?;
        Ok(StorageExecution {
            owners: Some(StorageOwners {
                ring: self,
                _command: command,
                done,
                uses,
            }),
            retired: false,
        })
    }
}
struct StorageOwners {
    ring: StorageRing,
    _command: PreparedGpu,
    done: Completion,
    uses: Vec<DeviceUse>,
}
/// Live GPU file work. Waits only wake idle SQPOLL; the host never authors SQEs,
/// consumes CQEs, or reads/writes application payload to make device progress.
pub struct StorageExecution {
    owners: Option<StorageOwners>,
    retired: bool,
}
impl StorageExecution {
    /// Cached combined GPU/kernel-I/O retirement.
    pub fn is_complete(&self) -> bool {
        self.retired
    }
    /// Service idle wakes and check both completion domains under one deadline.
    /// Timeout retains files, ring pages, payload and their residency charges.
    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<bool> {
        if self.retired {
            return Ok(true);
        }
        let start = Instant::now();
        let owners = self.owners.as_mut().unwrap();
        loop {
            owners.ring.wake()?;
            let gpu = owners.done.refresh()?;
            if gpu && owners.ring.drained() {
                self.retired = true;
                owners.uses.clear();
                return Ok(true);
            }
            if start.elapsed() >= timeout {
                return Ok(false);
            }
            std::thread::yield_now();
        }
    }
}
impl Drop for StorageExecution {
    fn drop(&mut self) {
        if !self.wait_timeout(Duration::from_secs(10)).unwrap_or(false) {
            std::mem::forget(self.owners.take());
        }
    }
}
