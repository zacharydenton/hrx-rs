//! Address compatibility with explicitly scoped streams. New models use binding dispatch.
use super::{Error, Result, check, sys};
use std::{
    collections::BTreeMap,
    ffi::{CString, c_void},
    sync::{Arc, Mutex, Weak},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
/// A numeric GPU address; host dereferencing is never provided.
pub struct DevicePtr(usize);
impl DevicePtr {
    /// The null device address.
    pub const NULL: Self = Self(0);
    /// Wrap a numeric address; transfers validate it against live stream allocations.
    pub const fn from_address(address: usize) -> Self {
        Self(address)
    }
    /// The numeric device address.
    pub const fn address(self) -> usize {
        self.0
    }
    /// Whether the address is null.
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
    /// Offset the address in bytes; panics on arithmetic overflow.
    pub const fn offset(self, bytes: usize) -> Self {
        Self(self.0.checked_add(bytes).expect("device address overflow"))
    }
}
struct Allocation {
    raw: sys::Buffer,
    bytes: usize,
    address: usize,
    owner: Weak<Device>,
}
// Mapping is created once, before publishing the allocation. No host reference
// to it is exposed. All GPU work uses streams; mapped storage is retained until
// direct-address submissions complete.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        let mut registry = ALLOCATIONS.lock().unwrap_or_else(|e| e.into_inner());
        // A newer allocation can reuse an address only after native release.
        registry.remove(&self.address);
        drop(registry);
        unsafe {
            sys::hrx_buffer_release(self.raw);
        }
    }
}
static ALLOCATIONS: Mutex<BTreeMap<usize, Weak<Allocation>>> = Mutex::new(BTreeMap::new());

/// An owned mapped compatibility allocation tied to one stream.
pub struct Buffer {
    allocation: Arc<Allocation>,
    stream: Weak<Device>,
}
impl Buffer {
    /// The base device address.
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr(self.allocation.address)
    }
    /// The allocated byte length, rounded to at least four bytes.
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }
    /// Whether the allocation has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The owning stream, if it is still alive.
    pub fn stream(&self) -> Option<Arc<Device>> {
        self.stream.upgrade()
    }
}
struct State {
    stream: sys::Stream,
    pending: BTreeMap<usize, Arc<Allocation>>,
    allocations: Vec<Weak<Allocation>>,
}
unsafe impl Send for State {}
/// A thread-safe, mutex-serialized compatibility command stream.
pub struct Device {
    device: sys::Device,
    target: crate::Target,
    state: Mutex<State>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
static DEFAULT: Mutex<Option<Arc<Device>>> = Mutex::new(None);
thread_local! { static CURRENT: std::cell::RefCell<Option<Arc<Device>>> = const { std::cell::RefCell::new(None) }; }
/// Restores the enclosing model's stream even if a nested call unwinds. It cannot
/// move to another thread; it does not hold a mutex across user callbacks.
#[must_use = "keep the scope guard alive for the session"]
pub struct Scope {
    previous: Option<Arc<Device>>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}
/// Get the scoped or default stream, panicking if initialization fails.
pub fn device() -> Arc<Device> {
    try_device().unwrap_or_else(|e| panic!("{e}"))
}
/// Get the scoped or default stream; failed initialization remains retryable.
pub fn try_device() -> Result<Arc<Device>> {
    if let Some(d) = CURRENT.with(|c| c.borrow().clone()) {
        return Ok(d);
    }
    crate::cached_init(&DEFAULT, Device::open)
}
impl Device {
    /// Open a stream on the first GPU matching the default compiler target.
    pub fn open() -> Result<Arc<Self>> {
        Self::open_for_target(&crate::Target::default())
    }
    /// Open a stream on the first GPU reporting the requested architecture.
    pub fn open_for_target(target: &crate::Target) -> Result<Arc<Self>> {
        crate::runtime::initialize_runtime()?;
        let mut count = 0;
        unsafe {
            check(sys::hrx_gpu_device_count(&mut count))?;
        }
        for index in 0..count {
            let (_, architecture) = crate::runtime::open_device(index)?;
            if &architecture == target {
                return Self::open_device(index);
            }
        }
        Err(Error::Message(format!(
            "no GPU matching {}",
            target.as_str()
        )))
    }
    /// Open an independent stream on the selected GPU index.
    pub fn open_device(index: i32) -> Result<Arc<Self>> {
        let (device, target) = crate::runtime::open_device(index)?;
        unsafe {
            let mut stream = std::ptr::null_mut();
            check(sys::hrx_stream_create(device, 0, &mut stream))?;
            Ok(Arc::new(Self {
                device,
                target,
                state: Mutex::new(State {
                    stream,
                    pending: BTreeMap::new(),
                    allocations: Vec::new(),
                }),
            }))
        }
    }
    /// Architecture reported by the selected device.
    pub fn target(&self) -> &crate::Target {
        &self.target
    }
    /// Return the current thread's scoped stream, or open an independent stream
    /// when no scope is active.
    pub fn current_or_new() -> Result<Arc<Self>> {
        CURRENT
            .with(|c| c.borrow().clone())
            .map_or_else(Self::open, Ok)
    }
    /// Select this stream for the current thread until the returned guard is dropped.
    pub fn enter(self: &Arc<Self>) -> Scope {
        let previous = CURRENT.with(|c| c.replace(Some(self.clone())));
        Scope {
            previous,
            _thread: std::marker::PhantomData,
        }
    }
    pub(crate) fn raw(&self) -> sys::Device {
        self.device
    }
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| Error::Message("GPU stream is poisoned".into()))
    }
    fn drain(state: &mut State) -> Result<()> {
        unsafe {
            check(sys::hrx_stream_synchronize(state.stream))?;
        }
        state.pending.clear();
        Ok(())
    }
    /// Wait for this stream and release its pending direct-address allocations.
    pub fn synchronize(&self) -> Result<()> {
        Self::drain(&mut *self.lock()?)
    }
    /// Allocate mapped storage owned by this stream, rounded to at least four bytes.
    pub fn allocate(self: &Arc<Self>, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(4);
        unsafe {
            let mut raw = std::ptr::null_mut();
            check(sys::hrx_allocator_allocate_buffer(
                sys::hrx_device_allocator(self.device),
                sys::BufferParams {
                    memory_type: sys::MEMORY_TYPE_DEVICE_LOCAL | sys::MEMORY_TYPE_HOST_VISIBLE,
                    access: sys::MEMORY_ACCESS_ALL,
                    usage: sys::BUFFER_USAGE_DEFAULT | sys::BUFFER_USAGE_MAPPING_SCOPED,
                    queue_affinity: u64::MAX,
                },
                bytes,
                &mut raw,
            ))?;
            let mut pointer = std::ptr::null_mut();
            if let Err(e) = check(sys::hrx_buffer_get_device_ptr(raw, &mut pointer)) {
                sys::hrx_buffer_release(raw);
                return Err(e);
            }
            let allocation = Arc::new(Allocation {
                raw,
                bytes,
                address: pointer as usize,
                owner: Arc::downgrade(self),
            });
            ALLOCATIONS
                .lock()
                .map_err(|_| Error::Message("allocation registry poisoned".into()))?
                .insert(pointer as usize, Arc::downgrade(&allocation));
            {
                let mut state = self.lock()?;
                // Bound dead weak entries even when this stream never uses raw arguments.
                if state.allocations.len() % 64 == 0 {
                    state.allocations.retain(|a| a.strong_count() != 0);
                }
                state.allocations.push(Arc::downgrade(&allocation));
            }
            Ok(Buffer {
                allocation,
                stream: Arc::downgrade(self),
            })
        }
    }
    fn find(&self, pointer: DevicePtr, bytes: usize) -> Result<(Arc<Allocation>, usize)> {
        let allocations = ALLOCATIONS
            .lock()
            .map_err(|_| Error::Message("allocation registry poisoned".into()))?;
        let (base, allocation) = allocations
            .range(..=pointer.address())
            .rev()
            .find_map(|(base, w)| w.upgrade().map(|a| (*base, a)))
            .ok_or_else(|| Error::Message("address does not name a live GPU allocation".into()))?;
        drop(allocations);
        if !std::ptr::eq(allocation.owner.as_ptr(), self) {
            return Err(Error::Message(
                "allocation belongs to another stream".into(),
            ));
        }
        let offset = pointer.address() - base;
        crate::runtime::checked_span(offset, bytes, allocation.bytes)?;
        Ok((allocation, offset))
    }
    /// Queue a checked copy between addresses owned by this stream.
    pub fn copy_device_to_device(
        &self,
        destination: DevicePtr,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        let (dst, d) = self.find(destination, bytes)?;
        let (src, s) = self.find(source, bytes)?;
        let mut state = self.lock()?;
        state.pending.insert(dst.address, dst.clone());
        state.pending.insert(src.address, src.clone());
        unsafe {
            check(sys::hrx_stream_copy_buffer(
                state.stream,
                src.raw,
                s,
                dst.raw,
                d,
                bytes,
            ))
        }
    }
    /// Drain this stream and upload plain-data values to a checked address.
    pub fn write<T: bytemuck::Pod>(&self, destination: DevicePtr, source: &[T]) -> Result<()> {
        self.copy_from_host(destination, bytemuck::cast_slice(source))
    }
    /// Drain this stream and read plain-data values from a checked address.
    pub fn read<T: bytemuck::Pod>(&self, destination: &mut [T], source: DevicePtr) -> Result<()> {
        self.copy_to_host(bytemuck::cast_slice_mut(destination), source)
    }
    /// Drain this stream and upload bytes to one of its allocations.
    pub fn copy_from_host(&self, destination: DevicePtr, source: &[u8]) -> Result<()> {
        let (dst, offset) = self.find(destination, source.len())?;
        let mut state = self.lock()?;
        Self::drain(&mut state)?;
        if source.is_empty() {
            return Ok(());
        }
        unsafe {
            check(sys::hrx_synchronous_h2d(
                self.device,
                source.as_ptr().cast(),
                dst.raw,
                offset,
                source.len(),
            ))
        }
    }
    /// Drain this stream and download bytes from one of its allocations.
    pub fn copy_to_host(&self, destination: &mut [u8], source: DevicePtr) -> Result<()> {
        let (src, offset) = self.find(source, destination.len())?;
        let mut state = self.lock()?;
        Self::drain(&mut state)?;
        if destination.is_empty() {
            return Ok(());
        }
        unsafe {
            check(sys::hrx_synchronous_d2h(
                self.device,
                src.raw,
                offset,
                destination.as_mut_ptr().cast(),
                destination.len(),
            ))
        }
    }
    /// Queue a zero fill over a checked span owned by this stream.
    pub fn zero(&self, destination: DevicePtr, bytes: usize) -> Result<()> {
        let (dst, offset) = self.find(destination, bytes)?;
        let mut state = self.lock()?;
        state.pending.insert(dst.address, dst.clone());
        let pattern = 0u8;
        unsafe {
            check(sys::hrx_stream_fill_buffer(
                state.stream,
                dst.raw,
                offset,
                bytes,
                (&pattern as *const u8).cast(),
                1,
            ))
        }
    }
    /// # Safety
    /// The invocation must satisfy Kernel::launch's address and layout contract.
    pub(crate) unsafe fn dispatch(
        &self,
        executable: sys::Executable,
        ordinal: u32,
        config: &sys::DispatchConfig,
        args: &super::Args,
    ) -> Result<()> {
        let mut state = self.lock()?;
        if args.opaque {
            // O(live allocations on this stream), independent of other sessions.
            // Explicit pointer arguments avoid this conservative raw-blob fallback.
            let State {
                allocations,
                pending,
                ..
            } = &mut *state;
            allocations.retain(|allocation| {
                if let Some(a) = allocation.upgrade() {
                    pending.entry(a.address).or_insert(a);
                    true
                } else {
                    false
                }
            });
        } else {
            for &address in &args.pointers[..args.pointer_count] {
                // Prepared model dispatches normally reuse already-retained
                // weights and scratch. Avoid registry locks and Arc churn there.
                if state
                    .pending
                    .range(..=address)
                    .next_back()
                    .is_some_and(|(base, allocation)| address - base < allocation.bytes)
                {
                    continue;
                }
                let (a, _) = self.find(DevicePtr(address), 0)?;
                state.pending.entry(a.address).or_insert(a);
            }
        }
        unsafe {
            check(sys::hrx_stream_dispatch(
                state.stream,
                executable,
                ordinal,
                config,
                args.as_bytes().as_ptr() as *const c_void,
                args.as_bytes().len(),
                std::ptr::null(),
                0,
                sys::DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
            ))
        }
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        unsafe {
            if check(sys::hrx_stream_synchronize(state.stream)).is_err() {
                std::mem::forget(std::mem::take(&mut state.pending));
            }
            sys::hrx_stream_release(state.stream);
        }
    }
}
pub(crate) fn c_string(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error::Message("string contains NUL".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151"]
    fn dropped_inflight_allocations_leave_no_registry_entries() -> Result<()> {
        let device = Device::open()?;
        let buffer = device.allocate(16)?;
        let address = buffer.ptr().address();
        device.zero(buffer.ptr(), 16)?;
        drop(buffer);
        assert!(ALLOCATIONS.lock().unwrap().contains_key(&address));
        device.synchronize()?;
        assert!(!ALLOCATIONS.lock().unwrap().contains_key(&address));
        Ok(())
    }
}
