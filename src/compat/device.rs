//! Address compatibility with explicitly scoped streams. New models use binding dispatch.
use super::{Error, Result, check, sys};
use std::{
    collections::BTreeMap,
    ffi::{CString, c_void},
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct DevicePtr(usize);
impl DevicePtr {
    pub const NULL: Self = Self(0);
    pub const fn from_address(address: usize) -> Self {
        Self(address)
    }
    pub const fn address(self) -> usize {
        self.0
    }
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
    pub const fn offset(self, bytes: usize) -> Self {
        Self(self.0.checked_add(bytes).expect("device address overflow"))
    }
}
struct Allocation {
    raw: sys::Buffer,
    bytes: usize,
    address: usize,
}
// Mapping is created once, before publishing the allocation. No host reference
// to it is exposed. All GPU work uses streams; mapped storage is retained until
// direct-address submissions complete.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            sys::hrx_buffer_release(self.raw);
        }
    }
}
static ALLOCATIONS: Mutex<BTreeMap<usize, Weak<Allocation>>> = Mutex::new(BTreeMap::new());

pub struct Buffer {
    allocation: Arc<Allocation>,
    stream: Weak<Device>,
}
impl Buffer {
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr(self.allocation.address)
    }
    pub fn len(&self) -> usize {
        self.allocation.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn stream(&self) -> Option<Arc<Device>> {
        self.stream.upgrade()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // Remove weak registry entries without waiting for GPU work. The stream
        // keeps any direct-address references; native binding copies retain HAL buffers.
        let mut allocations = ALLOCATIONS.lock().unwrap_or_else(|e| e.into_inner());
        if allocations
            .get(&self.allocation.address)
            .is_some_and(|w| w.strong_count() == 1)
        {
            allocations.remove(&self.allocation.address);
        }
    }
}
struct State {
    stream: sys::Stream,
    pending: BTreeMap<usize, Arc<Allocation>>,
}
unsafe impl Send for State {}
pub struct Device {
    device: sys::Device,
    state: Mutex<State>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
static DEFAULT: OnceLock<Result<Arc<Device>>> = OnceLock::new();
thread_local! { static CURRENT: std::cell::RefCell<Option<Arc<Device>>> = const { std::cell::RefCell::new(None) }; }
/// Restores the enclosing model's stream even if a nested call unwinds. It cannot
/// move to another thread; it does not hold a mutex across user callbacks.
pub struct Scope {
    previous: Option<Arc<Device>>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}
pub fn device() -> Arc<Device> {
    try_device().unwrap_or_else(|e| panic!("{e}"))
}
pub fn try_device() -> Result<Arc<Device>> {
    if let Some(d) = CURRENT.with(|c| c.borrow().clone()) {
        return Ok(d);
    }
    DEFAULT.get_or_init(Device::open).clone()
}
impl Device {
    pub fn open() -> Result<Arc<Self>> {
        crate::runtime::initialize_runtime()?;
        unsafe {
            let mut count = 0;
            check(sys::hrx_gpu_device_count(&mut count))?;
            for index in 0..count {
                let mut device = std::ptr::null_mut();
                check(sys::hrx_gpu_device_get(index, &mut device))?;
                let mut arch = [0u8; 64];
                check(sys::hrx_device_get_property(
                    device,
                    1,
                    arch.as_mut_ptr().cast(),
                    arch.len(),
                ))?;
                if arch.split(|b| *b == 0).next() != Some(b"gfx1151") {
                    continue;
                }
                let mut stream = std::ptr::null_mut();
                check(sys::hrx_stream_create(device, 0, &mut stream))?;
                return Ok(Arc::new(Self {
                    device,
                    state: Mutex::new(State {
                        stream,
                        pending: BTreeMap::new(),
                    }),
                }));
            }
        }
        Err(Error("gfx1151 GPU required".into()))
    }
    /// Nested block sessions inherit their pipeline's stream; standalone sessions
    /// open one of their own. Legacy low-level callers still have a default stream.
    pub fn current_or_new() -> Result<Arc<Self>> {
        CURRENT
            .with(|c| c.borrow().clone())
            .map_or_else(Self::open, Ok)
    }
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
            .map_err(|_| Error("GPU stream is poisoned".into()))
    }
    fn drain(state: &mut State) -> Result<()> {
        unsafe {
            check(sys::hrx_stream_synchronize(state.stream))?;
        }
        state.pending.clear();
        // Bound weak entries left behind when a buffer was dropped in flight.
        ALLOCATIONS
            .lock()
            .map_err(|_| Error("allocation registry poisoned".into()))?
            .retain(|_, w| w.strong_count() != 0);
        Ok(())
    }
    pub fn synchronize(&self) -> Result<()> {
        Self::drain(&mut *self.lock()?)
    }
    pub fn allocate(self: &Arc<Self>, bytes: usize) -> Result<Buffer> {
        let bytes = bytes.max(4);
        unsafe {
            let mut raw = std::ptr::null_mut();
            check(sys::hrx_allocator_allocate_buffer(
                sys::hrx_device_allocator(self.device),
                sys::BufferParams {
                    memory_type: 0x30 | 2,
                    access: 7,
                    usage: 0xc03 | 0x0100_0000,
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
            });
            ALLOCATIONS
                .lock()
                .map_err(|_| Error("allocation registry poisoned".into()))?
                .insert(pointer as usize, Arc::downgrade(&allocation));
            Ok(Buffer {
                allocation,
                stream: Arc::downgrade(self),
            })
        }
    }
    fn find(pointer: DevicePtr, bytes: usize) -> Result<(Arc<Allocation>, usize)> {
        let allocations = ALLOCATIONS
            .lock()
            .map_err(|_| Error("allocation registry poisoned".into()))?;
        let (base, allocation) = allocations
            .range(..=pointer.address())
            .rev()
            .find_map(|(base, w)| w.upgrade().map(|a| (*base, a)))
            .ok_or_else(|| Error("address does not name a live GPU allocation".into()))?;
        let offset = pointer.address() - base;
        crate::runtime::checked_span(offset, bytes, allocation.bytes)?;
        Ok((allocation, offset))
    }
    pub fn copy_device_to_device(
        &self,
        destination: DevicePtr,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        let (dst, d) = Self::find(destination, bytes)?;
        let (src, s) = Self::find(source, bytes)?;
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
    pub fn write<T: bytemuck::Pod>(&self, destination: DevicePtr, source: &[T]) -> Result<()> {
        self.copy_from_host(destination, bytemuck::cast_slice(source))
    }
    pub fn read<T: bytemuck::Pod>(&self, destination: &mut [T], source: DevicePtr) -> Result<()> {
        self.copy_to_host(bytemuck::cast_slice_mut(destination), source)
    }
    pub fn copy_from_host(&self, destination: DevicePtr, source: &[u8]) -> Result<()> {
        let (dst, offset) = Self::find(destination, source.len())?;
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
    pub fn copy_to_host(&self, destination: &mut [u8], source: DevicePtr) -> Result<()> {
        let (src, offset) = Self::find(source, destination.len())?;
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
    pub fn zero(&self, destination: DevicePtr, bytes: usize) -> Result<()> {
        let (dst, offset) = Self::find(destination, bytes)?;
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
    pub(crate) fn dispatch(
        &self,
        executable: sys::Executable,
        ordinal: u32,
        config: &sys::DispatchConfig,
        args: &super::Args,
    ) -> Result<()> {
        let mut state = self.lock()?;
        if args.opaque {
            // Escape hatch for existing test bridges. With no pointer metadata,
            // conservatively retain all live allocations through completion.
            let allocations = ALLOCATIONS
                .lock()
                .map_err(|_| Error("allocation registry poisoned".into()))?;
            for (address, allocation) in allocations.iter() {
                if let Some(a) = allocation.upgrade() {
                    state.pending.insert(*address, a);
                }
            }
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
                let (a, _) = Self::find(DevicePtr(address), 0)?;
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
                1,
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
    CString::new(text).map_err(|_| Error("string contains NUL".into()))
}
