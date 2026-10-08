use super::*;
use std::sync::Mutex;

/// Shared backing with explicit device access and checked host transfers.
#[derive(Clone)]
pub struct Buffer(pub(super) Arc<Memory>);
pub(super) struct Memory {
    pub(super) instance: Arc<Instance>,
    pub(super) devices: Vec<Device>,
    pub(super) raw: *mut amdf_memory_t,
    pub(super) mapping: *mut amdf_host_mapping_t,
    pub(super) pointer: *mut u8,
    bytes: usize,
    needs_cache: Arc<std::sync::atomic::AtomicBool>,
    pub(super) source: Option<Buffer>,
    pub(super) reservation: Option<Arc<crate::residency::MemoryReservation>>,
    host_storage: Option<Arc<memmap2::MmapMut>>,
    // Submission leases and host access serialize through this lock. A lease
    // lasts through terminal completion, not just command publication.
    pub(super) uses: Arc<Mutex<usize>>,
}
// The immutable mapping is accessed only under uses; commands must acquire a
// use lease before touching backing. Native metadata queries are thread-safe.
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}
impl Drop for Memory {
    fn drop(&mut self) {
        let api = self.instance.api.core;
        unsafe {
            let mapping_released =
                self.mapping.is_null() || api.host_mapping_destroy.unwrap()(self.mapping) == 0;
            if !mapping_released || api.memory_destroy.unwrap()(self.raw) != 0 {
                // Native detach failed: preserve the required owner lifetime.
                std::mem::forget(self.instance.clone());
                std::mem::forget(self.devices.clone());
                std::mem::forget(self.source.take());
                std::mem::forget(self.reservation.take());
                std::mem::forget(self.host_storage.take());
            }
        }
    }
}
impl Buffer {
    pub(super) fn owns_registered_pages(&self) -> bool {
        self.0.host_storage.is_some()
    }

    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    pub(crate) fn same_backing(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn exclusively_owned(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
    pub(crate) fn zero(&self) -> Result<()> {
        let uses = self
            .0
            .uses
            .lock()
            .map_err(|_| Error::DeviceLost("memory ownership poisoned".into()))?;
        if *uses != 0 {
            return Err(Error::Busy("memory is executing".into()));
        }
        unsafe {
            ptr::write_bytes(self.host_pointer(), 0, self.len());
            self.cache_control(true, 0, self.len())
        }
    }

    pub(super) fn retain_use(&self) -> Result<DeviceUse> {
        let mut uses = self
            .0
            .uses
            .lock()
            .map_err(|_| Error::DeviceLost("memory ownership poisoned".into()))?;
        *uses = uses
            .checked_add(1)
            .ok_or_else(|| Error::Busy("memory lease limit".into()))?;
        Ok(DeviceUse(self.clone()))
    }
    // Only privately owned fence buffers use this path while a device writes.
    pub(super) fn fence_value(&self) -> Result<u32> {
        if self.len() < 4 {
            return Err(Error::Message("completion storage is too small".into()));
        }
        unsafe {
            // Fence allocations have only a HOST_COHERENT GPU consumer. GPU
            // release packets and the acquire load establish visibility.
            if self
                .0
                .needs_cache
                .load(std::sync::atomic::Ordering::Acquire)
            {
                return Err(Error::Message(
                    "completion fence must be GPU-coherent only".into(),
                ));
            }
            Ok(self
                .0
                .pointer
                .cast::<std::sync::atomic::AtomicU32>()
                .as_ref()
                .unwrap()
                .load(std::sync::atomic::Ordering::Acquire))
        }
    }
    pub(crate) fn host_pointer(&self) -> *mut u8 {
        self.0.pointer
    }
    /// Publish or acquire a mapped range under an external scheduling lease.
    ///
    /// # Safety
    /// The caller must exclude conflicting host and device access until this
    /// cache operation completes, and retain this allocation through that use.
    pub unsafe fn cache_control(&self, flush: bool, offset: usize, length: usize) -> Result<()> {
        self.range(offset, length)?;
        // Owned imports retain the original mapping as the maintenance site
        // for these exact physical bytes, including its cache-policy contract.
        if let Some(source) = &self.0.source {
            return unsafe { source.cache_control(flush, offset, length) };
        }
        if !self
            .0
            .needs_cache
            .load(std::sync::atomic::Ordering::Acquire)
        {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            return Ok(());
        }
        check("explicit mapped visibility", unsafe {
            entry!(self.0.instance.api.core, host_mapping_cache_control)(
                self.0.mapping,
                if flush {
                    AMDF_HOST_CACHE_OPERATION_FLUSH
                } else {
                    AMDF_HOST_CACHE_OPERATION_INVALIDATE
                },
                offset as u64,
                length as u64,
            )
        })
    }
    /// Logical byte length; allocation rounding grants no additional access.
    pub fn len(&self) -> usize {
        self.0.bytes
    }
    /// Allocations are always nonempty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn range(&self, offset: usize, length: usize) -> Result<()> {
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.len())
        {
            return Err(Error::Message("buffer range is out of bounds".into()));
        }
        Ok(())
    }
    /// Copy and publish host bytes. In-flight device use returns Busy.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.range(offset, bytes.len())?;
        let uses = self
            .0
            .uses
            .lock()
            .map_err(|_| Error::DeviceLost("memory ownership poisoned".into()))?;
        if *uses != 0 {
            return Err(Error::Busy("buffer is in use by a device".into()));
        }
        unsafe {
            // A partial host write must preserve bytes in the same cache line
            // that the GPU may have changed since the last host read.
            self.cache_control(false, offset, bytes.len())?;
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.0.pointer.add(offset), bytes.len());
            self.cache_control(true, offset, bytes.len())
        }
    }
    /// Acquire and copy completed device results. In-flight use returns Busy.
    pub fn read(&self, offset: usize, bytes: &mut [u8]) -> Result<()> {
        self.range(offset, bytes.len())?;
        let uses = self
            .0
            .uses
            .lock()
            .map_err(|_| Error::DeviceLost("memory ownership poisoned".into()))?;
        if *uses != 0 {
            return Err(Error::Busy("buffer is in use by a device".into()));
        }
        unsafe {
            self.cache_control(false, offset, bytes.len())?;
            ptr::copy_nonoverlapping(self.0.pointer.add(offset), bytes.as_mut_ptr(), bytes.len());
        }
        Ok(())
    }
    // Private runtime state needs one indivisible host transaction against
    // submission admission; a separate read/write pair would race new uses.
    pub(super) fn with_host_bytes<T>(
        &self,
        action: impl FnOnce(&mut [u8]) -> Result<T>,
    ) -> Result<T> {
        let uses = self
            .0
            .uses
            .lock()
            .map_err(|_| Error::DeviceLost("memory ownership poisoned".into()))?;
        if *uses != 0 {
            return Err(Error::Busy("buffer is in use by a device".into()));
        }
        unsafe {
            self.cache_control(false, 0, self.len())?;
            let result = action(std::slice::from_raw_parts_mut(self.0.pointer, self.len()));
            self.cache_control(true, 0, self.len())?;
            result
        }
    }
    /// Address in an explicitly admitted device's primary data interface.
    /// The address is valid only while this buffer and device remain alive.
    pub fn device_address(&self, device: &Device) -> Result<u64> {
        let index = self
            .0
            .devices
            .iter()
            .position(|d| Arc::ptr_eq(&d.0, &device.0));
        let Some(index) = index else {
            return self.0.source.as_ref().map_or_else(
                || Err(Error::Message("device has no access to this buffer".into())),
                |source| source.device_address(device),
            );
        };
        let kind = match device.endpoint().engine() {
            Engine::Gpu => AMDF_MEMORY_ADDRESS_GPU,
            Engine::Xdna => AMDF_MEMORY_ADDRESS_XDNA_DMA,
        };
        let mut address = 0;
        check("memory_query_address", unsafe {
            entry!(self.0.instance.api.core, memory_query_address)(
                self.0.raw,
                index as u32,
                kind,
                &mut address,
            )
        })?;
        Ok(address)
    }
}

pub(super) struct DeviceUse(Buffer);
impl Drop for DeviceUse {
    fn drop(&mut self) {
        let mut uses = self.0.0.uses.lock().unwrap_or_else(|e| e.into_inner());
        *uses -= 1;
    }
}

#[derive(Default)]
struct MemoryOwners {
    coherent: bool,
    profile: Option<(*mut amdf_memory_scope_t, u32)>,
    source: Option<Buffer>,
    reservation: Option<Arc<crate::residency::MemoryReservation>>,
    host_storage: Option<Arc<memmap2::MmapMut>>,
}
struct ExternalMemory {
    api: Arc<Api>,
    value: amdf_external_memory_t,
}
impl Drop for ExternalMemory {
    fn drop(&mut self) {
        unsafe {
            self.api.core.external_memory_release.unwrap()(&mut self.value);
        }
    }
}
impl Fabric {
    /// Allocate page-rounded anonymous host storage and register device access.
    /// The mapping and optional budget charge remain owned by every buffer alias,
    /// pending device use, and native-detach quarantine. Useful for kernel I/O
    /// registration requiring ordinary host pages rather than device allocations.
    /// Linux KFD requires [`NativeLifetime::Process`] for this construction role.
    pub fn allocate_registered(
        &self,
        bytes: usize,
        devices: &[Device],
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Buffer> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 || bytes == 0 {
            return Err(Error::Message("invalid registered allocation size".into()));
        }
        let page = page as usize;
        let bytes = bytes
            .checked_add(page - 1)
            .map(|v| v / page * page)
            .ok_or_else(|| Error::Message("registered allocation overflow".into()))?;
        let reservation = budget.map(|b| b.reserve(bytes).map(Arc::new)).transpose()?;
        let mut mapping = memmap2::MmapOptions::new().len(bytes).map_anon()?;
        let pointer = mapping.as_mut_ptr();
        self.create_memory(
            bytes,
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            page as u64,
            Some(pointer.cast()),
            MemoryOwners {
                coherent: true,
                reservation,
                host_storage: Some(Arc::new(mapping)),
                ..Default::default()
            },
            None,
        )
    }
    /// Allocate GPU-coherent host storage for direct host access and polling.
    /// Host and device accesses must still be ordered by completion.
    pub fn allocate_shared(&self, bytes: usize, devices: &[Device]) -> Result<Buffer> {
        self.create_memory(
            bytes,
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            None,
            MemoryOwners {
                coherent: true,
                ..Default::default()
            },
            None,
        )
    }
    /// Allocate host-visible system backing with access for every listed device.
    /// Devices must belong to this fabric; no implicit device activation occurs.
    pub fn allocate(&self, bytes: usize, devices: &[Device]) -> Result<Buffer> {
        self.allocate_access(
            bytes,
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
        )
    }
    /// Allocate and publish a nonempty host byte slice in one initialization pass.
    /// The returned storage owns a copy; the input may be released immediately.
    pub fn allocate_from(&self, data: &[u8], devices: &[Device]) -> Result<Buffer> {
        self.create_memory(
            data.len(),
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            None,
            MemoryOwners::default(),
            Some(data),
        )
    }
    /// Allocate backing under a shared residency ceiling. Clones and in-flight
    /// native owners retain the charge until safe destruction or quarantine.
    pub fn allocate_budgeted(
        &self,
        bytes: usize,
        devices: &[Device],
        budget: &crate::residency::MemoryBudget,
    ) -> Result<Buffer> {
        let reservation = Arc::new(budget.reserve(bytes)?);
        self.create_memory(
            bytes,
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            64,
            None,
            MemoryOwners {
                coherent: false,
                profile: None,
                source: None,
                host_storage: None,
                reservation: Some(reservation),
            },
            None,
        )
    }
    pub(super) fn allocate_access(
        &self,
        bytes: usize,
        devices: &[Device],
        access: u32,
        alignment: u64,
    ) -> Result<Buffer> {
        self.create_memory(
            bytes,
            devices,
            access,
            alignment,
            None,
            MemoryOwners::default(),
            None,
        )
    }
    pub(super) fn allocate_owned(
        &self,
        bytes: usize,
        device: &Device,
        access: u32,
        alignment: u64,
        coherent: bool,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Buffer> {
        let reservation = budget.map(|b| b.reserve(bytes).map(Arc::new)).transpose()?;
        self.create_memory(
            bytes,
            std::slice::from_ref(device),
            access,
            alignment,
            None,
            MemoryOwners {
                coherent,
                profile: None,
                source: None,
                reservation,
                host_storage: None,
            },
            None,
        )
    }
    #[cfg(feature = "npu")]
    pub(crate) fn share_owned(&self, source: &Buffer, devices: &[Device]) -> Result<Buffer> {
        let additional: Vec<_> = devices
            .iter()
            .filter(|device| source.device_address(device).is_err())
            .cloned()
            .collect();
        if additional.is_empty() {
            return Ok(source.clone());
        }
        self.create_memory(
            source.len(),
            &additional,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            4096,
            None,
            MemoryOwners {
                coherent: false,
                profile: None,
                host_storage: None,
                source: Some(source.clone()),
                reservation: None,
            },
            None,
        )
    }

    /// Register caller-owned host pages with explicit device access.
    ///
    /// # Safety
    /// The aligned host range must stay allocated and stable until this buffer,
    /// its clones, and every accepted device use have been destroyed. Host access
    /// must be ordered against all device use, including native aliases.
    pub unsafe fn register_host(
        &self,
        pointer: *mut std::ffi::c_void,
        bytes: usize,
        devices: &[Device],
    ) -> Result<Buffer> {
        if pointer.is_null() {
            return Err(Error::Message("host registration pointer is null".into()));
        }
        self.create_memory(
            bytes,
            devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            4096,
            Some(pointer),
            MemoryOwners {
                coherent: true,
                ..Default::default()
            },
            None,
        )
    }
    fn create_memory(
        &self,
        bytes: usize,
        devices: &[Device],
        access: u32,
        alignment: u64,
        host: Option<*mut std::ffi::c_void>,
        owners: MemoryOwners,
        initial_data: Option<&[u8]>,
    ) -> Result<Buffer> {
        if bytes == 0 {
            return Err(Error::Message("allocation must be nonempty".into()));
        }
        if initial_data
            .is_some_and(|data| data.len() != bytes || host.is_some() || owners.source.is_some())
        {
            return Err(Error::Message(
                "initial data must cover exactly one new allocation".into(),
            ));
        }
        for (index, device) in devices.iter().enumerate() {
            if !Arc::ptr_eq(&self.0, &device.0.endpoint.0.instance)
                || devices[..index]
                    .iter()
                    .any(|other| Arc::ptr_eq(&other.0, &device.0))
            {
                return Err(Error::Message(
                    "allocation devices must be unique and belong to this fabric".into(),
                ));
            }
        }
        let api = self.0.api.core;
        let _ = entry!(api, memory_destroy);
        let _ = entry!(api, host_mapping_destroy);
        let _ = entry!(api, host_mapping_cache_control);
        let mut external = if let Some(source) = &owners.source {
            let _ = entry!(api, external_memory_release);
            let mut external = ExternalMemory {
                api: self.0.api.clone(),
                value: Default::default(),
            };
            let options = amdf_memory_export_info_t {
                type_: AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
                structure_size: size_of::<amdf_memory_export_info_t>() as u32,
                external_memory_type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
                byte_length: bytes as u64,
                ..Default::default()
            };
            check("memory_export", unsafe {
                entry!(api, memory_export)(source.0.raw, &options, &mut external.value)
            })?;
            Some(external)
        } else {
            None
        };
        let accesses: Vec<_> = devices
            .iter()
            .map(|device| amdf_memory_device_access_t {
                device: device.0.raw,
                requirements: amdf_memory_access_requirements_t {
                    access,
                    flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS as u64
                        | if owners.coherent && device.endpoint().engine() == Engine::Gpu {
                            AMDF_MEMORY_FLAG_HOST_COHERENT as u64
                        } else {
                            0
                        },
                    address_kinds: 1
                        << match device.endpoint().engine() {
                            Engine::Gpu => AMDF_MEMORY_ADDRESS_GPU,
                            Engine::Xdna => AMDF_MEMORY_ADDRESS_XDNA_DMA,
                        },
                    ..Default::default()
                },
            })
            .collect();
        let mut count = 0;
        let status = unsafe {
            entry!(api, instance_enumerate_memory_scopes)(
                self.0.raw,
                0,
                ptr::null_mut(),
                &mut count,
            )
        };
        if status != AMDF_STATUS_CODE_BUFFER_TOO_SMALL as u64 {
            check("enumerate memory scopes", status)?;
        }
        let mut scopes = vec![ptr::null_mut(); count as usize];
        check("enumerate memory scopes", unsafe {
            entry!(api, instance_enumerate_memory_scopes)(
                self.0.raw,
                count,
                scopes.as_mut_ptr(),
                &mut count,
            )
        })?;
        for scope in scopes {
            let mut scope_info = amdf_memory_scope_info_t {
                type_: AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
                structure_size: size_of::<amdf_memory_scope_info_t>() as u32,
                ..Default::default()
            };
            check("memory_scope_query_info", unsafe {
                entry!(api, memory_scope_query_info)(scope, &mut scope_info)
            })?;
            if scope_info.kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM {
                continue;
            }
            for ordinal in 0..scope_info.memory_profile_count {
                if owners
                    .profile
                    .is_some_and(|selected| selected != (scope, ordinal))
                {
                    continue;
                }
                let mut profile = amdf_memory_profile_t {
                    type_: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                    structure_size: size_of::<amdf_memory_profile_t>() as u32,
                    ..Default::default()
                };
                let mut capabilities = vec![
                    amdf_memory_access_capabilities_t {
                        type_: AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
                        structure_size: size_of::<amdf_memory_access_capabilities_t>() as u32,
                        ..Default::default()
                    };
                    devices.len()
                ];
                let status = unsafe {
                    entry!(api, memory_scope_query_device_profile)(
                        scope,
                        ordinal,
                        accesses.len() as u32,
                        accesses.as_ptr(),
                        &mut profile,
                        capabilities.as_mut_ptr(),
                    )
                };
                if status == AMDF_STATUS_CODE_UNSUPPORTED as u64 {
                    continue;
                }
                check("memory_scope_query_device_profile", status)?;
                let role = if external.is_some() {
                    AMDF_MEMORY_PROFILE_ROLE_IMPORT
                } else if host.is_some() {
                    AMDF_MEMORY_PROFILE_ROLE_REGISTER
                } else {
                    AMDF_MEMORY_PROFILE_ROLE_CREATE
                };
                let construction = if external.is_some() {
                    &profile.import
                } else if host.is_some() {
                    &profile.registration
                } else {
                    &profile.allocation
                };
                if profile.roles & (role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) as u64
                    != (role | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) as u64
                    || profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE as u64 == 0
                    || construction.maximum_byte_length < bytes as u64
                {
                    continue;
                }
                let options = amdf_memory_create_info_t {
                    type_: AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
                    structure_size: size_of::<amdf_memory_create_info_t>() as u32,
                    memory_profile_ordinal: ordinal,
                    access_count: accesses.len() as u32,
                    required_flags: AMDF_MEMORY_FLAG_HOST_VISIBLE as u64,
                    byte_length: bytes as u64,
                    minimum_alignment: alignment,
                    accesses: accesses.as_ptr(),
                    registered_host_pointer: host.unwrap_or(ptr::null_mut()),
                    registered_host_cacheability: if host.is_some() {
                        AMDF_HOST_CACHEABILITY_WRITE_BACK
                    } else {
                        0
                    },
                    ..Default::default()
                };
                let mut raw = ptr::null_mut();
                if let Some(external) = &mut external {
                    let options = amdf_memory_import_info_t {
                        type_: AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
                        structure_size: size_of::<amdf_memory_import_info_t>() as u32,
                        memory_profile_ordinal: ordinal,
                        access_count: accesses.len() as u32,
                        required_flags: AMDF_MEMORY_FLAG_HOST_VISIBLE as u64,
                        minimum_alignment: alignment,
                        accesses: accesses.as_ptr(),
                        ..Default::default()
                    };
                    check("memory_import", unsafe {
                        entry!(api, memory_import)(scope, &options, &mut external.value, &mut raw)
                    })?;
                } else {
                    check("memory_create", unsafe {
                        entry!(api, memory_create)(scope, &options, &mut raw)
                    })?;
                }
                if raw.is_null() {
                    return Err(missing("created memory"));
                }
                let needs_cache = owners.source.as_ref().map_or_else(
                    || Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    |source| source.0.needs_cache.clone(),
                );
                if !owners.coherent
                    || devices.is_empty()
                    || devices
                        .iter()
                        .any(|device| device.endpoint().engine() != Engine::Gpu)
                {
                    needs_cache.store(true, std::sync::atomic::Ordering::Release);
                }
                let mut memory = Memory {
                    instance: self.0.clone(),
                    devices: devices.to_vec(),
                    raw,
                    mapping: ptr::null_mut(),
                    pointer: ptr::null_mut(),
                    bytes,
                    needs_cache,
                    source: owners.source.clone(),
                    reservation: owners.reservation.clone(),
                    host_storage: owners.host_storage.clone(),
                    uses: owners
                        .source
                        .as_ref()
                        .map_or_else(|| Arc::new(Mutex::new(0)), |source| source.0.uses.clone()),
                };
                let options = amdf_memory_map_info_t {
                    type_: AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
                    structure_size: size_of::<amdf_memory_map_info_t>() as u32,
                    byte_length: bytes as u64,
                    flags: AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                    ..Default::default()
                };
                check("memory_map", unsafe {
                    entry!(api, memory_map)(raw, &options, &mut memory.mapping)
                })?;
                let mut info = amdf_host_mapping_info_t {
                    type_: AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
                    structure_size: size_of::<amdf_host_mapping_info_t>() as u32,
                    ..Default::default()
                };
                check("host_mapping_query_info", unsafe {
                    entry!(api, host_mapping_query_info)(memory.mapping, &mut info)
                })?;
                if info.pointer.is_null() || info.byte_length < bytes as u64 {
                    return Err(missing("complete host mapping"));
                }
                memory.pointer = info.pointer.cast();
                unsafe {
                    if host.is_none() && owners.source.is_none() {
                        if let Some(data) = initial_data {
                            // Fresh private backing has no device-written cache
                            // lines to acquire. Initialize all bytes before the
                            // single publication below or exposing a safe Buffer.
                            ptr::copy_nonoverlapping(data.as_ptr(), memory.pointer, bytes);
                        } else {
                            ptr::write_bytes(memory.pointer, 0, bytes);
                        }
                    }
                }
                let buffer = Buffer(Arc::new(memory));
                unsafe {
                    buffer.cache_control(true, 0, bytes)?;
                }
                return Ok(buffer);
            }
        }
        Err(Error::Unsupported(
            "no system memory profile admits all requested consumers".into(),
        ))
    }
}

#[cfg(all(test, feature = "npu"))]
mod alias_tests {
    use super::*;
    #[test]
    #[ignore = "requires native compiler, gfx1151 and NPU5"]
    fn imported_aliases_share_host_leases_and_cache_requirements() -> Result<()> {
        let gpu = Device::open(Engine::Gpu, 0)?;
        let npu = Device::open(Engine::Xdna, 0)?;
        let fabric = gpu.fabric();
        let source = fabric.allocate(4096, std::slice::from_ref(&gpu))?;
        let alias = fabric.share_owned(&source, &[gpu, npu.clone()])?;
        assert!(
            source
                .0
                .needs_cache
                .load(std::sync::atomic::Ordering::Acquire)
        );
        let output = fabric.allocate(4096, std::slice::from_ref(&npu))?;
        let artifact = crate::loom::Compiler::for_target(None, npu.target())?
            .module(include_str!("../../tests/kernels/copy.xdna.loom"))
            .compile(&crate::loom::Specialization::new("copy").with_config("copy.packets", "1"))?;
        let program = unsafe {
            npu.prepare_xdna(
                &artifact,
                1,
                &[
                    XdnaBinding {
                        buffer: &alias,
                        offset: 0,
                        length: 4096,
                    },
                    XdnaBinding {
                        buffer: &output,
                        offset: 0,
                        length: 4096,
                    },
                ],
            )
        }?;
        for value in [0x53, 0x27, 0xb8] {
            source.write(0, &[value; 4096])?;
            let done = unsafe { program.dispatch() }?;
            assert!(matches!(source.write(0, &[0]), Err(Error::Busy(_))));
            assert!(matches!(alias.read(0, &mut [0]), Err(Error::Busy(_))));
            done.wait()?;
            let mut actual = [0; 4096];
            output.read(0, &mut actual)?;
            assert_eq!(actual, [value; 4096]);
        }
        drop(source);
        alias.write(0, &[0x4b; 4096])?;
        unsafe { program.dispatch() }?.wait()?;
        let mut actual = [0; 4096];
        output.read(0, &mut actual)?;
        assert_eq!(actual, [0x4b; 4096]);
        Ok(())
    }
}

/// One host-visible allocation profile and its complete device access set.
/// Queries allocate no backing; allocation uses this exact profile and flags.
/// Registration and external imports have separate native contracts and are not
/// admitted by this allocation-only plan.
#[derive(Clone)]
pub struct AllocationProfile {
    fabric: Fabric,
    devices: Vec<Device>,
    scope: *mut amdf_memory_scope_t,
    ordinal: u32,
    coherent: bool,
    limits: amdf_memory_construction_capabilities_t,
}
// Native scopes and immutable profile facts live with the retained instance.
unsafe impl Send for AllocationProfile {}
unsafe impl Sync for AllocationProfile {}
impl Fabric {
    /// Enumerate allocation profiles for a complete, unique set of live devices.
    /// `coherent_gpu` requires coherent GPU attachments; it does not infer NPU
    /// coherence or promise system atomic support.
    pub fn allocation_profiles(
        &self,
        devices: &[Device],
        coherent_gpu: bool,
    ) -> Result<Vec<AllocationProfile>> {
        for (i, device) in devices.iter().enumerate() {
            if !Arc::ptr_eq(&self.0, &device.0.endpoint.0.instance)
                || devices[..i]
                    .iter()
                    .any(|other| Arc::ptr_eq(&other.0, &device.0))
            {
                return Err(Error::Message(
                    "allocation devices must be unique and belong to this fabric".into(),
                ));
            }
        }
        let api = self.0.api.core;
        let accesses = profile_accesses(devices, coherent_gpu);
        let mut count = 0;
        let status = unsafe {
            entry!(api, instance_enumerate_memory_scopes)(
                self.0.raw,
                0,
                ptr::null_mut(),
                &mut count,
            )
        };
        if status != AMDF_STATUS_CODE_BUFFER_TOO_SMALL as u64 {
            check("enumerate memory scopes", status)?;
        }
        let mut scopes = vec![ptr::null_mut(); count as usize];
        check("enumerate memory scopes", unsafe {
            entry!(api, instance_enumerate_memory_scopes)(
                self.0.raw,
                count,
                scopes.as_mut_ptr(),
                &mut count,
            )
        })?;
        scopes.truncate(count as usize);
        let mut result = Vec::new();
        for scope in scopes {
            let mut info = amdf_memory_scope_info_t {
                type_: AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
                structure_size: size_of::<amdf_memory_scope_info_t>() as u32,
                ..Default::default()
            };
            check("memory_scope_query_info", unsafe {
                entry!(api, memory_scope_query_info)(scope, &mut info)
            })?;
            if info.kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM {
                continue;
            }
            for ordinal in 0..info.memory_profile_count {
                let mut profile = amdf_memory_profile_t {
                    type_: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                    structure_size: size_of::<amdf_memory_profile_t>() as u32,
                    ..Default::default()
                };
                let mut capabilities = vec![
                    amdf_memory_access_capabilities_t {
                        type_: AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
                        structure_size: size_of::<amdf_memory_access_capabilities_t>() as u32,
                        ..Default::default()
                    };
                    devices.len()
                ];
                let status = unsafe {
                    entry!(api, memory_scope_query_device_profile)(
                        scope,
                        ordinal,
                        accesses.len() as u32,
                        accesses.as_ptr(),
                        &mut profile,
                        capabilities.as_mut_ptr(),
                    )
                };
                if status == AMDF_STATUS_CODE_UNSUPPORTED as u64 {
                    continue;
                }
                check("memory_scope_query_device_profile", status)?;
                let roles =
                    (AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) as u64;
                if profile.roles & roles != roles
                    || profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE as u64 == 0
                {
                    continue;
                }
                result.push(AllocationProfile {
                    fabric: self.clone(),
                    devices: devices.to_vec(),
                    scope,
                    ordinal,
                    coherent: coherent_gpu,
                    limits: profile.allocation,
                });
            }
        }
        Ok(result)
    }
}
fn profile_accesses(devices: &[Device], coherent: bool) -> Vec<amdf_memory_device_access_t> {
    devices
        .iter()
        .map(|device| amdf_memory_device_access_t {
            device: device.0.raw,
            requirements: amdf_memory_access_requirements_t {
                access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                flags: AMDF_MEMORY_FLAG_DEVICE_ADDRESS as u64
                    | if coherent && device.endpoint().engine() == Engine::Gpu {
                        AMDF_MEMORY_FLAG_HOST_COHERENT as u64
                    } else {
                        0
                    },
                address_kinds: 1
                    << match device.endpoint().engine() {
                        Engine::Gpu => AMDF_MEMORY_ADDRESS_GPU,
                        Engine::Xdna => AMDF_MEMORY_ADDRESS_XDNA_DMA,
                    },
                ..Default::default()
            },
        })
        .collect()
}
impl AllocationProfile {
    /// Largest logical allocation admitted by this profile.
    pub fn maximum_bytes(&self) -> u64 {
        self.limits.maximum_byte_length
    }
    /// Required logical allocation length granularity.
    pub fn byte_granularity(&self) -> u64 {
        self.limits.byte_length_granularity
    }
    /// Allocate under precisely the queried construction contract.
    pub fn allocate(
        &self,
        bytes: usize,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Buffer> {
        if bytes == 0
            || bytes as u64 > self.maximum_bytes()
            || !((bytes as u64).is_multiple_of(self.byte_granularity().max(1)))
        {
            return Err(Error::Message(
                "allocation does not satisfy the selected profile's size limits".into(),
            ));
        }
        self.fabric.create_memory(
            bytes,
            &self.devices,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            self.limits.minimum_alignment,
            None,
            MemoryOwners {
                coherent: self.coherent,
                profile: Some((self.scope, self.ordinal)),
                source: None,
                host_storage: None,
                reservation: budget.map(|b| b.reserve(bytes).map(Arc::new)).transpose()?,
            },
            None,
        )
    }
    /// Qualify a directional visibility relation before allocating any backing.
    /// Returned facts retain this profile and all devices. They describe queue
    /// sites, not cache instructions executed within a shader.
    pub fn visibility(
        &self,
        producer: MemorySite<'_>,
        consumer: MemorySite<'_>,
    ) -> Result<ProspectiveVisibility> {
        let site = |site: MemorySite<'_>| -> Result<amdf_memory_profile_site_t> {
            let mut result = amdf_memory_profile_site_t::default();
            match site {
                MemorySite::Host => {
                    result.kind = AMDF_MEMORY_SITE_KIND_HOST;
                    result.value.host_access =
                        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
                }
                MemorySite::Device(device, family) => {
                    let ordinal = self
                        .devices
                        .iter()
                        .position(|d| Arc::ptr_eq(&d.0, &device.0))
                        .ok_or_else(|| {
                            Error::Message(
                                "visibility device is absent from allocation profile".into(),
                            )
                        })?;
                    result.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
                    result.value.device = amdf_memory_profile_site_t__bindgen_ty_1__bindgen_ty_1 {
                        access_ordinal: ordinal as u32,
                        queue_family_ordinal: family,
                    };
                }
            }
            Ok(result)
        };
        let accesses = profile_accesses(&self.devices, self.coherent);
        let query = amdf_memory_profile_pair_query_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
            structure_size: size_of::<amdf_memory_profile_pair_query_t>() as u32,
            memory_profile_ordinal: self.ordinal,
            access_count: accesses.len() as u32,
            accesses: accesses.as_ptr(),
            required_flags: AMDF_MEMORY_FLAG_HOST_VISIBLE as u64,
            producer: site(producer)?,
            consumer: site(consumer)?,
            ..Default::default()
        };
        let mut info = amdf_memory_pair_info_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            structure_size: size_of::<amdf_memory_pair_info_t>() as u32,
            ..Default::default()
        };
        check("memory_scope_query_pair_info", unsafe {
            entry!(self.fabric.0.api.core, memory_scope_query_pair_info)(
                self.scope, &query, &mut info,
            )
        })?;
        Ok(ProspectiveVisibility {
            _profile: self.clone(),
            facts: visibility::VisibilityFacts::from_native(info)?,
        })
    }
}
/// Prospective facts retaining their allocation profile and complete device set.
pub struct ProspectiveVisibility {
    _profile: AllocationProfile,
    facts: visibility::VisibilityFacts,
}
impl std::ops::Deref for ProspectiveVisibility {
    type Target = visibility::VisibilityFacts;
    fn deref(&self) -> &Self::Target {
        &self.facts
    }
}
