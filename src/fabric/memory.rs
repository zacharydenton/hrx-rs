use super::*;
use std::sync::Mutex;

/// Shared backing with explicit device access and checked host transfers.
#[derive(Clone)]
pub struct Buffer(pub(super) Arc<Memory>);
pub(super) struct Memory {
    instance: Arc<Instance>,
    devices: Vec<Device>,
    pub(super) raw: *mut amdf_memory_t,
    mapping: *mut amdf_host_mapping_t,
    pub(super) pointer: *mut u8,
    bytes: usize,
    needs_cache: Arc<std::sync::atomic::AtomicBool>,
    source: Option<Buffer>,
    reservation: Option<Arc<crate::residency::MemoryReservation>>,
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
            }
        }
    }
}
impl Buffer {
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
    source: Option<Buffer>,
    reservation: Option<Arc<crate::residency::MemoryReservation>>,
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
                source: None,
                reservation: Some(reservation),
            },
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
                source: Some(source.clone()),
                reservation: None,
            },
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
    ) -> Result<Buffer> {
        if bytes == 0 {
            return Err(Error::Message("allocation must be nonempty".into()));
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
                        ptr::write_bytes(memory.pointer, 0, bytes);
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
