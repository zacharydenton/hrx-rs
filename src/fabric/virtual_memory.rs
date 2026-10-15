use super::*;

/// One host or live-device consumer of stable virtual addresses.
pub struct ReservationAccess<'a> {
    /// None selects the process host address space.
    pub device: Option<&'a Device>,
    /// Consumer-local address domain; ignored for the host.
    pub address_domain: u32,
    /// Maximum READ/WRITE/EXECUTE permissions, using [`native`] constants.
    pub access: u32,
    /// Mapping offset/length unit, or zero for provider policy.
    pub mapping_granularity: u64,
}
/// One replacement (Some backing) or hole (None) in a reservation.
pub struct MappingUpdate<'a> {
    /// Destination byte offset.
    pub target_offset: u64,
    /// Nonzero mapped/detached length.
    pub bytes: u64,
    /// Prepared backing retained until checked whole-group detachment.
    pub buffer: Option<&'a Buffer>,
    /// Backing's access ordinal for the exact device; zero for host or holes.
    pub source_access: u32,
    /// Installed permissions; zero for holes.
    pub access: u32,
    /// Source logical byte offset; zero for holes.
    pub source_offset: u64,
}
/// Stable addresses with replaceable backing and conservative ownership.
///
/// All old and proposed sources stay leased until a successful whole-group
/// detach or destruction. Failed updates may partially map and poison the
/// native group. Failed destruction quarantines backing and owners instead of
/// releasing potentially live pages. GPU work and external aliases must be
/// retired by the caller before updates or destruction.
pub struct VirtualMemory {
    fabric: Fabric,
    devices: Vec<Device>,
    raw: *mut amdf_memory_reservation_t,
    groups: Vec<u32>,
    backing: Vec<Vec<memory::DeviceUse>>,
}
// Native queries are thread-safe and immutable. Mutation requires &mut self
// plus the unsafe caller's exclusion of external users; owners are retained.
unsafe impl Send for VirtualMemory {}
unsafe impl Sync for VirtualMemory {}
impl Fabric {
    /// Passive OS reservation geometry; allocates no backing or address space.
    pub fn host_reservation_capabilities(&self) -> Result<amdf_memory_reservation_capabilities_t> {
        let mut info = capabilities();
        check(
            "instance_query_host_memory_reservation_capabilities",
            unsafe {
                entry!(
                    self.0.api.core,
                    instance_query_host_memory_reservation_capabilities
                )(self.0.raw, &mut info)
            },
        )?;
        Ok(info)
    }
    /// Reserve stable addresses without allocating backing. Consumer order
    /// defines the access ordinals used by mapping updates and queries.
    pub fn reserve_virtual_memory(
        &self,
        bytes: u64,
        alignment: u64,
        accesses: &[ReservationAccess<'_>],
    ) -> Result<VirtualMemory> {
        let count = u32::try_from(accesses.len())
            .map_err(|_| Error::Message("too many reservation consumers".into()))?;
        let mut devices = Vec::new();
        let mut native = Vec::new();
        for access in accesses {
            if let Some(device) = access.device {
                if !Arc::ptr_eq(&self.0, &device.0.endpoint.0.instance) {
                    return Err(Error::Message(
                        "reservation device belongs to another fabric".into(),
                    ));
                }
                devices.push(device.clone());
            }
            native.push(amdf_memory_reservation_access_t {
                device: access.device.map_or(ptr::null_mut(), |d| d.0.raw),
                address_domain_ordinal: if access.device.is_some() {
                    access.address_domain
                } else {
                    AMDF_ADDRESS_DOMAIN_ORDINAL_NONE
                },
                access: access.access,
                mapping_granularity: access.mapping_granularity,
            });
        }
        let options = amdf_memory_reservation_create_info_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
            structure_size: size_of::<amdf_memory_reservation_create_info_t>() as u32,
            byte_length: bytes,
            minimum_alignment: alignment,
            access_count: count,
            accesses: native.as_ptr(),
            ..Default::default()
        };
        let _ = entry!(self.0.api.core, memory_reservation_destroy);
        let mut raw = ptr::null_mut();
        check("memory_reservation_create", unsafe {
            entry!(self.0.api.core, memory_reservation_create)(self.0.raw, &options, &mut raw)
        })?;
        if raw.is_null() {
            return Err(missing("created memory reservation"));
        }
        let mut reservation = VirtualMemory {
            fabric: self.clone(),
            devices,
            raw,
            groups: Vec::new(),
            backing: (0..count).map(|_| Vec::new()).collect(),
        };
        for ordinal in 0..count {
            let group = reservation.access_info(ordinal)?.mapping_access_ordinal;
            if group >= count {
                return Err(Error::Message("invalid reservation mapping group".into()));
            }
            reservation.groups.push(group);
        }
        Ok(reservation)
    }
}
impl Device {
    /// Cached live-device virtual-memory geometry for one address domain.
    pub fn reservation_capabilities(
        &self,
        domain: u32,
    ) -> Result<amdf_memory_reservation_capabilities_t> {
        let mut info = capabilities();
        check("device_query_memory_reservation_capabilities", unsafe {
            entry!(
                self.0.endpoint.0.instance.api.core,
                device_query_memory_reservation_capabilities
            )(self.0.raw, domain, &mut info)
        })?;
        Ok(info)
    }
}
fn capabilities() -> amdf_memory_reservation_capabilities_t {
    amdf_memory_reservation_capabilities_t {
        type_: AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
        structure_size: size_of::<amdf_memory_reservation_capabilities_t>() as u32,
        ..Default::default()
    }
}
impl VirtualMemory {
    /// Immutable geometry, stable address and alias-group identity. An address
    /// can be queried while unbacked; that never permits dereferencing a hole.
    pub fn access_info(&self, ordinal: u32) -> Result<amdf_memory_reservation_access_info_t> {
        let mut info = amdf_memory_reservation_access_info_t {
            type_: AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
            structure_size: size_of::<amdf_memory_reservation_access_info_t>() as u32,
            ..Default::default()
        };
        check("memory_reservation_query_access_info", unsafe {
            entry!(self.fabric.0.api.core, memory_reservation_query_access_info)(
                self.raw, ordinal, &mut info,
            )
        })?;
        Ok(info)
    }
    /// Borrow the reservation for advanced native APIs. External mappings must
    /// independently retain all old/proposed backing and track checked removal.
    pub fn native_handle(&self) -> *mut amdf_memory_reservation_t {
        self.raw
    }
    /// Replace ranges synchronously. Host Buffer access remains blocked while
    /// any source is retained, including sources replaced by a later update.
    ///
    /// # Safety
    /// Retire all users of affected aliases, serialize native mapping changes,
    /// and retain this reservation through external uses. On Linux with foreign
    /// KFD clients, serialize process-wide VM mutation as required by libamdf.
    pub unsafe fn update(&mut self, ordinal: u32, updates: &[MappingUpdate<'_>]) -> Result<()> {
        let group = *self
            .groups
            .get(ordinal as usize)
            .ok_or_else(|| Error::Message("invalid reservation access ordinal".into()))?
            as usize;
        let count = u32::try_from(updates.len())
            .map_err(|_| Error::Message("too many mapping updates".into()))?;
        let mut native = Vec::new();
        let mut leases = Vec::new();
        for update in updates {
            if let Some(buffer) = update.buffer {
                if !Arc::ptr_eq(&self.fabric.0, &buffer.0.instance) {
                    return Err(Error::Message(
                        "mapping source belongs to another fabric".into(),
                    ));
                }
                leases.push(buffer.retain_use()?);
            }
            native.push(amdf_memory_mapping_update_t {
                target_byte_offset: update.target_offset,
                byte_length: update.bytes,
                memory: update.buffer.map_or(ptr::null_mut(), |b| b.0.raw),
                memory_access_ordinal: update.source_access,
                access: update.access,
                source_byte_offset: update.source_offset,
            });
        }
        let update = entry!(self.fabric.0.api.core, memory_reservation_update);
        // Retain proposals BEFORE entering a call that may partially install them.
        self.backing[group].extend(leases);
        check("memory_reservation_update", unsafe {
            update(self.raw, ordinal, count, native.as_ptr())
        })
    }
    /// Remove an entire alias group and release its backing after checked removal.
    /// A previously poisoned native group remains poisoned after successful detach.
    ///
    /// # Safety
    /// Retire every user and queued mapping operation for this alias group first.
    pub unsafe fn detach(&mut self, ordinal: u32) -> Result<()> {
        let group = *self
            .groups
            .get(ordinal as usize)
            .ok_or_else(|| Error::Message("invalid reservation access ordinal".into()))?
            as usize;
        check("memory_reservation_detach", unsafe {
            entry!(self.fabric.0.api.core, memory_reservation_detach)(self.raw, ordinal)
        })?;
        self.backing[group].clear();
        Ok(())
    }
    /// Destroy with a checked result. Failure consumes the handle and quarantines
    /// retained resources. Drop applies the same policy but cannot return errors.
    ///
    /// # Safety
    /// Retire all external users, host views and mapping operations first.
    pub unsafe fn destroy(mut self) -> Result<()> {
        self.release()
    }
    fn release(&mut self) -> Result<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        let raw = std::mem::replace(&mut self.raw, ptr::null_mut());
        let status = unsafe { self.fabric.0.api.core.memory_reservation_destroy.unwrap()(raw) };
        if status != 0 {
            std::mem::forget(self.fabric.clone());
            std::mem::forget(std::mem::take(&mut self.devices));
            std::mem::forget(std::mem::take(&mut self.backing));
        }
        check("memory_reservation_destroy", status)
    }
}
impl Drop for VirtualMemory {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
