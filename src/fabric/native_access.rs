use super::*;

impl amdf_xdna_timer_record_t {
    /// Combine a completed RECORD_TIMER result's high/low words. This performs
    /// no clock conversion and establishes no epoch relation to other records.
    pub fn timestamp_ticks(&self) -> u64 {
        (u64::from(self.timestamp.high) << 32) | u64::from(self.timestamp.low)
    }
}

/// Borrowed, version-checked libamdf tables for advanced native integrations.
///
/// The raw types and constants are in [`native`]. Calling these tables is
/// unsafe: follow each function's ownership, synchronization and failure
/// contracts. In particular, failed removal can require leaking backing;
/// dropping an owner is not proof that native users have retired. Tables and
/// borrowed handles must not outlive their Rust owners. Do not destroy or
/// mutate Rust-owned handles through this interface.
pub struct NativeApi<'a> {
    api: &'a Api,
}
impl NativeApi<'_> {
    /// Core ABI, including reservation and queue-ordered memory updates.
    pub fn core(&self) -> &amdf_api_t {
        self.api.core
    }
    /// GPU extension, including kernel queues and device clock sampling.
    pub fn gpu(&self) -> &amdf_gpu_api_t {
        self.api.gpu
    }
    /// XDNA extension, including ordered command arrays and context results.
    pub fn xdna(&self) -> &amdf_xdna_api_t {
        self.api.xdna
    }
    /// Shared profiling host clock. Negotiation initializes no device.
    pub fn observation(&self) -> Result<&amdf_observation_api_t> {
        let mut raw = ptr::null();
        check("query observation extension", unsafe {
            entry!(self.api.core, query_extension)(
                AMDF_EXTENSION_OBSERVATION,
                AMDF_OBSERVATION_EXTENSION_VERSION_LATEST,
                AMDF_OBSERVATION_EXTENSION_VERSION_LATEST,
                &mut raw,
            )
        })?;
        if raw.is_null() {
            return Err(missing("observation extension"));
        }
        unsafe {
            validate_table(
                raw,
                size_of::<amdf_observation_api_t>(),
                AMDF_OBSERVATION_EXTENSION_VERSION_LATEST,
            )?;
            Ok(&*raw.cast())
        }
    }
}
impl Fabric {
    /// Access version-checked native services while this fabric remains alive.
    pub fn native_api(&self) -> NativeApi<'_> {
        NativeApi { api: &self.0.api }
    }
    /// Borrow the instance handle. Never transfer ownership or destroy it.
    pub fn native_handle(&self) -> *mut amdf_instance_t {
        self.0.raw
    }
}
impl Device {
    /// Borrow the live device handle. Never transfer ownership or destroy it.
    pub fn native_handle(&self) -> *mut amdf_device_t {
        self.0.raw
    }
}
impl Endpoint {
    /// Borrow the discovery endpoint. It remains owned by this endpoint.
    pub fn native_handle(&self) -> *mut amdf_endpoint_t {
        self.0.raw
    }
}
impl Buffer {
    /// Borrow native backing, including its established device access ordinals.
    /// External users must retain this buffer and exclude host reads/writes
    /// while native work is live; raw submissions do not acquire Rust leases.
    pub fn native_handle(&self) -> *mut amdf_memory_t {
        self.0.raw
    }
}
