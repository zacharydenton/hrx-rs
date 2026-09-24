//! Native AMD device ownership through libamdf's versioned C ABI.
//! Discovery is passive; only opening a device activates hardware.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::all
)]
mod ffi;
use crate::{Error, Result, Target};
use ffi::*;
use std::{ffi::CStr, path::Path, ptr, sync::Arc};

fn check(operation: &'static str, status: amdf_status_t) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let domain = (status >> 32) as u32;
    let code = status as u32;
    Err(Error::Backend {
        backend: "libamdf",
        operation,
        code: code as i32,
        message: format!("native status domain {domain}, code {code}").into_boxed_str(),
    })
}
fn missing(name: &str) -> Error {
    Error::Unsupported(format!("libamdf ABI is missing {name}"))
}
macro_rules! entry {
    ($table:expr, $name:ident) => {
        $table.$name.ok_or_else(|| missing(stringify!($name)))?
    };
}

mod xdna;
pub use xdna::{XdnaBinding, XdnaCompletion, XdnaProgram};
pub(crate) mod profile;
pub use profile::{DeviceInterval, DeviceProfile, ProfiledGpu};
mod queue;
pub use queue::{Completion, PreparedGpu, Queue};
mod executable;
pub use executable::{Argument, Kernel};
mod memory;
pub use memory::Buffer;

struct Api {
    core: &'static amdf_api_t,
    gpu: &'static amdf_gpu_api_t,
    xdna: &'static amdf_xdna_api_t,
    // Every native owner retains this mapping through its last native call.
    _library: Amdf,
    directory: std::path::PathBuf,
    bridge: executable::BridgeSlot,
}
impl Api {
    unsafe fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Amdf::new(path)? };
        let mut core = ptr::null();
        check("query_api", unsafe {
            library.amdf_query_api(HRX_AMDF_ABI_VERSION, HRX_AMDF_ABI_VERSION, &mut core)
        })?;
        if core.is_null() {
            return Err(missing("core table"));
        }
        unsafe {
            validate_table(core.cast(), size_of::<amdf_api_t>(), HRX_AMDF_ABI_VERSION)?;
        }
        let core = unsafe { &*core };
        let mut gpu = ptr::null();
        let mut xdna = ptr::null();
        check("query_extension(GPU)", unsafe {
            entry!(core, query_extension)(
                AMDF_EXTENSION_GPU,
                AMDF_GPU_EXTENSION_VERSION_LATEST,
                AMDF_GPU_EXTENSION_VERSION_LATEST,
                &mut gpu,
            )
        })?;
        check("query_extension(XDNA)", unsafe {
            entry!(core, query_extension)(
                AMDF_EXTENSION_XDNA,
                AMDF_XDNA_EXTENSION_VERSION_LATEST,
                AMDF_XDNA_EXTENSION_VERSION_LATEST,
                &mut xdna,
            )
        })?;
        if gpu.is_null() || xdna.is_null() {
            return Err(missing("device extension table"));
        }
        unsafe {
            validate_table(
                gpu,
                size_of::<amdf_gpu_api_t>(),
                AMDF_GPU_EXTENSION_VERSION_LATEST,
            )?;
            validate_table(
                xdna,
                size_of::<amdf_xdna_api_t>(),
                AMDF_XDNA_EXTENSION_VERSION_LATEST,
            )?;
        }
        let gpu = unsafe { &*gpu.cast::<amdf_gpu_api_t>() };
        let xdna = unsafe { &*xdna.cast::<amdf_xdna_api_t>() };
        validate_bridge_api(core, xdna)?;
        Ok(Self {
            core,
            gpu,
            xdna,
            _library: library,
            directory: path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            bridge: Default::default(),
        })
    }
}

// Query results guarantee a two-word header, not the requested full structure.
unsafe fn validate_table(
    pointer: *const std::ffi::c_void,
    bytes: usize,
    version: u32,
) -> Result<()> {
    if pointer.is_null() {
        return Err(missing("table header"));
    }
    let header = unsafe { pointer.cast::<[u32; 2]>().read() };
    if (header[0] as usize) < bytes || header[1] != version {
        return Err(missing("complete table with the requested version"));
    }
    Ok(())
}
// C adapters borrow these tables and call the entries directly. Validate them
// before any native ownership is acquired, including error-path destructors.
fn validate_bridge_api(core: &amdf_api_t, xdna: &amdf_xdna_api_t) -> Result<()> {
    let _ = entry!(core, endpoint_query_info);
    let _ = entry!(core, endpoint_query_queue_family_info);
    let _ = entry!(core, host_mapping_cache_control);
    let _ = entry!(core, host_mapping_destroy);
    let _ = entry!(core, host_mapping_query_info);
    let _ = entry!(core, instance_enumerate_memory_scopes);
    let _ = entry!(core, kernel_queue_destroy);
    let _ = entry!(core, kernel_queue_wait);
    let _ = entry!(core, memory_create);
    let _ = entry!(core, memory_destroy);
    let _ = entry!(core, memory_map);
    let _ = entry!(core, memory_query_address);
    let _ = entry!(core, memory_scope_query_device_profile);
    let _ = entry!(core, memory_scope_query_info);
    let _ = entry!(xdna, context_create);
    let _ = entry!(xdna, context_destroy);
    let _ = entry!(xdna, context_enumerate_memory_scopes);
    let _ = entry!(xdna, device_query_info);
    let _ = entry!(xdna, endpoint_query_info);
    let _ = entry!(xdna, kernel_queue_create);
    let _ = entry!(xdna, kernel_queue_submit);
    Ok(())
}

/// One provider instance and its passive memory/discovery domain.
#[derive(Clone)]
pub struct Fabric(Arc<Instance>);
struct Instance {
    api: Arc<Api>,
    raw: *mut amdf_instance_t,
}
// Instance operations used here are documented as thread-safe. Arc ownership
// prevents destruction while an endpoint, device, or allocation remains alive.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}
impl Drop for Instance {
    fn drop(&mut self) {
        let status = unsafe { self.api.core.instance_destroy.unwrap()(self.raw) };
        if status != 0 {
            std::mem::forget(self.api.clone());
        }
    }
}

/// Native engine type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// RDNA GPU.
    Gpu,
    /// Native XDNA array.
    Xdna,
}

/// A passive endpoint snapshot bound to its owning fabric.
#[derive(Clone)]
pub struct Endpoint(Arc<EndpointInner>);
struct EndpointInner {
    instance: Arc<Instance>,
    raw: *mut amdf_endpoint_t,
    name: String,
    engine: Engine,
    target: Target,
}
unsafe impl Send for EndpointInner {}
unsafe impl Sync for EndpointInner {}
impl Drop for EndpointInner {
    fn drop(&mut self) {
        let status = unsafe { self.instance.api.core.endpoint_close.unwrap()(self.raw) };
        if status != 0 {
            std::mem::forget(self.instance.clone());
        }
    }
}
impl Endpoint {
    /// Human-readable passive device name.
    pub fn name(&self) -> &str {
        &self.0.name
    }
    /// Exact compiler profile required by this endpoint.
    pub fn target(&self) -> &Target {
        &self.0.target
    }
    /// Engine selected without activating the device.
    pub fn engine(&self) -> Engine {
        self.0.engine
    }
    /// Activate this exact endpoint. Unsupported hardware is never substituted.
    pub fn open(&self) -> Result<Device> {
        if self.target().as_str() != "gfx1151" && !self.target().is_xdna() {
            return Err(Error::Unsupported(format!(
                "unqualified device {}",
                self.target().as_str()
            )));
        }
        let api = &self.0.instance.api;
        // Validate the destructor before taking ownership of a native handle.
        let _ = entry!(api.core, device_destroy);
        let mut raw = ptr::null_mut();
        unsafe {
            match self.engine() {
                Engine::Gpu => {
                    let options = amdf_gpu_device_create_info_t {
                        type_: HRX_AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
                        structure_size: size_of::<amdf_gpu_device_create_info_t>() as u32,
                        ..Default::default()
                    };
                    check(
                        "gpu.device_create",
                        entry!(api.gpu, device_create)(self.0.raw, &options, &mut raw),
                    )?;
                }
                Engine::Xdna => {
                    let options = amdf_xdna_device_create_info_t {
                        type_: HRX_AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
                        structure_size: size_of::<amdf_xdna_device_create_info_t>() as u32,
                        ..Default::default()
                    };
                    check(
                        "xdna.device_create",
                        entry!(api.xdna, device_create)(self.0.raw, &options, &mut raw),
                    )?;
                }
            }
        }
        if raw.is_null() {
            return Err(missing("created device"));
        }
        Ok(Device(Arc::new(DeviceInner {
            endpoint: self.clone(),
            raw,
        })))
    }
}

/// An activated native address/execution domain, independent of any program.
#[derive(Clone)]
pub struct Device(Arc<DeviceInner>);
struct DeviceInner {
    endpoint: Endpoint,
    raw: *mut amdf_device_t,
}
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}
impl Drop for DeviceInner {
    fn drop(&mut self) {
        let api = &self.endpoint.0.instance.api;
        let status = unsafe { api.core.device_destroy.unwrap()(self.raw) };
        if status != 0 {
            std::mem::forget(self.endpoint.clone());
        }
    }
}
impl Device {
    /// Passive identity retained by the activated device.
    pub fn endpoint(&self) -> &Endpoint {
        &self.0.endpoint
    }
    /// Exact target for offline compilation and executable admission.
    pub fn target(&self) -> &Target {
        self.endpoint().target()
    }
}

impl Fabric {
    /// Load a selected native provider and create an instance without discovery.
    pub fn load(path: &Path) -> Result<Self> {
        let api = Arc::new(unsafe { Api::load(path)? });
        let _ = entry!(api.core, instance_destroy);
        let options = amdf_instance_create_info_t {
            type_: AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            structure_size: size_of::<amdf_instance_create_info_t>() as u32,
            native_lifetime: AMDF_NATIVE_LIFETIME_INSTANCE,
            ..Default::default()
        };
        let mut raw = ptr::null_mut();
        check("instance_create", unsafe {
            entry!(api.core, instance_create)(&options, &mut raw)
        })?;
        if raw.is_null() {
            return Err(missing("created instance"));
        }
        Ok(Self(Arc::new(Instance { api, raw })))
    }

    /// Enumerate current endpoints without activating devices or creating queues.
    pub fn endpoints(&self) -> Result<Vec<Endpoint>> {
        let api = &self.0.api;
        let _ = entry!(api.core, endpoint_close);
        let mut count = 0;
        check("endpoint_enumerate", unsafe {
            entry!(api.core, endpoint_enumerate)(self.0.raw, 0, ptr::null_mut(), &mut count)
        })?;
        // Device arrivals can grow the snapshot between calls. Bound retries
        // instead of assuming the first count remains sufficient.
        let mut summaries = Vec::new();
        for attempt in 0..4 {
            summaries.resize(count as usize, amdf_endpoint_summary_t::default());
            let status = unsafe {
                entry!(api.core, endpoint_enumerate)(
                    self.0.raw,
                    summaries.len() as u32,
                    summaries.as_mut_ptr(),
                    &mut count,
                )
            };
            if status as u32 == AMDF_STATUS_CODE_BUFFER_TOO_SMALL && attempt < 3 {
                continue;
            }
            check("endpoint_enumerate", status)?;
            summaries.truncate(count as usize);
            break;
        }
        let mut endpoints = Vec::new();
        for summary in summaries {
            let mut raw = ptr::null_mut();
            check("endpoint_open", unsafe {
                entry!(api.core, endpoint_open)(self.0.raw, &summary.id, &mut raw)
            })?;
            if raw.is_null() {
                return Err(missing("opened endpoint"));
            }
            let metadata = (|| -> Result<_> {
                unsafe {
                    let mut info = amdf_endpoint_info_t {
                        type_: AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
                        structure_size: size_of::<amdf_endpoint_info_t>() as u32,
                        ..Default::default()
                    };
                    check(
                        "endpoint_query_info",
                        entry!(api.core, endpoint_query_info)(raw, &mut info),
                    )?;
                    let name = CStr::from_ptr(info.name.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                    let (engine, target) = match info.engine_kind {
                        AMDF_ENGINE_KIND_GPU => {
                            let mut gpu = amdf_gpu_endpoint_info_t {
                                type_: HRX_AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
                                structure_size: size_of::<amdf_gpu_endpoint_info_t>() as u32,
                                ..Default::default()
                            };
                            check(
                                "gpu.endpoint_query_info",
                                entry!(api.gpu, endpoint_query_info)(raw, &mut gpu),
                            )?;
                            (
                                Engine::Gpu,
                                Target::new(&format!(
                                    "gfx{}{}{:x}",
                                    gpu.gfx_ip.major, gpu.gfx_ip.minor, gpu.gfx_ip.stepping
                                ))?,
                            )
                        }
                        AMDF_ENGINE_KIND_XDNA => {
                            // The first release qualifies only the exact NPU5 deployment.
                            if info.pci.device_id != 0x17f0 || info.pci.revision_id != 0x11 {
                                return Err(Error::Unsupported(
                                    "unqualified XDNA deployment".into(),
                                ));
                            }
                            (Engine::Xdna, Target::xdna())
                        }
                        _ => return Err(Error::Unsupported("unknown AMD engine".into())),
                    };
                    Ok((name, engine, target))
                }
            })();
            match metadata {
                Ok((name, engine, target)) => endpoints.push(Endpoint(Arc::new(EndpointInner {
                    instance: self.0.clone(),
                    raw,
                    name,
                    engine,
                    target,
                }))),
                Err(error) => {
                    let status = unsafe { api.core.endpoint_close.unwrap()(raw) };
                    if status != 0 {
                        std::mem::forget(self.0.clone());
                    }
                    return Err(error);
                }
            }
        }
        Ok(endpoints)
    }
}

impl Fabric {
    /// Resolve the verified native bundle, or an explicit developer library.
    pub fn resolve() -> Result<Self> {
        static INSTANCE: std::sync::Mutex<std::sync::Weak<Instance>> =
            std::sync::Mutex::new(std::sync::Weak::new());
        let mut slot = INSTANCE
            .lock()
            .map_err(|_| Error::Message("fabric registry poisoned".into()))?;
        if let Some(instance) = slot.upgrade() {
            return Ok(Self(instance));
        }
        let path = match std::env::var_os("HRX_AMDF_LIBRARY") {
            Some(path) => std::path::PathBuf::from(path),
            None => crate::bundle::resolve()?.join("libamdf.so"),
        };
        let fabric = Self::load(&path)?;
        *slot = Arc::downgrade(&fabric.0);
        Ok(fabric)
    }
}
impl Device {
    /// Open a selected engine ordinal in the shared, weakly cached provider.
    /// Live buffers keep their exact device identity; the registry owns no device.
    pub fn open(engine: Engine, index: usize) -> Result<Self> {
        type Devices = Vec<(Engine, usize, std::sync::Weak<DeviceInner>)>;
        static DEVICES: std::sync::Mutex<Devices> = std::sync::Mutex::new(Vec::new());
        let mut devices = DEVICES
            .lock()
            .map_err(|_| Error::Message("device registry poisoned".into()))?;
        if let Some(device) = devices.iter().find_map(|(kind, ordinal, device)| {
            (*kind == engine && *ordinal == index)
                .then(|| device.upgrade())
                .flatten()
        }) {
            return Ok(Self(device));
        }
        devices.retain(|(_, _, device)| device.strong_count() != 0);
        let device = Fabric::resolve()?
            .endpoints()?
            .into_iter()
            .filter(|endpoint| endpoint.engine() == engine)
            .nth(index)
            .ok_or_else(|| Error::Unsupported(format!("{engine:?} device {index} is unavailable")))?
            .open()?;
        devices.push((engine, index, Arc::downgrade(&device.0)));
        Ok(device)
    }
    /// Stable identity while this device or any dependent owner is live.
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    /// Allocation and discovery domain retained by this device.
    pub fn fabric(&self) -> Fabric {
        Fabric(self.0.endpoint.0.instance.clone())
    }
}

#[cfg(test)]
mod abi_tests {
    use super::*;
    #[test]
    fn reject_truncated_and_mismatched_tables_before_full_reference() {
        for header in [[8u32, 5], [size_of::<amdf_api_t>() as u32, 4]] {
            assert!(
                unsafe { validate_table(header.as_ptr().cast(), size_of::<amdf_api_t>(), 5) }
                    .is_err()
            );
        }
        assert!(validate_bridge_api(&amdf_api_t::default(), &amdf_xdna_api_t::default()).is_err());
    }
}
