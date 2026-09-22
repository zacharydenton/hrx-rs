use super::*;
use std::{ffi::CString, sync::Mutex};
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::all
)]
#[path = "bridge_ffi.rs"]
pub(super) mod ffi;
use ffi::*;

pub(super) fn bridge_check(api: &Bridge, status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(api.hrx_fabric_error()) }.to_string_lossy();
    Err(Error::Backend {
        backend: "native executable",
        operation: "executable",
        code: status,
        message: message.into_owned().into_boxed_str(),
    })
}

/// One loaded GPU entry and its immutable native code and metadata.
#[derive(Clone)]
pub struct Kernel(pub(super) Arc<KernelInner>);
pub(super) struct KernelInner {
    pub(super) device: Device,
    pub(super) code: Buffer,
    image: GpuImage,
    pub(super) info: hrx_fabric_gpu_info,
}
struct GpuImage {
    api: Arc<Bridge>,
    raw: *mut hrx_fabric_gpu_image,
}
// Native image metadata is immutable after open; command construction uses
// only caller-owned output. Thread-local diagnostics are copied immediately.
unsafe impl Send for GpuImage {}
unsafe impl Sync for GpuImage {}
impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe { self.api.hrx_fabric_gpu_image_close(self.raw) };
    }
}

/// One argument in native declaration order.
pub enum Argument<'a> {
    /// Device-visible address at a checked offset within shared backing.
    Buffer(&'a Buffer, usize),
    /// Exact bytes of one scalar/record argument.
    Value(&'a [u8]),
}
impl Kernel {
    pub(crate) fn same_entry(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn argument_layout(&self) -> Result<Vec<(u32, usize)>> {
        (0..self.0.info.argument_count)
            .map(|index| {
                let mut info = hrx_fabric_gpu_argument::default();
                bridge_check(&self.0.image.api, unsafe {
                    self.0.image.api.hrx_fabric_gpu_argument_info(
                        self.0.image.raw,
                        index,
                        &mut info,
                    )
                })?;
                Ok((info.kind, info.size as usize))
            })
            .collect()
    }
    /// Required workgroup dimensions, when fixed by the compiled kernel.
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.0.info.workgroup_size
    }
    /// Device that owns the loaded executable allocation.
    pub fn device(&self) -> &Device {
        &self.0.device
    }
    pub(super) fn arguments(&self, arguments: &[Argument<'_>]) -> Result<(Vec<u8>, Vec<Buffer>)> {
        if arguments.len() != self.0.info.argument_count as usize {
            return Err(Error::Message("native argument count mismatch".into()));
        }
        let mut packed = vec![0; self.0.info.kernarg_bytes as usize];
        let mut resources = vec![self.0.code.clone()];
        for (index, argument) in arguments.iter().enumerate() {
            let mut info = hrx_fabric_gpu_argument::default();
            bridge_check(&self.0.image.api, unsafe {
                self.0.image.api.hrx_fabric_gpu_argument_info(
                    self.0.image.raw,
                    index as u32,
                    &mut info,
                )
            })?;
            let offset = info.offset as usize;
            let end = offset
                .checked_add(info.size as usize)
                .filter(|end| *end <= packed.len())
                .ok_or_else(|| Error::Message("native argument range overflow".into()))?;
            let destination = &mut packed[offset..end];
            match (info.kind, argument) {
                (1, Argument::Value(bytes)) if bytes.len() == destination.len() => {
                    destination.copy_from_slice(bytes)
                }
                (2, Argument::Buffer(buffer, offset))
                    if destination.len() == 8 && *offset < buffer.len() =>
                {
                    let address = buffer
                        .device_address(self.device())?
                        .checked_add(*offset as u64)
                        .ok_or_else(|| Error::Message("device argument address overflow".into()))?;
                    destination.copy_from_slice(&address.to_le_bytes());
                    resources.push((*buffer).clone());
                }
                _ => {
                    return Err(Error::Message(format!(
                        "native argument {index} has an incompatible kind, offset or size"
                    )));
                }
            }
        }
        Ok((packed, resources))
    }
    pub(super) fn dispatch_words(
        &self,
        grid: [u32; 3],
        block: [u16; 3],
        kernarg: &[u8],
        address: u64,
        scratch: Option<(&Buffer, u32, u32)>,
    ) -> Result<Vec<u32>> {
        let mut words = vec![0; 128];
        let mut count = 0;
        bridge_check(&self.0.image.api, unsafe {
            self.0.image.api.hrx_fabric_gpu_dispatch(
                self.0.image.raw,
                self.0.code.device_address(self.device())?,
                block.as_ptr(),
                grid.as_ptr(),
                address,
                kernarg.as_ptr(),
                kernarg.len(),
                scratch
                    .map(|(buffer, _, _)| buffer.device_address(self.device()))
                    .transpose()?
                    .unwrap_or(0),
                scratch.map_or(0, |(buffer, _, _)| buffer.len() as u64),
                scratch.map_or(0, |(_, waves, _)| waves),
                scratch.map_or(0, |(_, _, engines)| engines),
                words.as_mut_ptr(),
                words.len() as u32,
                &mut count,
            )
        })?;
        if count as usize > words.len() {
            return Err(Error::Message("native command output overflow".into()));
        }
        words.truncate(count as usize);
        Ok(words)
    }
}

impl Device {
    /// Load an entry from a compiler-produced artifact onto this GPU.
    ///
    /// # Safety
    /// The executable is native code. The caller must trust its code and its
    /// declared memory/argument contract; parsing is not a sandbox.
    pub unsafe fn load(&self, artifact: &crate::loom::Artifact) -> Result<Kernel> {
        if self.endpoint().engine() != Engine::Gpu || artifact.target() != self.target().as_str() {
            return Err(Error::Unsupported("artifact and GPU target differ".into()));
        }
        unsafe { self.load_bytes(artifact.bytes(), artifact.symbol()) }
    }
    /// Load a native gfx1151 code object and select its named entry.
    ///
    /// # Safety
    /// The native code must be trusted and obey its declared memory contract.
    pub unsafe fn load_bytes(&self, bytes: &[u8], symbol: &str) -> Result<Kernel> {
        if self.endpoint().engine() != Engine::Gpu {
            return Err(Error::Unsupported("GPU code requires a GPU device".into()));
        }
        let api = &self.0.endpoint.0.instance.api;
        let bridge = api.load_bridge()?;
        let symbol = CString::new(symbol)
            .map_err(|_| Error::Message("kernel symbol contains NUL".into()))?;
        let mut raw = ptr::null_mut();
        let mut info = hrx_fabric_gpu_info::default();
        bridge_check(&bridge, unsafe {
            bridge.hrx_fabric_gpu_image_open(
                bytes.as_ptr(),
                bytes.len(),
                symbol.as_ptr(),
                &mut raw,
                &mut info,
            )
        })?;
        if raw.is_null() {
            return Err(missing("GPU image"));
        }
        let image = GpuImage { api: bridge, raw };
        let size = usize::try_from(info.storage_bytes)
            .map_err(|_| Error::Message("GPU image exceeds address space".into()))?;
        let fabric = Fabric(self.0.endpoint.0.instance.clone());
        let code = fabric.allocate_access(
            size,
            std::slice::from_ref(self),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | AMDF_MEMORY_ACCESS_EXECUTE,
            4096,
        )?;
        let mut contents = vec![0; size];
        bridge_check(&image.api, unsafe {
            image.api.hrx_fabric_gpu_image_load(
                image.raw,
                contents.as_mut_ptr(),
                contents.len(),
                code.device_address(self)?,
            )
        })?;
        code.write(0, &contents)?;
        Ok(Kernel(Arc::new(KernelInner {
            device: self.clone(),
            code,
            image,
            info,
        })))
    }
}

pub(super) type BridgeSlot = Mutex<Option<Arc<Bridge>>>;

impl Api {
    pub(super) fn load_bridge(&self) -> Result<Arc<Bridge>> {
        let api = self;

        let mut slot = api
            .bridge
            .lock()
            .map_err(|_| Error::Message("bridge loader poisoned".into()))?;
        if slot.is_none() {
            let path = std::env::var_os("HRX_FABRIC_LIBRARY")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| api.directory.join("libhrx_fabric.so"));
            *slot = Some(Arc::new(unsafe { Bridge::new(path)? }));
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}
