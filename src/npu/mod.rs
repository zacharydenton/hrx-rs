//! XDNA2 programs and fixed instruction specializations.
//!
//! Programs are explicitly trusted native code. Buffers and execution are managed
//! through [`crate::execution`]; the [`raw`] API requires external synchronization.

#[cfg(feature = "npu-compile")]
pub mod compiler;
pub mod provision;
pub mod raw;
use crate::{Error, Result, execution::KernelContract};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) fn load_shim() -> Result<libloading::Library> {
    let directory = if let Some(path) = std::env::var_os("HRX_NPU_RUNTIME_DIR") {
        PathBuf::from(path)
    } else if let Some(path) = std::env::var_os("HRX_NPU_BUNDLE_MANIFEST") {
        provision::Manifest::load(path)?.prepare(std::env::var_os("HRX_OFFLINE").is_some())?
    } else {
        crate::bundle::resolve()?
    };
    if directory.join("component.json").is_file() {
        provision::Manifest::load(directory.join("component.json"))?.verify(&directory)?;
    }
    let path = directory.join("libhrx_npu.so.1");
    let library = unsafe { libloading::Library::new(&path) }.map_err(|e| {
        Error::from(e).context(format!(
            "loading {}; build scripts/build-npu-shim.sh or prepare an NPU runtime",
            path.display()
        ))
    })?;
    let abi = unsafe { library.get::<unsafe extern "C" fn() -> u32>(b"hrx_npu_abi_version\0") }?;
    if unsafe { abi() } != 1 {
        return Err(Error::Unsupported("NPU shim ABI must be 1".into()));
    }
    Ok(library)
}
/// An immutable, resident NPU device image. Cloning reuses the native context.
#[derive(Clone)]
pub struct NpuProgram {
    pub(crate) inner: Arc<ProgramInner>,
}
pub(crate) struct ProgramInner {
    pub context: raw::Context,
    pub device: i32,
    pub identity: String,
}
impl NpuProgram {
    /// Load an xclbin on an explicitly selected NPU.
    /// # Safety
    /// The image is trusted native code for this device; loading it can start
    /// tile programs. It must not perform unbound DMA or access host memory
    /// except through subsequent correctly contracted invocations.
    pub unsafe fn load(device: i32, path: impl AsRef<Path>) -> Result<Self> {
        let path = std::fs::canonicalize(path)?;
        let identity = crate::bundle::file_digest(&path)?;
        type Programs = std::collections::BTreeMap<(i32, String), std::sync::Weak<ProgramInner>>;
        static PROGRAMS: std::sync::OnceLock<std::sync::Mutex<Programs>> =
            std::sync::OnceLock::new();
        let mut programs = PROGRAMS
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| Error::Message("NPU program registry poisoned".into()))?;
        if let Some(inner) = programs
            .get(&(device, identity.clone()))
            .and_then(std::sync::Weak::upgrade)
        {
            return Ok(Self { inner });
        }
        programs.retain(|_, program| program.strong_count() != 0);
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::Message("xclbin path is not UTF-8".into()))?;
        let context = unsafe { raw::Context::new(device, path_str) }.map_err(Error::Message)?;
        if crate::bundle::file_digest(&path)? != identity {
            return Err(Error::Message("xclbin changed while loading".into()));
        }
        let inner = Arc::new(ProgramInner {
            context,
            device,
            identity: identity.clone(),
        });
        programs.insert((device, identity), Arc::downgrade(&inner));
        Ok(Self { inner })
    }
    /// Content identity of the resident xclbin.
    pub fn identity(&self) -> &str {
        &self.inner.identity
    }
    /// Bind a fixed instruction specialization to this image and contract.
    /// # Safety
    /// Instructions must match this exact image and target. All accesses must
    /// remain within the contract, with the declared read/write modes, for every
    /// accepted binding. Argument aliasing is rejected by the safe graph API.
    pub unsafe fn kernel(
        &self,
        instructions: &[u8],
        contract: KernelContract,
    ) -> Result<NpuKernel> {
        contract.validate()?;
        if !contract.constants.is_empty()
            || instructions.is_empty()
            || !instructions.len().is_multiple_of(4)
            || instructions.len() / 4 > u32::MAX as usize
        {
            return Err(Error::Message(
                "MLIR_AIE requires a nonempty u32 instruction stream and no extra scalar arguments"
                    .into(),
            ));
        }
        let context = &self.inner.context;
        let insts = context
            .alloc_bo(
                instructions.len(),
                raw::BoKind::Cacheable,
                context.group_id(1).map_err(Error::Message)?,
            )
            .map_err(Error::Message)?;
        insts.write(instructions).map_err(Error::Message)?;
        let groups = (0..contract.bindings.len())
            .map(|i| context.group_id((i + 3) as i32).map_err(Error::Message))
            .collect::<Result<Vec<_>>>()?;
        Ok(NpuKernel {
            inner: Arc::new(KernelInner {
                program: self.clone(),
                instructions: insts,
                contract,
                groups,
            }),
        })
    }
}
/// A reusable fixed NPU instruction specialization and its trusted contract.
#[derive(Clone)]
pub struct NpuKernel {
    pub(crate) inner: Arc<KernelInner>,
}
pub(crate) struct KernelInner {
    pub program: NpuProgram,
    pub instructions: raw::Bo,
    pub contract: KernelContract,
    pub groups: Vec<i32>,
}
impl NpuKernel {
    /// The contract checked for every graph binding.
    pub fn contract(&self) -> &KernelContract {
        &self.inner.contract
    }
}
