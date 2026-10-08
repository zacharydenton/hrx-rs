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
    image: Arc<GpuImage>,
    sanitizer: Option<Arc<super::sanitizer::Feedback>>,
    feedback_global: std::ops::Range<usize>,
    race: Option<RaceTemplate>,
    pub(super) race_state: Option<super::sanitizer::RaceState>,
    pub(super) info: hrx_fabric_gpu_info,
}
struct RaceTemplate {
    global: std::ops::Range<usize>,
    options: super::SanitizerRuntimeOptions,
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
    /// Drain owned value/operation and workgroup race reports after all invocations retire.
    /// Returns Busy while any submitted invocation retains the feedback storage.
    /// A full channel reports dropped packets explicitly; it never stalls a kernel.
    pub fn sanitizer_reports(&self) -> Result<super::SanitizerReports> {
        self.0
            .sanitizer
            .as_ref()
            .ok_or_else(|| Error::Unsupported("kernel has no sanitizer runtime".into()))?
            .drain()
    }
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
    /// Hardware subgroup width recorded in the loaded executable.
    pub fn subgroup_size(&self) -> u32 {
        self.0.info.wave_size
    }
    /// Static workgroup-local storage required by the loaded executable.
    pub fn workgroup_storage_bytes(&self) -> u64 {
        u64::from(self.0.info.local_bytes)
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
        if let Some(feedback) = &self.0.sanitizer {
            resources.push(feedback.buffer().clone());
        }
        if let Some(state) = &self.0.race_state {
            resources.extend([state.shadow.clone(), state.queue_state.clone()]);
        }
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
    pub(super) fn for_aql(&self, grid: [u32; 3], ring: u64, mask: u64) -> Result<Self> {
        let Some(race) = &self.0.race else {
            return Ok(self.clone());
        };
        let state = super::sanitizer::RaceState::new(
            self.device(),
            &race.options,
            self.0.info.local_bytes,
            grid,
            ring,
            mask,
        )?;
        let code = self.device().fabric().allocate_owned(
            self.0.code.len(),
            self.device(),
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | AMDF_MEMORY_ACCESS_EXECUTE,
            4096,
            false,
            race.options.memory_budget.as_ref(),
        )?;
        let mut contents = vec![0; code.len()];
        bridge_check(&self.0.image.api, unsafe {
            self.0.image.api.hrx_fabric_gpu_image_load(
                self.0.image.raw,
                contents.as_mut_ptr(),
                contents.len(),
                code.device_address(self.device())?,
            )
        })?;
        self.0
            .sanitizer
            .as_ref()
            .ok_or_else(|| missing("race feedback"))?
            .configure(&mut contents[self.0.feedback_global.clone()])?;
        state.configure(&mut contents[race.global.clone()])?;
        code.write(0, &contents)?;
        Ok(Self(Arc::new(KernelInner {
            device: self.device().clone(),
            code,
            image: self.0.image.clone(),
            sanitizer: self.0.sanitizer.clone(),
            feedback_global: self.0.feedback_global.clone(),
            race: None,
            race_state: Some(state),
            info: self.0.info,
        })))
    }
    pub(super) fn race_budget(&self) -> Option<&crate::residency::MemoryBudget> {
        self.0
            .race
            .as_ref()
            .and_then(|r| r.options.memory_budget.as_ref())
    }
    pub(super) fn dispatch_words(
        &self,
        grid: [u32; 3],
        block: [u16; 3],
        kernarg: &[u8],
        address: u64,
        scratch: Option<(&Buffer, u32, u32)>,
    ) -> Result<Vec<u32>> {
        if self.0.sanitizer.is_some() {
            return Err(Error::Unsupported(
                "sanitizer instrumentation requires AQL dispatch".into(),
            ));
        }
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
    /// Load an artifact with an owned, bounded sanitizer feedback channel.
    /// Supports value/operation reports and workgroup-local race checking.
    /// Address instrumentation requires a bounded shadow runtime and is rejected.
    /// Dispatch through an AQL queue, which supplies the native dispatch pointer.
    /// # Safety
    /// The artifact is trusted native code. Use report-only instrumentation when
    /// execution must complete after a diagnostic; trap mode can fault the queue.
    pub unsafe fn load_sanitized(
        &self,
        artifact: &crate::loom::Artifact,
        options: &super::SanitizerRuntimeOptions,
    ) -> Result<Kernel> {
        if artifact.target() != self.target().as_str() {
            return Err(Error::Unsupported("artifact and GPU target differ".into()));
        }
        unsafe { self.load_image(artifact.bytes(), artifact.symbol(), Some(options), None) }
    }
    /// Load a native gfx1151 code object and select its named entry.
    ///
    /// # Safety
    /// The native code must be trusted and obey its declared memory contract.
    pub unsafe fn load_bytes(&self, bytes: &[u8], symbol: &str) -> Result<Kernel> {
        unsafe { self.load_image(bytes, symbol, None, None) }
    }
    pub(super) unsafe fn load_image(
        &self,
        bytes: &[u8],
        symbol: &str,
        sanitizer_options: Option<&super::SanitizerRuntimeOptions>,
        code_budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Kernel> {
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
        let global = |name: &std::ffi::CStr| -> Result<std::ops::Range<usize>> {
            let mut offset = 0;
            let mut length = 0;
            bridge_check(&image.api, unsafe {
                image.api.hrx_fabric_gpu_global_info(
                    image.raw,
                    name.as_ptr(),
                    &mut offset,
                    &mut length,
                )
            })?;
            let start = usize::try_from(offset)
                .map_err(|_| Error::Message("global offset overflow".into()))?;
            let length = usize::try_from(length)
                .map_err(|_| Error::Message("global size overflow".into()))?;
            let end = start
                .checked_add(length)
                .filter(|end| *end as u64 <= info.storage_bytes)
                .ok_or_else(|| Error::Message("global extent outside image".into()))?;
            Ok(start..end)
        };
        let feedback_global = global(c"iree_feedback_config")?;
        if !global(c"iree_asan_config")?.is_empty() {
            return Err(Error::Unsupported("instrumented artifact requires bounded address shadow storage; this loader does not supply it".into()));
        }
        match (feedback_global.is_empty(), sanitizer_options.is_some()) {
            (false, false) => {
                return Err(Error::Unsupported(
                    "artifact requires sanitizer feedback; use load_sanitized".into(),
                ));
            }
            (true, true) => {
                return Err(Error::Unsupported(
                    "artifact has no structured sanitizer feedback global".into(),
                ));
            }
            _ => (),
        }
        let race_global = global(c"iree_tsan_config")?;
        if !race_global.is_empty() && race_global.len() != 96 {
            return Err(Error::Unsupported(
                "unexpected race configuration ABI".into(),
            ));
        }
        let race = if race_global.is_empty() {
            None
        } else {
            Some(RaceTemplate {
                global: race_global,
                options: sanitizer_options
                    .ok_or_else(|| {
                        Error::Unsupported("race instrumentation requires load_sanitized".into())
                    })?
                    .clone(),
            })
        };
        let sites_global = global(c"loom_sanitizer_sites")?;
        let size = usize::try_from(info.storage_bytes)
            .map_err(|_| Error::Message("GPU image exceeds address space".into()))?;
        let fabric = Fabric(self.0.endpoint.0.instance.clone());
        let code = fabric.allocate_owned(
            size,
            self,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | AMDF_MEMORY_ACCESS_EXECUTE,
            4096,
            false,
            code_budget,
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
        let sanitizer = sanitizer_options
            .map(|options| {
                let feedback =
                    super::sanitizer::Feedback::new(self, options, &contents[sites_global])?;
                let config = contents
                    .get_mut(feedback_global.clone())
                    .filter(|bytes| bytes.len() == 64)
                    .ok_or_else(|| {
                        Error::Unsupported("unexpected feedback configuration ABI".into())
                    })?;
                feedback.configure(config)?;
                Ok::<_, Error>(Arc::new(feedback))
            })
            .transpose()?;
        code.write(0, &contents)?;
        Ok(Kernel(Arc::new(KernelInner {
            sanitizer,
            feedback_global,
            race,
            race_state: None,
            device: self.clone(),
            code,
            image: Arc::new(image),
            info,
        })))
    }
}

pub(super) type BridgeSlot = Mutex<Option<Arc<Bridge>>>;

impl Api {
    /// The native bridge library: `HRX_FABRIC_LIBRARY`, or the one beside
    /// `libamdf.so`. Every loader of bridge symbols uses this one path.
    pub(super) fn bridge_path(&self) -> std::path::PathBuf {
        std::env::var_os("HRX_FABRIC_LIBRARY")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.directory.join("libhrx_fabric.so"))
    }

    pub(super) fn load_bridge(&self) -> Result<Arc<Bridge>> {
        let api = self;

        let mut slot = api
            .bridge
            .lock()
            .map_err(|_| Error::Message("bridge loader poisoned".into()))?;
        if slot.is_none() {
            *slot = Some(Arc::new(unsafe { Bridge::new(api.bridge_path())? }));
        }
        Ok(slot.as_ref().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loom::{Compiler, CxxSource, Specialization};

    #[test]
    #[ignore = "requires native compiler, bridge, libamdf and gfx1151"]
    fn replacing_retired_code_at_the_same_address_invalidates_all_instruction_lines() -> Result<()>
    {
        let gpu = Device::open(Engine::Gpu, 0)?;
        let fabric = gpu.fabric();
        let compiler = Compiler::for_target(None, gpu.target())?;
        let mut kernels = Vec::new();
        for stamp in [97u32, 173] {
            let source = format!(
                "#include <hip/hip_runtime.h>\n\
                 __global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(256, 1, 1)]]\n\
                 void stamp(unsigned* output) {{\n\
                   unsigned i = blockIdx.x * 64u + threadIdx.x;\n\
                   output[i] = i ^ {stamp}u;\n\
                 }}"
            );
            let artifact = compiler
                .import_cxx(CxxSource::new("stamp.cpp", source))?
                .compile(&Specialization::new("stamp"))?;
            // The owned fixture writes exactly 256 * 64 u32 values.
            kernels.push(unsafe { gpu.load(&artifact) }?);
        }
        let mut replacement = Arc::try_unwrap(kernels.pop().unwrap().0)
            .unwrap_or_else(|_| panic!("newly loaded fixture unexpectedly shared"));
        let original = kernels.pop().unwrap();
        assert_eq!(
            original.0.info.storage_bytes,
            replacement.info.storage_bytes
        );
        // Deliberately reuse one executable address, independent of allocator
        // placement. Every rewrite below occurs after checked GPU retirement.
        // This private test bypasses the public immutable-code API solely to
        // reproduce freeing a code allocation and reusing its address.
        replacement.code = original.0.code.clone();
        let replacement = Kernel(Arc::new(replacement));
        let queue = gpu.queue()?;
        let output = fabric.allocate(256 * 64 * 4, std::slice::from_ref(&gpu))?;
        let mut actual = vec![0; output.len()];
        for _ in 0..8 {
            for (kernel, stamp) in [(&original, 97u32), (&replacement, 173u32)] {
                let mut contents = vec![0; kernel.0.info.storage_bytes as usize];
                bridge_check(&kernel.0.image.api, unsafe {
                    kernel.0.image.api.hrx_fabric_gpu_image_load(
                        kernel.0.image.raw,
                        contents.as_mut_ptr(),
                        contents.len(),
                        kernel.0.code.device_address(&gpu)?,
                    )
                })?;
                kernel.0.code.write(0, &contents)?;
                output.write(0, &vec![0xcd; output.len()])?;
                // Same complete output binding for both known kernels.
                let done = unsafe {
                    queue.dispatch(
                        kernel,
                        [256, 1, 1],
                        [64, 1, 1],
                        &[Argument::Buffer(&output, 0)],
                    )
                }?;
                if !done.wait_timeout(std::time::Duration::from_secs(5))? {
                    std::mem::forget(done);
                    return Err(Error::DeviceLost(
                        "code-reuse fixture did not retire".into(),
                    ));
                }
                output.read(0, &mut actual)?;
                for (index, bytes) in actual.as_chunks::<4>().0.iter().enumerate() {
                    assert_eq!(
                        u32::from_le_bytes(*bytes),
                        index as u32 ^ stamp,
                        "stale executable at element {index}, expected stamp {stamp}"
                    );
                }
            }
        }
        Ok(())
    }
}
