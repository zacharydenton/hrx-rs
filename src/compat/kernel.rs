//! A compiled Loom export, ready to dispatch.
use std::sync::Arc;

use super::device::c_string;
use super::{Args, Error, Result, check, sys, try_device};

struct Executable(sys::Executable);

// The handle is only used through &Device, under its mutex.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        // Native commands retain the HAL executable after this wrapper is released.
        unsafe {
            sys::hrx_executable_release(self.0);
        }
    }
}

/// One export of one HSACO. Cloning shares the loaded executable.
#[derive(Clone)]
pub struct Kernel {
    executable: Arc<Executable>,
    ordinal: u32,
    info: sys::ExportInfo,
    device: Arc<super::Device>,
}

// Metadata is immutable and its name pointer is retained by executable.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel(export {})", self.ordinal)
    }
}

impl Kernel {
    /// Loads `path` for the selected device and looks up `symbol`.
    /// # Safety
    /// The code object must be trusted native code, just like a shared library.
    /// ```compile_fail
    /// let _ = hrx::compat::Kernel::load(std::path::Path::new("kernel.hsaco"), "k");
    /// ```
    pub unsafe fn load(path: &std::path::Path, symbol: &str) -> Result<Kernel> {
        let path_text = path
            .to_str()
            .ok_or_else(|| Error::Message(format!("{} is not UTF-8", path.display())))?;
        let path_c = c_string(path_text)?;
        let symbol_c = c_string(symbol)?;
        let device = try_device()?;
        let family = crate::TARGET_FAMILY;
        let target = device.target().as_c_str();
        // Safety: every string outlives the call; HRX writes the handle on
        // success, and the ordinal lookup happens on a loaded executable.
        unsafe {
            let mut executable: sys::Executable = std::ptr::null_mut();
            check(sys::hrx_executable_load_file(
                device.raw(),
                path_c.as_ptr(),
                family.as_ptr(),
                target.as_ptr(),
                &mut executable,
            ))?;
            let executable = Executable(executable);
            let mut ordinal: u32 = 0;
            check(sys::hrx_executable_lookup_export_by_name(
                executable.0,
                symbol_c.as_ptr(),
                &mut ordinal,
            ))?;
            let mut info = sys::ExportInfo::default();
            check(sys::hrx_executable_export_info(
                executable.0,
                ordinal,
                &mut info,
            ))?;
            Ok(Kernel {
                executable: Arc::new(executable),
                ordinal,
                info,
                device,
            })
        }
    }

    /// Load compiler-owned executable bytes directly into the scoped device.
    ///
    /// # Safety
    /// The artifact must be trusted native code, as for [`Kernel::load`].
    #[cfg(feature = "loom")]
    pub unsafe fn load_artifact(artifact: &crate::loom::Artifact) -> Result<Kernel> {
        let device = try_device()?;
        if artifact.target() != device.target().as_str() {
            return Err(Error::Message(
                "artifact target does not match this runtime".into(),
            ));
        }
        let symbol = c_string(artifact.symbol())?;
        unsafe {
            let mut raw = std::ptr::null_mut();
            check(sys::hrx_executable_load_data(
                device.raw(),
                artifact.bytes().as_ptr().cast(),
                artifact.bytes().len(),
                crate::TARGET_FAMILY.as_ptr(),
                device.target().as_c_str().as_ptr(),
                &mut raw,
            ))?;
            let executable = Executable(raw);
            let mut ordinal = 0;
            check(sys::hrx_executable_lookup_export_by_name(
                executable.0,
                symbol.as_ptr(),
                &mut ordinal,
            ))?;
            let mut info = sys::ExportInfo::default();
            check(sys::hrx_executable_export_info(
                executable.0,
                ordinal,
                &mut info,
            ))?;
            Ok(Kernel {
                executable: Arc::new(executable),
                ordinal,
                info,
                device,
            })
        }
    }

    /// One dispatch: `grid` workgroups of `block` work items, subgroup selection from the executable.
    /// # Safety
    /// Dimensions and argument layout must match the kernel. Every accessed
    /// address must stay within a live allocation owned by the current scoped
    /// device, and accesses must obey the kernel's aliasing requirements.
    /// ```compile_fail
    /// fn unchecked(k: &hrx::compat::Kernel, args: &hrx::compat::Args) {
    ///     k.launch([1; 3], [1; 3], args).unwrap();
    /// }
    /// ```
    pub unsafe fn launch(&self, grid: [u32; 3], block: [u32; 3], args: &Args) -> Result<()> {
        let device = try_device()?;
        if device.raw() != self.device.raw() {
            return Err(Error::Message("kernel belongs to another device".into()));
        }
        crate::runtime::validate_export_launch(&self.info, grid, block)?;
        let config = sys::DispatchConfig {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: sys::SUBGROUP_SIZE_FROM_EXECUTABLE,
        };
        unsafe { device.dispatch(self.executable.0, self.ordinal, &config, args) }
    }

    /// The common case: a 2-D grid of 1-D workgroups.
    /// # Safety
    /// The same invocation and allocation requirements as [`Self::launch`] apply.
    pub unsafe fn launch_2d(
        &self,
        grid_x: u32,
        grid_y: u32,
        threads: u32,
        args: &Args,
    ) -> Result<()> {
        unsafe { self.launch([grid_x, grid_y, 1], [threads, 1, 1], args) }
    }
}
