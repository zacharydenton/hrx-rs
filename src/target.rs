use crate::{Error, Result};
use std::ffi::CString;

/// Default compiler target; devices report their architecture at runtime.
pub const TARGET_KEY: &str = "gfx1151";

/// An exact AMDGPU or XDNA deployment key.
///
/// This is the *profile* target: what a device reports and what the compiler
/// emits for. It is deliberately narrower than the target a Loom source file
/// names in `amdgpu.target<...>`, which accepts generic families such as
/// `gfx11-generic`. The two are chosen independently — generic source compiles
/// under a bare profile — so this type rejects generic names rather than
/// silently accepting one a device can never report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target(CString);
impl Target {
    /// Validate an architecture key, for example gfx1100, gfx1151 or gfx942.
    /// Availability is determined by the selected native runtime and compiler.
    /// Keys must be bare architecture names, without feature suffixes.
    pub fn new(key: &str) -> Result<Self> {
        if key == "amd.xdna.strix_halo.17f0_11" {
            return Ok(Self(CString::new(key).unwrap()));
        }
        if key.contains(':') {
            return Err(Error::Message(format!(
                "expected a bare AMDGPU target, found feature-suffixed name {key:?}"
            )));
        }
        let suffix = key.strip_prefix("gfx").unwrap_or("");
        if !(3..=4).contains(&suffix.len())
            || !suffix
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Message(format!("invalid AMDGPU target {key:?}")));
        }
        Ok(Self(
            CString::new(key).expect("validated target has no NUL"),
        ))
    }
    /// Exact Strix Halo NPU5 deployment profile.
    pub fn xdna() -> Self {
        Self::new("amd.xdna.strix_halo.17f0_11").unwrap()
    }
    /// Whether this profile targets native XDNA execution.
    pub fn is_xdna(&self) -> bool {
        self.as_str().starts_with("amd.xdna.")
    }
    pub(crate) fn artifact_format(&self) -> &'static str {
        if self.is_xdna() {
            "xdna"
        } else {
            "amdgpu-hsaco"
        }
    }
    pub(crate) fn artifact_filename(&self) -> &'static str {
        if self.is_xdna() {
            "kernel.xdna"
        } else {
            "kernel.hsaco"
        }
    }
    /// Exact architecture key.
    pub fn as_str(&self) -> &str {
        self.0.to_str().expect("validated ASCII target")
    }
    /// Bundle platform and architecture identifier.
    pub fn manifest_key(&self) -> String {
        format!("x86_64-unknown-linux-gnu-{}", self.as_str())
    }
}

impl Default for Target {
    fn default() -> Self {
        Self::new(TARGET_KEY).expect("valid default target")
    }
}

impl Target {
    /// Hardware wavefront width that [`Target::loom_record`] selects.
    ///
    /// GCN and CDNA (`gfx9xx`) run wave64; RDNA targets run wave32.
    pub fn subgroup_size(&self) -> u32 {
        if self.as_str().starts_with("gfx9") {
            64
        } else {
            32
        }
    }

    /// A Loom `amdgpu.target` record named `@symbol` for this device.
    ///
    /// RDNA3 devices share the `gfx11-generic` family, so one artifact serves
    /// every gfx11 key; other targets name their exact architecture. The record
    /// fixes [`Target::subgroup_size`], so kernels that exchange values across
    /// lanes must size their reductions and wave layout from it.
    pub fn loom_record(&self, symbol: &str) -> String {
        let key = self.as_str();
        let family = if key.starts_with("gfx11") {
            "gfx11-generic"
        } else {
            key
        };
        format!(
            "amdgpu.target<{family}> @{symbol} {{subgroup_size = {}}}",
            self.subgroup_size()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Target;

    #[test]
    fn loom_records_select_family_and_wave_width() {
        let rdna = Target::new("gfx1151").unwrap();
        assert_eq!(rdna.subgroup_size(), 32);
        assert_eq!(
            rdna.loom_record("t"),
            "amdgpu.target<gfx11-generic> @t {subgroup_size = 32}"
        );
        let gcn = Target::new("gfx906").unwrap();
        assert_eq!(gcn.subgroup_size(), 64);
        assert_eq!(
            gcn.loom_record("t"),
            "amdgpu.target<gfx906> @t {subgroup_size = 64}"
        );
    }
}
