use crate::{Error, Result};
use std::ffi::{CStr, CString};

/// The native target family used for executable loading.
pub const TARGET_FAMILY: &CStr = c"amdgpu";
/// Default compiler target; devices report their architecture at runtime.
pub const TARGET_KEY: &str = "gfx1151";

/// An AMDGPU architecture key shared by devices, manifests and compiler profiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target(CString);
impl Target {
    /// Validate an architecture key, for example gfx1100, gfx1151 or gfx942.
    /// Availability is determined by the selected native runtime and compiler.
    /// Keys must be bare architecture names, without feature suffixes.
    pub fn new(key: &str) -> Result<Self> {
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
    // Device properties may include features such as :sramecc+:xnack-. Profiles
    // and executable loading take the base architecture; features are not configured here.
    pub(crate) fn from_device_architecture(name: &str) -> Result<Self> {
        Self::new(name.split(':').next().unwrap_or(name))
    }
    /// Exact architecture key.
    pub fn as_str(&self) -> &str {
        self.0.to_str().expect("validated ASCII target")
    }
    pub(crate) fn as_c_str(&self) -> &CStr {
        &self.0
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_features_are_separate_from_target_keys() {
        for name in ["gfx90a", "gfx90a:sramecc+:xnack-", "gfx90a:xnack+"] {
            assert_eq!(
                Target::from_device_architecture(name).unwrap().as_str(),
                "gfx90a"
            );
        }
        assert!(Target::from_device_architecture(":xnack-").is_err());
        assert!(Target::from_device_architecture("invalid:xnack-").is_err());
        assert!(
            Target::new("gfx90a:sramecc+:xnack-")
                .unwrap_err()
                .to_string()
                .contains("bare AMDGPU target")
        );
    }
}
