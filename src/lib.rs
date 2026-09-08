//! GPU execution and Loom compilation without a build-time native dependency.
//!
//! Native libraries are verified and loaded on first use; CPU-only consumers and
//! C ABI metadata calls never initialise the GPU. See [`bundle`] for offline setup.
pub mod bundle;
mod runtime;
pub mod sys;
pub use runtime::*;
/// Compatibility for existing address-based Krea kernels. New models should use
/// owned buffers and binding dispatch through [`Stream`].
#[cfg(feature = "compat")]
pub mod compat;
#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "loom")]
pub mod loom;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<String> for Error {
    fn from(e: String) -> Self {
        Self(e)
    }
}
impl From<&str> for Error {
    fn from(e: &str) -> Self {
        Self(e.into())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "runner")]
pub mod runner;
