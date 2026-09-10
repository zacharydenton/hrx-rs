use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Device access used for dependency inference and kernel contract validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    /// Reads existing bytes without modifying them.
    Read,
    /// Writes bytes without reading their previous contents.
    Write,
    /// Reads and writes bytes.
    ReadWrite,
}
impl Access {
    pub(super) fn writes(self) -> bool {
        self != Self::Read
    }
}
/// A fixed-specialization buffer argument contract.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingContract {
    /// Minimum accessible extent in bytes, including all DMA and vector accesses.
    pub bytes: usize,
    /// Required byte offset alignment; must be a power of two.
    pub alignment: usize,
    /// All reads and writes the kernel may perform through this argument.
    pub access: Access,
    /// Layout identity for diagnostics (packing remains the caller's responsibility).
    pub layout: String,
}
/// The trusted ABI of a fixed kernel specialization.
///
/// A manifest describes this contract but does not prove that code obeys it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelContract {
    /// Arguments in kernel ABI order.
    pub bindings: Vec<BindingContract>,
    /// Exact accepted scalar bytes. Specialize again to change scalar values.
    pub constants: Vec<u8>,
}
impl KernelContract {
    /// Check structural validity without asserting that a kernel obeys this contract.
    pub fn validate(&self) -> Result<()> {
        for binding in &self.bindings {
            if binding.bytes == 0
                || binding.bytes > isize::MAX as usize
                || !binding.alignment.is_power_of_two()
            {
                return Err(Error::Message(
                    "invalid kernel binding extent or alignment".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn check(&self, bindings: &[super::BufferView]) -> Result<()> {
        self.validate()?;
        if bindings.len() != self.bindings.len() {
            return Err(Error::Message("kernel binding count mismatch".into()));
        }
        for (view, contract) in bindings.iter().zip(&self.bindings) {
            if view.len() < contract.bytes || view.offset() % contract.alignment != 0 {
                return Err(Error::Message(format!(
                    "binding does not satisfy {}: {} bytes aligned to {} required",
                    contract.layout, contract.bytes, contract.alignment
                )));
            }
        }
        // Aliasing across arguments is not asserted by the initial contract format.
        for (index, a) in bindings.iter().enumerate() {
            for (other, b) in bindings[..index].iter().enumerate() {
                if a.overlaps(b)
                    && (self.bindings[index].access.writes()
                        || self.bindings[other].access.writes())
                {
                    return Err(Error::Message(
                        "overlapping writable kernel arguments require an unsafe dispatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}
