//! Native XDNA compilation and execution through the shared AMD fabric.
//! Programs use canonical Loom `.xdna` images; allocation is device-scoped.
use crate::{Error, Result, execution::KernelContract};
use std::sync::Arc;

/// A trusted canonical XDNA entry and its checked external access contract.
#[derive(Clone)]
pub struct NpuKernel {
    pub(crate) inner: Arc<KernelInner>,
}
pub(crate) struct KernelInner {
    pub device: crate::fabric::Device,
    pub artifact: crate::loom::Artifact,
    pub columns: u16,
    pub contract: KernelContract,
    _reservation: Option<crate::residency::MemoryReservation>,
}
impl NpuKernel {
    pub(crate) unsafe fn load(
        device: crate::fabric::Device,
        artifact: &crate::loom::Artifact,
        columns: u16,
        contract: KernelContract,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Self> {
        contract.validate()?;
        if device.endpoint().engine() != crate::fabric::Engine::Xdna
            || artifact.target() != device.target().as_str()
            || !(1..=8).contains(&columns)
            || !contract.constants.is_empty()
        {
            return Err(Error::Unsupported(
                "XDNA requires an exact device profile, 1..=8 columns, and buffer-only bindings"
                    .into(),
            ));
        }
        let reservation = budget
            .map(|budget| budget.reserve(artifact.bytes().len()))
            .transpose()?;
        Ok(Self {
            inner: Arc::new(KernelInner {
                device,
                artifact: artifact.clone(),
                columns,
                contract,
                _reservation: reservation,
            }),
        })
    }
    /// The explicit access and extent contract validated for each graph binding.
    pub fn contract(&self) -> &KernelContract {
        &self.inner.contract
    }
    /// Producing compiler's immutable artifact identity.
    pub fn artifact(&self) -> &crate::loom::Artifact {
        &self.inner.artifact
    }
}
