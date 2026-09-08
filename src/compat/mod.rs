//! gfx1151 dispatch through the public HRX C API.
//!
//! A process-wide [`Device`] owns an ordered stream and a registry of live
//! allocations. Execution barriers order dispatches and copies; host transfers
//! synchronize. [`DevicePtr`] values are device addresses, never host pointers.
//! Copy operations resolve allocation spans through the registry; kernel callers
//! must ensure their pointer arguments and shapes describe valid buffers.
pub use crate::sys;

mod args;
mod device;
mod kernel;

pub use args::Args;
pub use device::{Buffer, Device, DevicePtr, Scope, device, try_device};
pub use kernel::Kernel;

pub use crate::{Error, Result};
pub(crate) fn check(status: sys::Status) -> Result<()> {
    crate::runtime::check(status, "HRX")
}
