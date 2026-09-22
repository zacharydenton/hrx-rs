use super::{
    executable::{bridge_check, ffi::*},
    memory::DeviceUse,
    *,
};
use std::{ffi::CString, sync::Mutex, time::Duration};

/// A bounded logical range bound to an XDNA entry.
pub struct XdnaBinding<'a> {
    /// Shared native backing, admitted for the selected device.
    pub buffer: &'a Buffer,
    /// First byte of the logical range.
    pub offset: usize,
    /// Logical length, excluding allocation padding.
    pub length: usize,
}
/// Loaded, bound XDNA entry with a private context and establishing command.
/// Clone shares the same queue; each invocation must retire before reuse.
#[derive(Clone)]
pub struct XdnaProgram(Arc<Program>);
struct Program {
    bridge: Arc<Bridge>,
    device: Device,
    raw: *mut hrx_fabric_xdna_program,
    buffers: Vec<Buffer>,
    state: Mutex<State>,
}
#[derive(Default)]
struct State {
    pending: Option<u64>,
    leases: Vec<DeviceUse>,
    retired: u64,
}
// Every native operation is serialized by state. Binding and code storage are
// immutable after preparation and remain owned until checked retirement.
unsafe impl Send for Program {}
unsafe impl Sync for Program {}
impl Program {
    fn wait(&self, state: &mut State, submission: u64, timeout: u64) -> Result<bool> {
        if submission <= state.retired {
            return Ok(true);
        }
        let status = unsafe {
            self.bridge
                .hrx_fabric_xdna_wait(self.raw, submission, timeout)
        };
        if status == u64::from(AMDF_STATUS_CODE_DEADLINE_EXCEEDED) {
            return Ok(false);
        }
        check("XDNA retirement", status)?;
        state.retired = submission;
        state.pending = None;
        state.leases.clear();
        Ok(true)
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let retired = match state.pending.as_ref() {
            Some(submission) => {
                let submission = *submission;
                self.wait(&mut state, submission, 10_000_000_000)
                    .unwrap_or(false)
            }
            None => true,
        };
        if !retired || unsafe { self.bridge.hrx_fabric_xdna_close(self.raw) } != 0 {
            // No retirement/detach proof: quarantine the full ownership chain.
            std::mem::forget(std::mem::take(&mut state.leases));
            std::mem::forget(self.buffers.clone());
            std::mem::forget(self.device.clone());
            std::mem::forget(self.bridge.clone());
        }
    }
}
/// A checked native completion point retaining the program and all its bindings.
#[derive(Clone)]
pub struct XdnaCompletion {
    program: XdnaProgram,
    submission: u64,
}
impl XdnaCompletion {
    /// Check completion and inspect native command results without waiting.
    pub fn is_complete(&self) -> Result<bool> {
        self.wait_timeout(Duration::ZERO)
    }
    /// Wait under one native deadline. A timeout keeps all resources retained.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let mut state = self
            .program
            .0
            .state
            .lock()
            .map_err(|_| Error::DeviceLost("XDNA queue poisoned".into()))?;
        self.program.0.wait(
            &mut state,
            self.submission,
            timeout.as_nanos().min(u128::from(u64::MAX - 1)) as u64,
        )
    }
    /// Wait until checked retirement; failure does not permit storage reuse.
    pub fn wait(&self) -> Result<()> {
        let mut state = self
            .program
            .0
            .state
            .lock()
            .map_err(|_| Error::DeviceLost("XDNA queue poisoned".into()))?;
        self.program.0.wait(&mut state, self.submission, u64::MAX)?;
        Ok(())
    }
}
impl XdnaProgram {
    /// Publish the complete establishing command and retain all external backing.
    ///
    /// # Safety
    /// The caller orders conflicting accesses by other queues. The compiled
    /// program must obey each binding's declared range and access contract.
    pub unsafe fn dispatch(&self) -> Result<XdnaCompletion> {
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| Error::DeviceLost("XDNA queue poisoned".into()))?;
        if let Some(submission) = state.pending.as_ref() {
            let submission = *submission;
            if !self.0.wait(&mut state, submission, 0)? {
                return Err(Error::Busy("XDNA establishing command is in use".into()));
            }
        }
        for buffer in &self.0.buffers {
            match buffer.retain_use() {
                Ok(lease) => state.leases.push(lease),
                Err(error) => {
                    state.leases.clear();
                    return Err(error);
                }
            }
        }
        let mut submission = 0;
        if let Err(error) = check("XDNA submit", unsafe {
            self.0
                .bridge
                .hrx_fabric_xdna_submit(self.0.raw, &mut submission)
        }) {
            state.leases.clear();
            return Err(error);
        }
        state.pending = Some(submission);
        Ok(XdnaCompletion {
            program: self.clone(),
            submission,
        })
    }
}
impl Device {
    /// Load and bind a canonical XDNA image for the selected logical width.
    ///
    /// # Safety
    /// The native executable must be trusted and obey its binding contracts.
    pub unsafe fn prepare_xdna(
        &self,
        artifact: &crate::loom::Artifact,
        columns: u16,
        bindings: &[XdnaBinding<'_>],
    ) -> Result<XdnaProgram> {
        if self.endpoint().engine() != Engine::Xdna || artifact.target() != self.target().as_str() {
            return Err(Error::Unsupported("artifact and XDNA target differ".into()));
        }
        if !(1..=8).contains(&columns) || bindings.len() > u32::MAX as usize {
            return Err(Error::Message(
                "invalid XDNA context width or binding count".into(),
            ));
        }
        let mut native = Vec::with_capacity(bindings.len());
        let mut buffers = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if binding.length == 0
                || binding
                    .offset
                    .checked_add(binding.length)
                    .is_none_or(|end| end > binding.buffer.len())
            {
                return Err(Error::Message("XDNA binding range is out of bounds".into()));
            }
            let address = binding
                .buffer
                .device_address(self)?
                .checked_add(binding.offset as u64)
                .ok_or_else(|| Error::Message("XDNA binding address overflow".into()))?;
            native.push(hrx_fabric_xdna_binding {
                memory: binding.buffer.0.raw.cast(),
                host_pointer: binding.buffer.host_pointer().cast(),
                allocation_length: binding.buffer.len() as u64,
                offset: binding.offset as u64,
                length: binding.length as u64,
                address,
            });
            buffers.push(binding.buffer.clone());
        }
        buffers.sort_by_key(|buffer| Arc::as_ptr(&buffer.0) as usize);
        buffers.dedup_by(|a, b| Arc::ptr_eq(&a.0, &b.0));
        let api = &self.0.endpoint.0.instance.api;
        let bridge = api.load_bridge()?;
        let symbol = CString::new(artifact.symbol())
            .map_err(|_| Error::Message("symbol contains NUL".into()))?;
        let mut raw = ptr::null_mut();
        let status = unsafe {
            bridge.hrx_fabric_xdna_open(
                std::ptr::from_ref(api.core).cast(),
                std::ptr::from_ref(api.xdna).cast(),
                self.0.endpoint.0.instance.raw.cast(),
                self.0.endpoint.0.raw.cast(),
                self.0.raw.cast(),
                artifact.bytes().as_ptr(),
                artifact.bytes().len(),
                symbol.as_ptr(),
                columns,
                native.len() as u32,
                native.as_ptr(),
                &mut raw,
            )
        };
        // Copy diagnostics before closing a partial native ownership tree.
        let result = bridge_check(&bridge, status);
        if raw.is_null() {
            return result.and_then(|()| Err(missing("XDNA program")));
        }
        let state = Mutex::new(State {
            leases: Vec::with_capacity(buffers.len()),
            ..State::default()
        });
        let program = XdnaProgram(Arc::new(Program {
            bridge,
            device: self.clone(),
            raw,
            buffers,
            state,
        }));
        result?;
        Ok(program)
    }
}
