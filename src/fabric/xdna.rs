use super::{
    executable::{bridge_check, ffi::*},
    memory::DeviceUse,
    *,
};
use std::{
    collections::VecDeque,
    ffi::CString,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

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
/// Clones share one queue and up to 128 accepted invocations. Bindings remain
/// leased until every outstanding invocation has retired.
#[derive(Clone)]
pub struct XdnaProgram(Arc<Program>);
struct Program {
    bridge: Arc<Bridge>,
    device: Device,
    raw: *mut hrx_fabric_xdna_program,
    buffers: Vec<Buffer>,
    state: Mutex<State>,
    retired: AtomicU64,
    charges: Vec<Arc<crate::residency::MemoryReservation>>,
}
#[derive(Default)]
struct State {
    pending: VecDeque<u64>,
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
        while state
            .pending
            .front()
            .is_some_and(|point| *point <= submission)
        {
            state.pending.pop_front();
        }
        if state.pending.is_empty() {
            state.leases.clear();
        }
        self.retired.store(submission, Ordering::Release);
        Ok(true)
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let retired = match state.pending.back() {
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
            std::mem::forget(std::mem::take(&mut self.charges));
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
    /// Read cached retirement without locking, native queries, or progress work.
    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.program.0.retired.load(Ordering::Acquire) >= self.submission)
    }
    /// Inspect native command results and advance retirement without waiting.
    pub fn refresh(&self) -> Result<bool> {
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
    /// Compose trusted native transaction records around the bound invocation.
    /// This consumes an exclusive, never-submitted program. The original command
    /// body is preserved; only its transaction size/count header is updated.
    /// Additional addressed buffers are retained through checked retirement.
    /// # Safety
    /// Records must own disjoint native resources, preserve the establishing
    /// invocation, and drain every added DMA/worker before command completion.
    /// Every address they embed must belong to `retained` or existing bindings.
    pub unsafe fn wrap_transaction(
        mut self,
        prefix: &[u8],
        prefix_operations: u32,
        suffix: &[u8],
        suffix_operations: u32,
        retained: &[Buffer],
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Self> {
        let program = Arc::get_mut(&mut self.0)
            .ok_or_else(|| Error::Busy("XDNA command is shared".into()))?;
        let state = program
            .state
            .get_mut()
            .map_err(|_| Error::DeviceLost("XDNA queue poisoned".into()))?;
        if !state.pending.is_empty() || state.retired != 0 {
            return Err(Error::Busy(
                "XDNA command has already been submitted".into(),
            ));
        }
        for buffer in retained {
            buffer.device_address(&program.device)?;
        }
        let mut length = 0;
        bridge_check(&program.bridge, unsafe {
            program.bridge.hrx_fabric_xdna_command_copy(
                program.raw,
                ptr::null_mut(),
                0,
                &mut length,
            )
        })?;
        let mut original = vec![0; length];
        bridge_check(&program.bridge, unsafe {
            program.bridge.hrx_fabric_xdna_command_copy(
                program.raw,
                original.as_mut_ptr(),
                length,
                &mut length,
            )
        })?;
        let bytes = compose_transaction(
            &original,
            prefix,
            prefix_operations,
            suffix,
            suffix_operations,
        )?;
        if let Some(budget) = budget {
            program.charges.push(Arc::new(budget.reserve(bytes.len())?));
        }
        // Retain before native mutation: any failure still owns every dependency.
        program.buffers.extend_from_slice(retained);
        program
            .buffers
            .sort_by_key(|buffer| Arc::as_ptr(&buffer.0) as usize);
        program.buffers.dedup_by(|a, b| Arc::ptr_eq(&a.0, &b.0));
        state.leases.reserve(program.buffers.len());
        bridge_check(&program.bridge, unsafe {
            program
                .bridge
                .hrx_fabric_xdna_command_replace(program.raw, bytes.as_ptr(), bytes.len())
        })?;
        Ok(self)
    }
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
        if state.pending.len() == 128 {
            let oldest = *state.pending.front().unwrap();
            if !self.0.wait(&mut state, oldest, 0)? {
                return Err(Error::Busy(
                    "XDNA pending invocation capacity exhausted".into(),
                ));
            }
        }
        if state.leases.is_empty() {
            for buffer in &self.0.buffers {
                match buffer.retain_use() {
                    Ok(lease) => state.leases.push(lease),
                    Err(error) => {
                        state.leases.clear();
                        return Err(error);
                    }
                }
            }
        }
        let mut submission = 0;
        if let Err(error) = check("XDNA submit", unsafe {
            self.0
                .bridge
                .hrx_fabric_xdna_submit(self.0.raw, &mut submission)
        }) {
            if state.pending.is_empty() {
                state.leases.clear();
            }
            return Err(error);
        }
        state.pending.push_back(submission);
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
            pending: VecDeque::with_capacity(128),
            leases: Vec::with_capacity(buffers.len()),
            ..State::default()
        });
        let program = XdnaProgram(Arc::new(Program {
            bridge,
            device: self.clone(),
            raw,
            buffers,
            state,
            retired: AtomicU64::new(0),
            charges: Vec::new(),
        }));
        result?;
        Ok(program)
    }
}

fn compose_transaction(
    original: &[u8],
    prefix: &[u8],
    prefix_operations: u32,
    suffix: &[u8],
    suffix_operations: u32,
) -> Result<Vec<u8>> {
    let invalid = || Error::Message("invalid or oversized AIE2P transaction 0.1".into());
    if original.len() < 16
        || original[..4] != [0, 1, 4, 6]
        || original[5] != 1
        || original[4] == 0
        || original[4] > 8
        || u32::from_le_bytes(original[12..16].try_into().unwrap()) as usize != original.len()
    {
        return Err(invalid());
    }
    let length = original
        .len()
        .checked_add(prefix.len())
        .and_then(|n| n.checked_add(suffix.len()))
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or_else(invalid)?;
    let operations = u32::from_le_bytes(original[8..12].try_into().unwrap())
        .checked_add(prefix_operations)
        .and_then(|n| n.checked_add(suffix_operations))
        .ok_or_else(invalid)?;
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(&original[..16]);
    bytes.extend_from_slice(prefix);
    bytes.extend_from_slice(&original[16..]);
    bytes.extend_from_slice(suffix);
    bytes[8..12].copy_from_slice(&operations.to_le_bytes());
    bytes[12..16].copy_from_slice(&(length as u32).to_le_bytes());
    Ok(bytes)
}

#[cfg(test)]
mod transaction_tests {
    use super::compose_transaction;

    fn original() -> Vec<u8> {
        let mut command = vec![0, 1, 4, 6, 1, 1, 0, 0];
        command.extend_from_slice(&2u32.to_le_bytes());
        command.extend_from_slice(&24u32.to_le_bytes());
        command.extend_from_slice(&[11, 22, 33, 44, 55, 66, 77, 88]);
        command
    }

    #[test]
    fn wrapping_preserves_original_body_and_updates_only_extent_and_count() {
        let original = original();
        let result = compose_transaction(&original, &[9, 8, 7, 6], 1, &[5, 4, 3, 2], 3).unwrap();
        assert_eq!(&result[..8], &original[..8]);
        assert_eq!(u32::from_le_bytes(result[8..12].try_into().unwrap()), 6);
        assert_eq!(u32::from_le_bytes(result[12..16].try_into().unwrap()), 32);
        assert_eq!(&result[16..20], &[9, 8, 7, 6]);
        assert_eq!(&result[20..28], &original[16..]);
        assert_eq!(&result[28..], &[5, 4, 3, 2]);
    }

    #[test]
    fn wrapping_rejects_malformed_headers_and_operation_overflow() {
        let original = original();
        for length in 0..original.len() {
            assert!(compose_transaction(&original[..length], &[], 0, &[], 0).is_err());
        }
        for (offset, value) in [
            (0, 1),
            (1, 2),
            (2, 5),
            (3, 7),
            (4, 0),
            (4, 9),
            (5, 2),
            (12, 16),
        ] {
            let mut malformed = original.clone();
            malformed[offset] = value;
            assert!(compose_transaction(&malformed, &[], 0, &[], 0).is_err());
        }
        assert!(compose_transaction(&original, &[], u32::MAX, &[], 0).is_err());
        assert!(compose_transaction(&original, &[], u32::MAX - 2, &[], 1).is_err());
    }
}
