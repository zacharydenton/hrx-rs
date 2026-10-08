//! One outstanding invocation per device, with explicit startup and terminal joins.
use super::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Dedicated host-written startup allocation: WAIT=0, RUN=1, ABORT=2.
/// Both trusted programs must poll it before touching their shared payload.
pub struct ResidentStartup {
    buffer: Buffer,
    publication: Vec<PreparedHostVisibility>,
}
impl ResidentStartup {
    /// Allocate an isolated startup cache line accessible to both devices.
    pub fn new(
        gpu: &Device,
        npu: &Device,
        budget: Option<&crate::residency::MemoryBudget>,
    ) -> Result<Self> {
        if gpu.endpoint().engine() != Engine::Gpu || npu.endpoint().engine() != Engine::Xdna {
            return Err(Error::Unsupported(
                "resident startup requires a GPU and an NPU".into(),
            ));
        }
        let devices = [gpu.clone(), npu.clone()];
        let fabric = gpu.fabric();
        let profile = fabric
            .allocation_profiles(&devices, true)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                Error::Unsupported("resident startup requires coherent GPU backing".into())
            })?;
        let buffer = profile.allocate(64, budget)?;
        buffer.zero()?;
        let mut publication = Vec::new();
        for device in [gpu, npu] {
            for family in device.endpoint().queue_capabilities()? {
                if matches!(
                    family.command,
                    QueueCommand::Pm4 | QueueCommand::Aql | QueueCommand::Xdna
                ) {
                    let relation = buffer
                        .visibility(MemorySite::Host, MemorySite::Device(device, family.ordinal))?;
                    publication
                        .push(relation.prepare_release_host(std::slice::from_ref(&(0..64)))?);
                }
            }
        }
        Ok(Self {
            buffer,
            publication,
        })
    }
    /// Borrow the startup allocation while constructing both native invocations.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    fn publish(&self, decision: u32) -> Result<()> {
        // Trusted protocol: this dedicated allocation has exactly one host writer,
        // and devices only poll its first DWORD. Device-use leases intentionally
        // remain live while publishing this host-to-device control transition.
        unsafe {
            (&*self.buffer.host_pointer().cast::<AtomicU32>()).store(decision, Ordering::Release);
            for action in &self.publication {
                action.execute()?;
            }
            Ok(())
        }
    }
}
/// A prepared native GPU participant. Its command owns all addressed storage.
pub enum ResidentGpu {
    /// Direct PM4 compute command.
    Pm4(PreparedGpu),
    /// Native AQL dispatch.
    Aql(PreparedAql),
}
enum GpuDone {
    Pm4(Completion),
    Aql(AqlCompletion),
}
impl GpuDone {
    fn wait(&self, timeout: Duration) -> Result<bool> {
        match self {
            Self::Pm4(done) => done.wait_timeout(timeout),
            Self::Aql(done) => done.wait_timeout(timeout),
        }
    }
    fn complete(&self) -> Result<bool> {
        match self {
            Self::Pm4(done) => done.is_complete(),
            Self::Aql(done) => Ok(done.is_complete()),
        }
    }
}
struct Owners {
    startup: ResidentStartup,
    gpu: ResidentGpu,
    npu: XdnaProgram,
    gpu_done: Option<GpuDone>,
    npu_done: Option<XdnaCompletion>,
}
/// A single resident GPU–NPU exchange, including partial-admission rollback.
/// Submit both participants in either order, publish [`Self::start`], then join
/// both native completions. No host action is performed between device rounds.
/// Dropping before RUN publishes ABORT. A failed/timed-out teardown quarantines
/// the complete session instead of releasing storage reachable by either device.
pub struct ResidentSession {
    owners: Option<Owners>,
    decision: Option<u32>,
    poisoned: bool,
}
impl ResidentSession {
    /// Own two immutable native invocations and their startup protocol.
    /// # Safety
    /// Both programs must read WAIT/RUN/ABORT from exactly `startup`, never write
    /// that allocation, and terminate without peer progress after ABORT. Once RUN
    /// is observed, all exchange generations, credits and DMA drains must remain
    /// inside these invocations. Their commands must retain every reachable owner.
    /// All payload backing and shader memory operations must satisfy the native
    /// directional visibility contracts (including coherent GPU attachments where
    /// required). The caller must exclusively own payload until both retire.
    pub unsafe fn new(startup: ResidentStartup, gpu: ResidentGpu, npu: XdnaProgram) -> Self {
        Self {
            owners: Some(Owners {
                startup,
                gpu,
                npu,
                gpu_done: None,
                npu_done: None,
            }),
            decision: None,
            poisoned: false,
        }
    }
    fn ready(&self) -> Result<()> {
        if self.poisoned || self.decision.is_some() {
            return Err(Error::Busy(
                "resident startup is already decided or failed".into(),
            ));
        }
        Ok(())
    }
    /// Accept the GPU participant once. A native rejection publishes ABORT for its peer.
    pub fn submit_gpu(&mut self) -> Result<()> {
        self.ready()?;
        let owners = self.owners.as_mut().unwrap();
        if owners.gpu_done.is_some() {
            return Err(Error::Busy("GPU participant already submitted".into()));
        }
        let result = unsafe {
            match &owners.gpu {
                ResidentGpu::Pm4(command) => command.dispatch().map(GpuDone::Pm4),
                ResidentGpu::Aql(command) => command.dispatch().map(GpuDone::Aql),
            }
        };
        match result {
            Ok(done) => {
                owners.gpu_done = Some(done);
                Ok(())
            }
            Err(error) => {
                let _ = self.abort();
                Err(error)
            }
        }
    }
    /// Accept the NPU participant once. A native rejection publishes ABORT for its peer.
    pub fn submit_npu(&mut self) -> Result<()> {
        self.ready()?;
        let owners = self.owners.as_mut().unwrap();
        if owners.npu_done.is_some() {
            return Err(Error::Busy("NPU participant already submitted".into()));
        }
        match unsafe { owners.npu.dispatch() } {
            Ok(done) => {
                owners.npu_done = Some(done);
                Ok(())
            }
            Err(error) => {
                let _ = self.abort();
                Err(error)
            }
        }
    }
    fn decide(&mut self, value: u32) -> Result<()> {
        self.ready()?;
        // Never publish another decision if the cache transition itself fails.
        self.decision = Some(value);
        let result = self.owners.as_ref().unwrap().startup.publish(value);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    /// Publish RUN only after both native submissions have been accepted.
    pub fn start(&mut self) -> Result<()> {
        self.ready()?;
        let owners = self.owners.as_ref().unwrap();
        if owners.gpu_done.is_none() || owners.npu_done.is_none() {
            return Err(Error::Busy(
                "both resident participants must be accepted before RUN".into(),
            ));
        }
        self.decide(1)
    }
    /// Publish pre-start ABORT. This is not cancellation of a running exchange.
    pub fn abort(&mut self) -> Result<()> {
        self.decide(2)
    }
    /// Inspect cached retirement of every accepted participant, without progress.
    pub fn is_complete(&self) -> Result<bool> {
        let owners = self.owners.as_ref().unwrap();
        Ok(self.decision.is_some()
            && owners
                .gpu_done
                .as_ref()
                .map_or(Ok(true), GpuDone::complete)?
            && owners
                .npu_done
                .as_ref()
                .map_or(Ok(true), XdnaCompletion::is_complete)?)
    }
    /// Join both native invocations under one deadline. Timeout keeps ownership.
    pub fn wait_timeout(&mut self, timeout: Duration) -> Result<bool> {
        if self.decision.is_none() {
            return Err(Error::Busy("resident startup remains undecided".into()));
        }
        let start = Instant::now();
        let owners = self.owners.as_ref().unwrap();
        let gpu = owners
            .gpu_done
            .as_ref()
            .map_or(Ok(true), |done| done.wait(timeout));
        // Always attempt both joins, even when one native device reports failure.
        let npu = owners.npu_done.as_ref().map_or(Ok(true), |done| {
            done.wait_timeout(timeout.saturating_sub(start.elapsed()))
        });
        if gpu.is_err() || npu.is_err() {
            self.poisoned = true;
        }
        Ok(gpu? && npu?)
    }
}
impl Drop for ResidentSession {
    fn drop(&mut self) {
        if self.decision.is_none() {
            let _ = self.abort();
        }
        let retired = self.wait_timeout(Duration::from_secs(10)).unwrap_or(false);
        if !retired || self.poisoned {
            std::mem::forget(self.owners.take());
        }
    }
}
