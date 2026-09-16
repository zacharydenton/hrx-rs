//! Resident Loom workloads built from reusable buffers, kernels and graphs.
//!
//! This module owns the bookkeeping that otherwise tends to be repeated by
//! every model: allocation handles, loaded-kernel indices, graph dependency
//! inference, replay, readback and graph-versus-direct timing. Model-specific
//! shape validation and kernel construction remain with the caller.
//!
//! ```no_run
//! use hrx::loom::{
//!     Specialization,
//!     model::{Command, Dispatch, ModelSession},
//! };
//!
//! fn prepare(source: &str) -> hrx::Result<()> {
//!     let mut model = ModelSession::open(0)?;
//!     let input = model.allocate(4096)?;
//!     let output = model.allocate(4096)?;
//!     // The application owns and audits its embedded kernel source.
//!     let kernels = unsafe {
//!         model.compile(&[(source, Specialization::new("transform"))])?
//!     };
//!     let commands = [Command::Dispatch(Dispatch::indices(
//!         kernels[0],
//!         [1024],
//!         [4, 1, 1],
//!         vec![input.read(), output.write()],
//!     ))];
//!     // These access declarations cover every byte the kernel can touch.
//!     unsafe { model.record(1024, &commands)? };
//!     model.upload(input, &[0; 4096])?;
//!     model.replay(1024)?;
//!     model.read(output, &mut [0; 4096])
//! }
//! ```

use super::{Compiler, CompilerOptions, Kernels, Specialization};
pub use crate::execution::Access;
use crate::{
    Buffer, Constants, Device, Error, GraphExec, Kernel, Node, Result, Stream, Target,
    benchmark::percentile,
    dependency_frontier::{Frontier, Use},
};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

/// A checked byte region in one allocation owned by a [`ModelSession`].
///
/// Regions are cheap handles and do not borrow the session, so every operation
/// checks the session identity before indexing storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    session: u64,
    allocation: usize,
    offset: usize,
    bytes: usize,
}

impl Region {
    /// Length of this region in bytes.
    #[must_use]
    pub fn len(self) -> usize {
        self.bytes
    }

    /// Whether this region is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bytes == 0
    }

    /// Byte offset from the start of the owning allocation.
    #[must_use]
    pub fn offset(self) -> usize {
        self.offset
    }

    /// Return a checked subregion relative to this one.
    pub fn slice(self, offset: usize, bytes: usize) -> Result<Self> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Message("model region size overflow".into()))?;
        if end > self.bytes {
            return Err(Error::Message("model region exceeds its parent".into()));
        }
        Ok(Self {
            offset: self
                .offset
                .checked_add(offset)
                .ok_or_else(|| Error::Message("model region offset overflow".into()))?,
            bytes,
            ..self
        })
    }

    /// Bind this region for read-only kernel access.
    #[must_use]
    pub fn read(self) -> Binding {
        Binding {
            region: self,
            access: Access::Read,
        }
    }

    /// Bind this region for write-only kernel access.
    #[must_use]
    pub fn write(self) -> Binding {
        Binding {
            region: self,
            access: Access::Write,
        }
    }

    /// Bind this region for kernel access that reads and writes existing bytes.
    #[must_use]
    pub fn read_write(self) -> Binding {
        Binding {
            region: self,
            access: Access::ReadWrite,
        }
    }
}

/// One dispatch binding and the complete access the kernel performs through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// Allocation region passed to the kernel.
    pub region: Region,
    /// Reads and writes performed through this argument.
    pub access: Access,
}

/// A loaded kernel in one [`ModelSession`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelId {
    session: u64,
    index: usize,
}

/// Scalar arguments for a model dispatch.
#[derive(Clone, Debug)]
pub enum Arguments {
    /// Loom indices whose 32- or 64-bit width is derived from export metadata.
    Indices(Vec<u32>),
    /// Explicitly packed scalar values for mixed-width signatures.
    Packed(Box<Constants>),
}

impl Arguments {
    /// Homogeneous Loom index values in declaration order.
    pub fn indices(values: impl IntoIterator<Item = u32>) -> Self {
        Self::Indices(values.into_iter().collect())
    }

    /// No scalar arguments.
    #[must_use]
    pub fn none() -> Self {
        Self::Packed(Box::new(Constants::new()))
    }

    /// Explicitly packed mixed-width scalar values.
    #[must_use]
    pub fn packed(constants: Constants) -> Self {
        Self::Packed(Box::new(constants))
    }
}

/// One fixed-shape kernel dispatch in a reusable model graph.
#[derive(Clone, Debug)]
pub struct Dispatch {
    /// Kernel returned by [`ModelSession::compile`].
    pub kernel: KernelId,
    /// Scalar values in kernel declaration order.
    pub arguments: Arguments,
    /// Workgroup counts in x, y and z.
    pub grid: [u32; 3],
    /// Buffer arguments in kernel declaration order.
    pub bindings: Vec<Binding>,
}

impl Dispatch {
    /// Construct a dispatch whose scalar arguments are homogeneous Loom indices.
    pub fn indices(
        kernel: KernelId,
        values: impl IntoIterator<Item = u32>,
        grid: [u32; 3],
        bindings: Vec<Binding>,
    ) -> Self {
        Self {
            kernel,
            arguments: Arguments::indices(values),
            grid,
            bindings,
        }
    }
}

/// A command recorded into a reusable model graph.
#[derive(Clone, Debug)]
pub enum Command {
    /// Fill a region with a repeating byte value.
    Fill {
        /// Destination region.
        region: Region,
        /// Byte pattern.
        value: u8,
    },
    /// Invoke a compiled kernel.
    Dispatch(Dispatch),
}

#[derive(Clone)]
enum PreparedCommand {
    Fill {
        region: Region,
        value: u8,
    },
    Dispatch {
        kernel: KernelId,
        constants: Box<Constants>,
        grid: [u32; 3],
        bindings: Vec<Binding>,
    },
}

struct PreparedGraph {
    executable: GraphExec,
    commands: Vec<PreparedCommand>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    Device,
    Shared,
}

struct Allocation {
    buffer: Buffer,
    placement: Placement,
}

/// A resident model's exclusive stream, storage, kernels and reusable graphs.
///
/// The session infers graph edges from [`Binding::access`] at byte-region
/// granularity. Independent regions may overlap; reads follow the last writer,
/// and writes follow both the last writer and every intervening reader.
pub struct ModelSession {
    id: u64,
    stream: Stream,
    allocations: Vec<Allocation>,
    kernels: Vec<Kernel>,
    cache: Kernels,
    graphs: HashMap<usize, PreparedGraph>,
    readback: Option<Buffer>,
    failed: bool,
}

impl ModelSession {
    /// Open a device and its shared Loom compiler using the device's actual target.
    pub fn open(index: i32) -> Result<Self> {
        let device = Device::open(index)?;
        let options = CompilerOptions {
            target: device.target().clone(),
            ..CompilerOptions::default()
        };
        let compiler = Compiler::shared(None, options)?;
        Self::with_compiler(device, compiler)
    }

    /// Open a device only when it reports the architecture validated by a model.
    pub fn open_for(index: i32, expected_target: &str) -> Result<Self> {
        let session = Self::open(index)?;
        if session.target().as_str() != expected_target {
            return Err(Error::Message(format!(
                "model kernels require {expected_target}, found {}",
                session.target().as_str()
            )));
        }
        Ok(session)
    }

    /// Create a session with an explicitly selected compiler.
    ///
    /// The compiler profile must match the device architecture; mismatches are
    /// rejected before any native code is loaded.
    pub fn with_compiler(device: Device, compiler: Compiler) -> Result<Self> {
        if compiler.target() != device.target() {
            return Err(Error::Message(format!(
                "compiler target {} does not match device target {}",
                compiler.target().as_str(),
                device.target().as_str()
            )));
        }
        let stream = device.stream()?;
        Ok(Self {
            id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
            stream,
            allocations: Vec::new(),
            kernels: Vec::new(),
            cache: Kernels::new(compiler),
            graphs: HashMap::new(),
            readback: None,
            failed: false,
        })
    }

    /// Architecture selected by the device and compiler.
    #[must_use]
    pub fn target(&self) -> &Target {
        self.stream.target()
    }

    /// Allocate device-local model storage.
    pub fn allocate(&mut self, bytes: usize) -> Result<Region> {
        self.allocate_with(bytes, Placement::Device)
    }

    /// Allocate coherent host-local storage visible to the device.
    pub fn allocate_shared(&mut self, bytes: usize) -> Result<Region> {
        self.allocate_with(bytes, Placement::Shared)
    }

    fn allocate_with(&mut self, bytes: usize, placement: Placement) -> Result<Region> {
        self.usable()?;
        let buffer = match placement {
            Placement::Device => self.stream.allocate(bytes)?,
            Placement::Shared => self.stream.allocate_shared(bytes)?,
        };
        let allocation = self.allocations.len();
        self.allocations.push(Allocation { buffer, placement });
        Ok(Region {
            session: self.id,
            allocation,
            offset: 0,
            bytes,
        })
    }

    /// Allocate device-local storage and enqueue its initial contents.
    pub fn weight(&mut self, bytes: &[u8]) -> Result<Region> {
        let region = self.allocate(bytes.len())?;
        self.upload(region, bytes)?;
        Ok(region)
    }

    /// Compile and load a batch of trusted embedded Loom kernels.
    ///
    /// Returned IDs follow request order and remain valid for this session.
    ///
    /// # Safety
    /// Every source is compiled to native code. The caller must trust it not to
    /// violate the buffer and scalar contracts later supplied to [`Self::record`].
    pub unsafe fn compile(
        &mut self,
        specifications: &[(&str, Specialization)],
    ) -> Result<Vec<KernelId>> {
        self.usable()?;
        let pending = specifications
            .iter()
            .map(|(source, specialization)| self.cache.request(source, specialization))
            .collect::<Result<Vec<_>>>()?;
        // Safety: required from this method's caller for every requested source.
        unsafe { self.cache.build(&self.stream)? };
        let mut loaded = Vec::with_capacity(pending.len());
        for request in pending {
            // Safety: the pending request is one of the sources vouched for above.
            loaded.push(unsafe { request.resolve(&self.stream)? }.clone());
        }
        self.synchronize()?;
        let first = self.kernels.len();
        self.kernels.extend(loaded);
        Ok((first..self.kernels.len())
            .map(|index| KernelId {
                session: self.id,
                index,
            })
            .collect())
    }

    /// Whether a graph has already been prepared under `key`.
    #[must_use]
    pub fn is_recorded(&self, key: usize) -> bool {
        self.graphs.contains_key(&key)
    }

    /// Record commands once, inferring the minimal buffer hazard dependencies.
    ///
    /// Keys are unique within the session; use [`Self::is_recorded`] when a
    /// shape-dependent preparation path can run more than once.
    ///
    /// # Safety
    /// Each dispatch must match its compiled kernel: scalar types, binding
    /// order, accessible extents and declared access modes must describe every
    /// read and write the native kernel can perform. Regions that may alias
    /// outside this session require the caller to provide equivalent ordering.
    pub unsafe fn record(&mut self, key: usize, commands: &[Command]) -> Result<()> {
        self.usable()?;
        if commands.is_empty() {
            return Err(Error::Message("cannot record an empty model graph".into()));
        }
        if self.graphs.contains_key(&key) {
            return Err(Error::Message(format!(
                "model graph key {key} is already recorded"
            )));
        }
        let prepared = self.prepare(commands)?;
        let mut graph = self.stream.graph()?;
        let mut frontier = Frontier::default();
        let mut nodes: Vec<Node> = Vec::with_capacity(prepared.len());
        for (index, command) in prepared.iter().enumerate() {
            let dependencies = frontier.dependencies(index, command.uses());
            let after = dependencies
                .into_iter()
                .map(|dependency| nodes[dependency])
                .collect::<Vec<_>>();
            let node = match command {
                PreparedCommand::Fill { region, value } => {
                    graph.fill(&after, self.view(*region)?, *value)?
                }
                PreparedCommand::Dispatch {
                    kernel,
                    constants,
                    grid,
                    bindings,
                } => {
                    let kernel = self.kernel(*kernel)?;
                    let views = self.views(bindings)?;
                    // Safety: required from this method's caller for this dispatch.
                    unsafe {
                        graph.dispatch(
                            &after,
                            kernel,
                            *grid,
                            kernel.info().workgroup_size,
                            constants,
                            &views,
                        )?
                    }
                }
            };
            nodes.push(node);
        }
        let executable = graph.finish()?;
        self.graphs.insert(
            key,
            PreparedGraph {
                executable,
                commands: prepared,
            },
        );
        Ok(())
    }

    /// Enqueue a host-to-device transfer at the start of a region.
    pub fn upload(&mut self, region: Region, bytes: &[u8]) -> Result<()> {
        self.usable()?;
        if bytes.len() > region.bytes {
            return Err(Error::Message("upload exceeds model region".into()));
        }
        self.validate_region(region)?;
        let allocation = &self.allocations[region.allocation].buffer;
        let view = allocation
            .try_slice(region.offset, region.bytes)?
            .slice(0, bytes.len())?;
        let result = self.stream.upload(view, bytes);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Enqueue one replay of a prepared graph.
    pub fn replay(&mut self, key: usize) -> Result<()> {
        self.usable()?;
        let graph = self
            .graphs
            .get_mut(&key)
            .ok_or_else(|| Error::Message(format!("model graph key {key} is not recorded")))?;
        let result = self.stream.launch(&mut graph.executable);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Wait for all queued session work.
    pub fn synchronize(&mut self) -> Result<()> {
        self.usable()?;
        let result = self.stream.synchronize();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Preallocate reusable coherent storage for combined readback.
    ///
    /// Readback also grows this storage automatically; reserving avoids an
    /// allocation on the first inference.
    pub fn reserve_readback(&mut self, bytes: usize) -> Result<()> {
        self.usable()?;
        if self
            .readback
            .as_ref()
            .is_none_or(|buffer| buffer.bytes() < bytes)
        {
            self.readback = Some(self.stream.allocate_shared(bytes)?);
        }
        Ok(())
    }

    /// Wait for queued work and copy one region into caller-owned bytes.
    pub fn read(&mut self, region: Region, bytes: &mut [u8]) -> Result<()> {
        self.read_many(&mut [(region, bytes)])
    }

    /// Wait once and copy several regions into caller-owned byte slices.
    pub fn read_many(&mut self, outputs: &mut [(Region, &mut [u8])]) -> Result<()> {
        self.usable()?;
        let mut total = 0usize;
        let mut all_shared = true;
        for (region, output) in outputs.iter() {
            self.validate_region(*region)?;
            if output.len() > region.bytes {
                return Err(Error::Message("readback exceeds model region".into()));
            }
            total = total
                .checked_add(output.len())
                .ok_or_else(|| Error::Message("readback size overflow".into()))?;
            all_shared &= self.allocations[region.allocation].placement == Placement::Shared;
        }
        if all_shared {
            self.synchronize()?;
            for (region, output) in outputs.iter_mut() {
                if output.is_empty() {
                    continue;
                }
                let pointer = self.allocations[region.allocation]
                    .buffer
                    .device_ptr()?
                    .cast::<u8>();
                if pointer.is_null() {
                    self.failed = true;
                    return Err(Error::Message("null shared model pointer".into()));
                }
                // The allocation is host-local and coherent, synchronization
                // completed above, and private session storage cannot alias output.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pointer.add(region.offset),
                        output.as_mut_ptr(),
                        output.len(),
                    );
                }
            }
            return Ok(());
        }
        self.reserve_readback(total)?;
        {
            let readback = self.readback.as_ref().expect("reserved above");
            let mut offset = 0;
            for (region, output) in outputs.iter() {
                if !output.is_empty() {
                    let source = self.allocations[region.allocation]
                        .buffer
                        .try_slice(region.offset, output.len())?;
                    if let Err(error) = self
                        .stream
                        .copy(readback.try_slice(offset, output.len())?, source)
                    {
                        self.failed = true;
                        return Err(error);
                    }
                }
                offset += output.len();
            }
        }
        self.synchronize()?;
        if total == 0 {
            return Ok(());
        }
        let readback = self.readback.as_ref().expect("reserved above");
        let pointer = readback.device_ptr()?.cast::<u8>();
        if pointer.is_null() {
            self.failed = true;
            return Err(Error::Message("null model readback pointer".into()));
        }
        let mut offset = 0;
        for (_, output) in outputs.iter_mut() {
            if !output.is_empty() {
                // The private coherent readback was populated and synchronized
                // above, and distinct mutable slices cannot alias it.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pointer.add(offset),
                        output.as_mut_ptr(),
                        output.len(),
                    );
                }
            }
            offset += output.len();
        }
        Ok(())
    }

    /// Compare synchronized graph replay with the equivalent direct commands.
    ///
    /// Transfers are excluded. Ten untimed alternating warm-up pairs precede
    /// the requested samples to limit ordering and cache bias.
    pub fn benchmark(&mut self, key: usize, samples: usize) -> Result<ForwardTimings> {
        if samples < 10 {
            return Err(Error::Message(
                "use at least ten model timing samples".into(),
            ));
        }
        self.usable()?;
        let commands = self
            .graphs
            .get(&key)
            .ok_or_else(|| Error::Message(format!("model graph key {key} is not recorded")))?
            .commands
            .clone();
        let mut times = [Vec::with_capacity(samples), Vec::with_capacity(samples)];
        for iteration in 0..samples + 10 {
            for mode in [iteration % 2, 1 - iteration % 2] {
                let start = Instant::now();
                if mode == 0 {
                    self.replay(key)?;
                } else {
                    // Safety: these commands were accepted by unsafe `record`.
                    let result = unsafe { self.execute_direct(&commands) };
                    if result.is_err() {
                        self.failed = true;
                    }
                    result?;
                }
                self.synchronize()?;
                if iteration >= 10 {
                    times[mode].push(start.elapsed().as_secs_f64() * 1_000.0);
                }
            }
        }
        let [graph, direct] = times;
        Ok(ForwardTimings {
            samples,
            graph: Distribution::from_samples(graph),
            direct: Distribution::from_samples(direct),
        })
    }

    /// Profile each prepared command as an isolated synchronized submission.
    ///
    /// This is a diagnostic view of per-command host latency, not an estimate
    /// of graph contribution: it includes dispatch and synchronization overhead
    /// and removes graph overlap. One unmeasured warm-up precedes each command's
    /// samples. Commands execute repeatedly and may change resident outputs.
    pub fn profile_commands(&mut self, key: usize, samples: usize) -> Result<Vec<Distribution>> {
        if samples == 0 {
            return Err(Error::Message(
                "use at least one command timing sample".into(),
            ));
        }
        self.usable()?;
        let commands = self
            .graphs
            .get(&key)
            .ok_or_else(|| Error::Message(format!("model graph key {key} is not recorded")))?
            .commands
            .clone();
        let mut distributions = Vec::with_capacity(commands.len());
        for command in commands {
            // Safety: this command was accepted by unsafe `record`.
            let warmup = unsafe { self.execute_direct(std::slice::from_ref(&command)) };
            if warmup.is_err() {
                self.failed = true;
            }
            warmup?;
            self.synchronize()?;
            let mut times = Vec::with_capacity(samples);
            for _ in 0..samples {
                let start = Instant::now();
                // Safety: this command was accepted by unsafe `record`.
                let result = unsafe { self.execute_direct(std::slice::from_ref(&command)) };
                if result.is_err() {
                    self.failed = true;
                }
                result?;
                self.synchronize()?;
                times.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            distributions.push(Distribution::from_samples(times));
        }
        Ok(distributions)
    }

    fn usable(&self) -> Result<()> {
        if self.failed {
            Err(Error::Message(
                "model session is unusable after a GPU failure".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn validate_region(&self, region: Region) -> Result<()> {
        if region.session != self.id {
            return Err(Error::Message(
                "model region belongs to another session".into(),
            ));
        }
        let allocation = self
            .allocations
            .get(region.allocation)
            .ok_or_else(|| Error::Message("invalid model allocation".into()))?;
        let end = region
            .offset
            .checked_add(region.bytes)
            .ok_or_else(|| Error::Message("model region size overflow".into()))?;
        if end > allocation.buffer.bytes() {
            return Err(Error::Message("model region exceeds its allocation".into()));
        }
        Ok(())
    }

    fn view(&self, region: Region) -> Result<crate::View<'_>> {
        self.validate_region(region)?;
        self.allocations[region.allocation]
            .buffer
            .try_slice(region.offset, region.bytes)
    }

    fn views<'a>(&'a self, bindings: &[Binding]) -> Result<Vec<crate::View<'a>>> {
        bindings
            .iter()
            .map(|binding| self.view(binding.region))
            .collect()
    }

    fn kernel(&self, id: KernelId) -> Result<&Kernel> {
        if id.session != self.id {
            return Err(Error::Message(
                "model kernel belongs to another session".into(),
            ));
        }
        self.kernels
            .get(id.index)
            .ok_or_else(|| Error::Message("invalid model kernel".into()))
    }

    fn prepare(&self, commands: &[Command]) -> Result<Vec<PreparedCommand>> {
        commands
            .iter()
            .map(|command| match command {
                Command::Fill { region, value } => {
                    self.validate_region(*region)?;
                    if region.is_empty() {
                        return Err(Error::Message("cannot record an empty model fill".into()));
                    }
                    Ok(PreparedCommand::Fill {
                        region: *region,
                        value: *value,
                    })
                }
                Command::Dispatch(dispatch) => {
                    let kernel = self.kernel(dispatch.kernel)?;
                    for binding in &dispatch.bindings {
                        self.validate_region(binding.region)?;
                    }
                    let constants = match &dispatch.arguments {
                        Arguments::Indices(values) => Box::new(Constants::indices(kernel, values)?),
                        Arguments::Packed(constants) => constants.clone(),
                    };
                    Ok(PreparedCommand::Dispatch {
                        kernel: dispatch.kernel,
                        constants,
                        grid: dispatch.grid,
                        bindings: dispatch.bindings.clone(),
                    })
                }
            })
            .collect()
    }

    unsafe fn execute_direct(&self, commands: &[PreparedCommand]) -> Result<()> {
        for command in commands {
            match command {
                PreparedCommand::Fill { region, value } => {
                    self.stream.fill(self.view(*region)?, *value)?;
                }
                PreparedCommand::Dispatch {
                    kernel,
                    constants,
                    grid,
                    bindings,
                } => {
                    let kernel = self.kernel(*kernel)?;
                    let views = self.views(bindings)?;
                    // Safety: inherited from this method's caller.
                    unsafe {
                        self.stream.dispatch(
                            kernel,
                            *grid,
                            kernel.info().workgroup_size,
                            constants,
                            &views,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl PreparedCommand {
    fn uses(&self) -> Vec<Use> {
        match self {
            Self::Fill { region, .. } => vec![Use::new(*region, Access::Write)],
            Self::Dispatch { bindings, .. } => bindings
                .iter()
                .map(|binding| Use::new(binding.region, binding.access))
                .collect(),
        }
    }
}

impl Use {
    fn new(region: Region, access: Access) -> Self {
        Self {
            allocation: region.allocation,
            range: region.offset..region.offset + region.bytes,
            access,
        }
    }
}

/// Synchronized host timings for a resident forward pass, excluding transfers.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ForwardTimings {
    /// Number of measured samples for each mode.
    pub samples: usize,
    /// Reusable graph replay timings.
    pub graph: Distribution,
    /// Equivalent direct stream dispatch timings.
    pub direct: Distribution,
}

/// Millisecond distribution measured by the host around submission and completion.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Distribution {
    /// Nearest-rank 50th percentile.
    pub median_ms: f64,
    /// Nearest-rank 95th percentile.
    pub p95_ms: f64,
}

impl Distribution {
    fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.sort_by(f64::total_cmp);
        Self {
            median_ms: percentile(&samples, 50),
            p95_ms: percentile(&samples, 95),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    fn region(allocation: usize, range: Range<usize>) -> Region {
        Region {
            session: 1,
            allocation,
            offset: range.start,
            bytes: range.end - range.start,
        }
    }

    #[test]
    fn region_slices_are_relative_and_checked() {
        let whole = region(2, 100..200);
        let slice = whole.slice(25, 50).unwrap();
        assert_eq!(slice.offset(), 125);
        assert_eq!(slice.len(), 50);
        assert!(whole.slice(101, 0).is_err());
        assert!(whole.slice(usize::MAX, 2).is_err());
    }

    #[test]
    fn distributions_use_nearest_rank_quantiles() {
        let distribution = Distribution::from_samples((1..=100).map(f64::from).collect());
        assert_eq!(distribution.median_ms, 50.0);
        assert_eq!(distribution.p95_ms, 95.0);
    }
}
