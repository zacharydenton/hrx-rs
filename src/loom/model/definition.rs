use super::*;
use crate::{
    execution::{BindingContract, BufferView, GpuKernel, Graph, KernelContract, MemoryPlacement},
    inference::{InferenceGraph, ModelContext, PreparedModel},
    tensor::{DeviceTensor, TensorDesc},
};

#[derive(Clone)]
enum Storage {
    Weight(crate::execution::Buffer),
    Scratch(usize, Placement),
}

/// Immutable model code and weights in a shared context. Shape-specific plans
/// allocate private scratch and graph storage for each bounded inference slot.
pub struct ModelDefinition {
    id: u64,
    context: ModelContext,
    allocations: Vec<Storage>,
    kernels: Vec<Kernel>,
}

/// Validated, shape-specific model commands with shared immutable weights.
/// Record into a caller-owned graph to connect stages without copies, or build
/// a standalone bounded inference pool with [`Self::prepare`].
pub struct ModelFragment {
    context: ModelContext,
    allocations: Vec<Storage>,
    extents: Vec<usize>,
    commands: Vec<(Option<GpuKernel>, Vec<Region>, Option<u8>)>,
    inputs: Vec<(Region, TensorDesc)>,
    outputs: Vec<(Region, TensorDesc)>,
    reuse_private_scratch: bool,
}

impl ModelSession {
    /// Transfer code and immutable weights to a coordinated context. All raw
    /// graphs are destroyed after synchronization, before adopting allocations.
    /// Scratch and host staging are discarded; prepared slots initialize their
    /// own scratch. Values needed across inferences must use [`Self::weight`].
    pub fn freeze(mut self, context: &ModelContext) -> Result<ModelDefinition> {
        self.synchronize()?;
        if self.stream.device_id()
            != Device::open(context.runtime().gpu()?.index())?
                .stream()?
                .device_id()
        {
            return Err(Error::Message("model belongs to another device".into()));
        }
        self.graphs.clear();
        let allocations = self
            .allocations
            .drain(..)
            .map(|allocation| {
                if allocation.immutable {
                    // The session exclusively owns the allocation. Upload has been
                    // drained above; all old native graphs have been destroyed.
                    let buffer = unsafe {
                        context
                            .runtime()
                            .adopt_gpu_buffer(allocation.buffer, MemoryPlacement::GpuLocal)?
                    };
                    Ok(Storage::Weight(buffer))
                } else {
                    Ok(Storage::Scratch(
                        allocation.buffer.bytes(),
                        allocation.placement,
                    ))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ModelDefinition {
            id: self.id,
            context: context.clone(),
            allocations,
            kernels: self.kernels,
        })
    }
}

impl ModelDefinition {
    /// Shared scheduling and allocation domain.
    pub fn context(&self) -> &ModelContext {
        &self.context
    }

    /// Physical immutable weight bytes, counted once regardless of plan or slot count.
    pub fn weight_bytes(&self) -> usize {
        self.allocations
            .iter()
            .map(|storage| match storage {
                Storage::Weight(buffer) => buffer.len(),
                Storage::Scratch(..) => 0,
            })
            .sum()
    }

    fn validate_region(&self, region: Region) -> Result<()> {
        if region.session != self.id {
            return Err(Error::Message(
                "model region belongs to another definition".into(),
            ));
        }
        let bytes = match self.allocations.get(region.allocation) {
            Some(Storage::Weight(buffer)) => buffer.len(),
            Some(Storage::Scratch(bytes, _)) => *bytes,
            None => return Err(Error::Message("invalid model allocation".into())),
        };
        if region
            .offset
            .checked_add(region.bytes)
            .is_none_or(|end| end > bytes)
        {
            return Err(Error::Message("model region exceeds allocation".into()));
        }
        Ok(())
    }

    /// Prepare a shape with shared weights and fixed-address private slots.
    /// Input/output descriptors select leading bytes of their corresponding
    /// regions. Commands must restrict accesses to the selected shape.
    ///
    /// # Safety
    /// Dispatch bindings and scalar arguments must exactly describe the trusted
    /// compiled code, including accessible extents and every read or write.
    /// Writable argument aliasing must be safe for that kernel's execution.
    pub unsafe fn prepare(
        &self,
        commands: &[Command],
        inputs: &[(Region, TensorDesc)],
        outputs: &[(Region, TensorDesc)],
        capacity: usize,
    ) -> Result<PreparedModel> {
        // Safety: forwarded unchanged from the caller.
        unsafe { self.fragment(commands, inputs, outputs)? }.prepare(capacity)
    }

    /// Validate a shape's commands once, without allocating inference slots.
    /// The fragment owns shared weights and code independently of this definition.
    ///
    /// # Safety
    /// Dispatch contracts, accessible extents and writable aliases must satisfy
    /// the same requirements as [`Self::prepare`].
    pub unsafe fn fragment(
        &self,
        commands: &[Command],
        inputs: &[(Region, TensorDesc)],
        outputs: &[(Region, TensorDesc)],
    ) -> Result<ModelFragment> {
        for (region, desc) in inputs.iter().chain(outputs) {
            self.validate_region(*region)?;
            if desc.is_empty() || desc.bytes() > region.bytes {
                return Err(Error::Message(
                    "model IO descriptor exceeds its region or is empty".into(),
                ));
            }
            if matches!(self.allocations[region.allocation], Storage::Weight(_)) {
                return Err(Error::Message(
                    "model IO cannot alias immutable weights".into(),
                ));
            }
        }
        let mut prepared = Vec::with_capacity(commands.len());
        for command in commands {
            match command {
                Command::Fill { region, value } => {
                    self.validate_region(*region)?;
                    if matches!(self.allocations[region.allocation], Storage::Weight(_)) {
                        return Err(Error::Message("cannot fill immutable weights".into()));
                    }
                    prepared.push((None, vec![*region], Some(*value)));
                }
                Command::Dispatch(dispatch) => {
                    if dispatch.kernel.session != self.id {
                        return Err(Error::Message(
                            "kernel belongs to another definition".into(),
                        ));
                    }
                    let kernel = self
                        .kernels
                        .get(dispatch.kernel.index)
                        .ok_or_else(|| Error::Message("invalid model kernel".into()))?;
                    let mut bindings = Vec::with_capacity(dispatch.bindings.len());
                    for binding in &dispatch.bindings {
                        self.validate_region(binding.region)?;
                        if binding.access != Access::Read
                            && matches!(
                                self.allocations[binding.region.allocation],
                                Storage::Weight(_)
                            )
                        {
                            return Err(Error::Message("cannot write immutable weights".into()));
                        }
                        bindings.push(BindingContract {
                            bytes: binding.region.bytes,
                            alignment: 1,
                            access: binding.access,
                            layout: "model region".into(),
                        });
                    }
                    let constants = match &dispatch.arguments {
                        Arguments::Indices(values) => Constants::indices(kernel, values)?,
                        Arguments::Packed(constants) => *constants.clone(),
                    };
                    // Safety: inherited from this method's caller.
                    let kernel = unsafe {
                        self.context.runtime().adopt_gpu_kernel(
                            kernel.clone(),
                            dispatch.grid,
                            kernel.info().workgroup_size,
                            KernelContract {
                                bindings,
                                constants: constants.as_bytes().to_vec(),
                            },
                        )?
                    };
                    prepared.push((
                        Some(kernel),
                        dispatch.bindings.iter().map(|b| b.region).collect(),
                        None,
                    ));
                }
            }
        }
        let mut extents = vec![0; self.allocations.len()];
        for (_, regions, _) in &prepared {
            for region in regions {
                extents[region.allocation] =
                    extents[region.allocation].max(region.offset + region.bytes);
            }
        }
        for (region, desc) in inputs.iter().chain(outputs) {
            extents[region.allocation] =
                extents[region.allocation].max(region.offset + desc.bytes());
        }
        Ok(ModelFragment {
            context: self.context.clone(),
            allocations: self.allocations.clone(),
            extents,
            commands: prepared,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            reuse_private_scratch: false,
        })
    }
}

impl ModelFragment {
    /// Reuse private temporaries across fragments recorded into the same graph.
    /// Input/output allocations and immutable weights are never recycled. Each
    /// inference slot still owns a separate workspace. Graph hazards order reads
    /// before a later fragment overwrites that storage, potentially serializing
    /// otherwise independent branches in exchange for lower peak memory.
    ///
    /// # Safety
    /// Every private scratch byte read by this fragment must first be written
    /// by this fragment on every execution, including padding and conditional
    /// paths. Neither initial zeroes nor another fragment's values may be used.
    /// All scratch accesses must already be declared by the kernel contracts.
    pub unsafe fn reuse_private_scratch(mut self) -> Self {
        self.reuse_private_scratch = true;
        self
    }
    /// Allocate independent fixed-address slots, exposing their actual model IO.
    pub fn prepare(&self, capacity: usize) -> Result<PreparedModel> {
        PreparedModel::prepare(&self.context, capacity, |context| {
            let mut graph = context.runtime().graph();
            let (inputs, outputs) = self.append(&mut graph, None)?;
            Ok(InferenceGraph {
                inputs,
                outputs,
                graph: graph.prepare()?,
            })
        })
    }

    /// Append model commands directly to an unprepared graph. Inputs are bound
    /// without a device copy; outputs and scratch are allocated for this graph.
    /// Adjacent stages can consume the returned tensors before graph preparation.
    /// No pool, host staging, submission or intermediate synchronization is added.
    ///
    /// Inputs must describe the complete accessed regions of their allocations.
    /// Partial aliases which would require a copy are rejected. Initialization
    /// producers are drained here; this is a preparation-time operation.
    pub fn record(&self, graph: &mut Graph, inputs: &[DeviceTensor]) -> Result<Vec<DeviceTensor>> {
        Ok(self.append(graph, Some(inputs))?.1)
    }

    fn append(
        &self,
        graph: &mut Graph,
        supplied: Option<&[DeviceTensor]>,
    ) -> Result<(Vec<DeviceTensor>, Vec<DeviceTensor>)> {
        graph.validate_runtime(self.context.runtime())?;
        let mut buffers: Vec<Option<(usize, BufferView)>> = vec![None; self.allocations.len()];
        if let Some(inputs) = supplied {
            if inputs.len() != self.inputs.len() {
                return Err(Error::Message("fragment input count mismatch".into()));
            }
            for ((region, desc), tensor) in self.inputs.iter().zip(inputs) {
                self.context.validate(tensor)?;
                if tensor.desc() != desc {
                    return Err(Error::Message("fragment input descriptor mismatch".into()));
                }
                tensor.completion().wait()?;
                let binding = tensor.binding().expect("nonempty fragment IO");
                if buffers.iter().enumerate().any(|(index, old)| {
                    index != region.allocation && old.as_ref().is_some_and(|(_, old)| old.overlaps(&binding))
                }) || self.allocations.iter().any(|storage| {
                    matches!(storage, Storage::Weight(weight) if weight.view().overlaps(&binding))
                }) {
                    return Err(Error::Message("fragment remapping introduces an undeclared alias".into()));
                }
                if let Some((base, old)) = &buffers[region.allocation] {
                    if *base != region.offset || !old.same_region(&binding) {
                        return Err(Error::Message(
                            "fragment inputs alias one allocation inconsistently".into(),
                        ));
                    }
                } else {
                    buffers[region.allocation] = Some((region.offset, binding));
                }
            }
        }
        if self.reuse_private_scratch {
            let private: Vec<_> = self
                .allocations
                .iter()
                .enumerate()
                .filter_map(|(index, storage)| match storage {
                    Storage::Scratch(_, placement)
                        if self.extents[index] > 0
                            && !self
                                .inputs
                                .iter()
                                .chain(&self.outputs)
                                .any(|(region, _)| region.allocation == index) =>
                    {
                        Some((
                            index,
                            (self.extents[index], matches!(placement, Placement::Shared)),
                        ))
                    }
                    _ => None,
                })
                .collect();
            let requests: Vec<_> = private.iter().map(|(_, request)| *request).collect();
            for ((index, _), view) in private.into_iter().zip(graph.fragment_scratch(&requests)?) {
                buffers[index] = Some((0, view));
            }
        }
        for (index, storage) in self.allocations.iter().enumerate() {
            if buffers[index].is_none() {
                buffers[index] = match storage {
                    Storage::Weight(buffer) => Some((0, buffer.view())),
                    Storage::Scratch(..) if self.extents[index] == 0 => None,
                    Storage::Scratch(_, placement) => Some((
                        0,
                        self.context
                            .runtime()
                            .allocate(
                                self.extents[index],
                                match placement {
                                    Placement::Device => MemoryPlacement::GpuLocal,
                                    Placement::Shared => MemoryPlacement::HostVisible,
                                },
                            )?
                            .view(),
                    )),
                };
            }
        }
        let view = |region: Region| -> Result<BufferView> {
            let (base, buffer) = buffers[region.allocation]
                .as_ref()
                .expect("referenced allocation");
            let start = region
                .offset
                .checked_sub(*base)
                .ok_or_else(|| Error::Message("fragment access precedes bound input".into()))?;
            buffer.slice(start..start + region.bytes)
        };
        let tensors = |io: &[(Region, TensorDesc)]| {
            io.iter()
                .map(|(region, desc)| {
                    self.context.tensor(
                        desc.clone(),
                        view(region.slice(0, desc.bytes())?)?,
                        crate::Completion::ready(),
                    )
                })
                .collect::<Result<Vec<_>>>()
        };
        let inputs = tensors(&self.inputs)?;
        let outputs = tensors(&self.outputs)?;
        // Validate every binding before mutating the caller's graph.
        let bindings = self
            .commands
            .iter()
            .map(|(_, regions, _)| {
                regions
                    .iter()
                    .map(|&region| view(region))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        for ((kernel, _, fill), bindings) in self.commands.iter().zip(bindings) {
            if let Some(value) = fill {
                graph.fill(bindings[0].clone(), *value)?;
            } else if let Some(kernel) = kernel {
                // Safety: fragment construction validated the caller's trusted
                // contracts. Remapping preserves region offsets and extents.
                unsafe {
                    graph.gpu_aliasing(kernel, &bindings)?;
                }
            }
        }
        Ok((inputs, outputs))
    }
}
