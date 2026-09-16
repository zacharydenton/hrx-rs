use super::*;
use crate::{
    execution::{BindingContract, KernelContract, MemoryPlacement},
    inference::{ModelContext, PreparedModel},
    tensor::TensorDesc,
};

enum Storage {
    Weight(crate::execution::Buffer),
    Scratch(usize),
}

/// Immutable model code and weights in a shared context. Shape-specific plans
/// allocate private scratch and graph storage for each bounded inference slot.
pub struct ModelDefinition {
    id: u64,
    context: ModelContext,
    allocations: Vec<Storage>,
    kernels: Vec<Kernel>,
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
                    Ok(Storage::Scratch(allocation.buffer.bytes()))
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
                Storage::Scratch(_) => 0,
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
            Some(Storage::Scratch(bytes)) => *bytes,
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
        let input_descs = inputs
            .iter()
            .map(|(_, desc)| desc.clone())
            .collect::<Vec<_>>();
        let output_descs = outputs
            .iter()
            .map(|(_, desc)| desc.clone())
            .collect::<Vec<_>>();
        PreparedModel::prepare(
            &self.context,
            &input_descs,
            &output_descs,
            capacity,
            |context, input_tensors, output_tensors| {
                let buffers = self
                    .allocations
                    .iter()
                    .enumerate()
                    .map(|(index, storage)| match storage {
                        Storage::Weight(buffer) => Ok(Some(buffer.clone())),
                        Storage::Scratch(_) if extents[index] == 0 => Ok(None),
                        Storage::Scratch(_) => context
                            .runtime()
                            .allocate(extents[index], MemoryPlacement::GpuLocal)
                            .map(Some),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let view = |region: Region| {
                    buffers[region.allocation]
                        .as_ref()
                        .expect("referenced allocation")
                        .slice(region.offset..region.offset + region.bytes)
                };
                let mut graph = context.runtime().graph();
                for ((region, desc), tensor) in inputs.iter().zip(input_tensors) {
                    graph.copy(
                        view(region.slice(0, desc.bytes())?)?,
                        tensor.binding().expect("nonempty IO"),
                    )?;
                }
                for (kernel, regions, fill) in &prepared {
                    if let Some(value) = fill {
                        graph.fill(view(regions[0])?, *value)?;
                    } else if let Some(kernel) = kernel {
                        let bindings = regions
                            .iter()
                            .map(|&region| view(region))
                            .collect::<Result<Vec<_>>>()?;
                        // Safety: this preparation accepts the same aliasing contract
                        // as the model's trusted dispatch declarations.
                        unsafe {
                            graph.gpu_aliasing(kernel, &bindings)?;
                        }
                    }
                }
                for ((region, desc), tensor) in outputs.iter().zip(output_tensors) {
                    graph.copy(
                        tensor.binding().expect("nonempty IO"),
                        view(region.slice(0, desc.bytes())?)?,
                    )?;
                }
                graph.prepare()
            },
        )
    }
}
