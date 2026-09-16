//! Shared model context, owned tensor transfers, and bounded inference slots.

use crate::{
    Completion, Error, Result, Runtime,
    execution::{Buffer, ExecutableGraph, MemoryPlacement, RuntimeOptions},
    loom::Compiler,
    tensor::{DeviceTensor, TensorDesc},
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
};

/// Explicitly shared allocation, compilation, and scheduling domain for models.
#[derive(Clone)]
pub struct ModelContext {
    runtime: Runtime,
    compiler: Arc<Mutex<Option<Compiler>>>,
}
impl ModelContext {
    /// Construct a context; hardware and compiler initialization remain lazy.
    pub fn new(options: RuntimeOptions) -> Result<Self> {
        Ok(Self {
            runtime: Runtime::with_options(options)?,
            compiler: Arc::new(Mutex::new(None)),
        })
    }
    /// Coordinated runtime shared by every model and tensor in this context.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }
    /// Lazily obtain the compiler for the selected device's actual target.
    pub fn compiler(&self) -> Result<Compiler> {
        crate::cached_init(&self.compiler, || {
            Compiler::for_target(None, self.runtime.gpu()?.target())
        })
    }
    /// Allocate initialized device storage. Empty tensors allocate nothing.
    pub fn allocate(&self, desc: TensorDesc) -> Result<DeviceTensor> {
        let view = if desc.is_empty() {
            None
        } else {
            Some(
                self.runtime
                    .allocate(desc.bytes(), MemoryPlacement::GpuLocal)?
                    .view(),
            )
        };
        Ok(DeviceTensor {
            runtime: self.runtime.clone(),
            desc,
            view,
            producer: Completion::ready(),
            lease: None,
        })
    }
    /// Wrap existing tracked storage without copying. The producer must belong
    /// to this runtime or represent already completed initialization.
    pub fn tensor(
        &self,
        desc: TensorDesc,
        view: crate::execution::BufferView,
        producer: Completion,
    ) -> Result<DeviceTensor> {
        producer.validate_runtime(&self.runtime)?;
        if !self.runtime.owns(&view)
            || view.len() < desc.bytes()
            || !view.offset().is_multiple_of(desc.dtype().bytes())
        {
            return Err(Error::Message(
                "tensor storage has the wrong context, extent, or alignment".into(),
            ));
        }
        let view = if desc.is_empty() {
            None
        } else {
            Some(view.slice(0..desc.bytes())?)
        };
        Ok(DeviceTensor {
            runtime: self.runtime.clone(),
            desc,
            view,
            producer,
            lease: None,
        })
    }
    /// Copy host bytes into owned staging and enqueue upload. The caller may
    /// release or mutate its source as soon as this method returns.
    pub fn upload(&self, desc: TensorDesc, bytes: &[u8]) -> Result<DeviceTensor> {
        if bytes.len() != desc.bytes() {
            return Err(Error::Message(
                "upload byte count does not match tensor".into(),
            ));
        }
        let mut tensor = self.allocate(desc)?;
        if let Some(destination) = tensor.binding() {
            let staging = self
                .runtime
                .allocate(bytes.len(), MemoryPlacement::HostVisible)?;
            staging.map_write()?.copy_from_slice(bytes);
            let mut graph = self.runtime.graph();
            graph.copy(destination, staging.view())?;
            tensor.producer = graph.prepare()?.submit()?;
        }
        Ok(tensor)
    }
    /// Copy a contiguous tensor into a byte-aligned subrange of another tensor.
    /// Both tensors must be contiguous and have the same dtype. This updates the
    /// destination's producer dependency, preserving failures from earlier writes.
    /// Slot owners survive until the copy drains, even if the source is dropped.
    pub fn copy_into(
        &self,
        destination: &mut DeviceTensor,
        byte_offset: usize,
        source: &DeviceTensor,
    ) -> Result<()> {
        self.validate(destination)?;
        self.validate(source)?;
        let end = byte_offset
            .checked_add(source.desc.bytes())
            .filter(|&v| v <= destination.desc.bytes())
            .ok_or_else(|| Error::Message("tensor copy exceeds destination".into()))?;
        if destination.desc.dtype() != source.desc.dtype()
            || !destination.desc.is_contiguous()
            || !source.desc.is_contiguous()
            || !byte_offset.is_multiple_of(source.desc.dtype().bytes())
        {
            return Err(Error::Message(
                "invalid tensor copy dtype, strides or alignment".into(),
            ));
        }
        if let Some(from) = source.binding() {
            let to = destination
                .binding()
                .expect("checked nonempty destination")
                .slice(byte_offset..end)?;
            let mut graph = self.runtime.graph();
            graph.copy(to, from)?;
            destination.producer = graph
                .prepare()?
                .submit_after(&[destination.producer.clone(), source.producer.clone()])?;
        } else {
            source.producer.wait()?;
        }
        Ok(())
    }

    /// Concatenate contiguous tensors along their first axis on the device.
    /// Inputs must have identical trailing dimensions, dtype and layout. Empty
    /// inputs are allowed; scalars and an empty input list are not.
    pub fn concatenate(&self, inputs: &[DeviceTensor]) -> Result<DeviceTensor> {
        let first = inputs
            .first()
            .ok_or_else(|| Error::Message("concatenate needs inputs".into()))?;
        let prototype = first.desc();
        if prototype.shape().is_empty() || !prototype.is_contiguous() {
            return Err(Error::Message(
                "concatenate requires contiguous non-scalar tensors".into(),
            ));
        }
        let mut shape = prototype.shape().to_vec();
        shape[0] = 0;
        for input in inputs {
            self.validate(input)?;
            let desc = input.desc();
            if desc.dtype() != prototype.dtype()
                || desc.layout() != prototype.layout()
                || desc.shape().len() != prototype.shape().len()
                || desc.shape()[1..] != prototype.shape()[1..]
                || !desc.is_contiguous()
            {
                return Err(Error::Message("incompatible concatenate input".into()));
            }
            shape[0] = shape[0]
                .checked_add(desc.shape()[0])
                .ok_or_else(|| Error::Message("concatenate dimension overflow".into()))?;
        }
        let mut output = self.allocate(
            TensorDesc::new(prototype.dtype(), shape)?.with_layout(prototype.layout())?,
        )?;
        // Even zero-byte inputs may carry failed producers: do not silently
        // discard their dependencies by treating concatenation as a no-op.
        let dependencies: Vec<_> = inputs.iter().map(|t| t.producer.clone()).collect();
        if let Some(destination) = output.binding() {
            let mut graph = self.runtime.graph();
            let mut offset = 0;
            for input in inputs {
                if let Some(source) = input.binding() {
                    graph.copy(destination.slice(offset..offset + source.len())?, source)?;
                    offset += input.desc().bytes();
                }
            }
            output.producer = graph.prepare()?.submit_after(&dependencies)?;
        } else {
            // No native operation is needed. A caller cannot observe a successful
            // empty tensor until every producer has completed successfully.
            for dependency in dependencies {
                dependency.wait()?;
            }
        }
        Ok(output)
    }

    /// Queue readback after a tensor's producer, without a host wait.
    pub fn download(&self, tensor: &DeviceTensor) -> Result<TensorReadback> {
        self.validate(tensor)?;
        let Some(source) = tensor.binding() else {
            return Ok(TensorReadback {
                buffer: None,
                completion: tensor.producer.clone(),
            });
        };
        let staging = self
            .runtime
            .allocate(tensor.desc.bytes(), MemoryPlacement::HostVisible)?;
        let mut graph = self.runtime.graph();
        graph.copy(staging.view(), source)?;
        let completion = graph
            .prepare()?
            .submit_after(std::slice::from_ref(&tensor.producer))?;
        Ok(TensorReadback {
            buffer: Some(staging),
            completion,
        })
    }
    /// Reject a tensor from another allocation domain before any work is queued.
    pub fn validate(&self, tensor: &DeviceTensor) -> Result<()> {
        if !self.runtime.same_domain(&tensor.runtime) {
            return Err(Error::Message(
                "tensor belongs to another model context".into(),
            ));
        }
        Ok(())
    }
}

/// Owned asynchronous readback. Dropping it never invalidates queued work.
pub struct TensorReadback {
    buffer: Option<Buffer>,
    completion: Completion,
}
impl TensorReadback {
    /// Readback completion, suitable for polling or awaiting.
    pub fn completion(&self) -> &Completion {
        &self.completion
    }
    /// Wait and return the tensor's storage bytes, including declared stride gaps.
    pub fn wait(self) -> Result<Vec<u8>> {
        self.completion.wait()?;
        self.bytes()
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        self.buffer
            .as_ref()
            .map_or_else(|| Ok(Vec::new()), |buffer| Ok(buffer.map_read()?.to_vec()))
    }
}
impl Future for TensorReadback {
    type Output = Result<Vec<u8>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.completion).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => Poll::Ready(self.bytes()),
        }
    }
}

struct Slot {
    inputs: Vec<DeviceTensor>,
    outputs: Vec<DeviceTensor>,
    upload: Mutex<Option<Arc<Transfer>>>,
    download: Mutex<Option<Arc<Transfer>>>,
    graph: ExecutableGraph,
}

struct Transfer {
    buffers: Vec<Option<Buffer>>,
    graph: Option<ExecutableGraph>,
}
impl Transfer {
    fn prepare(context: &ModelContext, tensors: &[DeviceTensor], upload: bool) -> Result<Self> {
        let mut graph = context.runtime.graph();
        let buffers = tensors
            .iter()
            .map(|tensor| {
                let Some(device) = tensor.binding() else {
                    return Ok(None);
                };
                let host = context
                    .runtime
                    .allocate(tensor.desc.bytes(), MemoryPlacement::HostVisible)?;
                if upload {
                    graph.copy(device, host.view())?;
                } else {
                    graph.copy(host.view(), device)?;
                }
                Ok(Some(host))
            })
            .collect::<Result<Vec<_>>>()?;
        let graph = if buffers.iter().any(Option::is_some) {
            Some(graph.prepare()?)
        } else {
            None
        };
        Ok(Self { buffers, graph })
    }
}

/// One executable slot and its actual tensor bindings. Bindings may be slices
/// of shared scratch or alias each other for in-place execution. No extra model
/// input/output buffers are introduced by the inference pool.
pub struct InferenceGraph {
    /// Fixed-address inputs populated before execution.
    pub inputs: Vec<DeviceTensor>,
    /// Fixed-address outputs leased to the caller after submission.
    pub outputs: Vec<DeviceTensor>,
    /// The graph which reads inputs and produces outputs.
    pub graph: ExecutableGraph,
}
#[derive(Default)]
struct SlotState {
    leased: bool,
    completion: Option<Completion>,
}
struct PoolState {
    slots: Vec<SlotState>,
    wakers: Vec<Waker>,
}
struct Pool {
    context: ModelContext,
    slots: Vec<Slot>,
    state: Mutex<PoolState>,
    changed: Condvar,
}
struct Lease {
    pool: Arc<Pool>,
    index: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let wakers = {
            let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
            state.slots[self.index].leased = false;
            std::mem::take(&mut state.wakers)
        };
        self.pool.changed.notify_all();
        for waker in wakers {
            waker.wake();
        }
    }
}

/// A prepared model with shared immutable resources and fixed-address private
/// input, output, scratch, and graph storage for each bounded inference slot.
#[derive(Clone)]
pub struct PreparedModel {
    pool: Arc<Pool>,
}
impl PreparedModel {
    /// Build each slot once, including its actual input/output bindings. The
    /// builder may share immutable weights, but must allocate independent
    /// writable storage for each slot. Host staging is created on first use.
    pub fn prepare(
        context: &ModelContext,
        capacity: usize,
        mut build: impl FnMut(&ModelContext) -> Result<InferenceGraph>,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Message("inference capacity must be nonzero".into()));
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            let InferenceGraph {
                inputs,
                outputs,
                graph,
            } = build(context)?;
            graph.validate_runtime(&context.runtime)?;
            for tensor in inputs.iter().chain(&outputs) {
                context.validate(tensor)?;
                tensor.producer.wait()?;
                if let Some(view) = tensor.binding()
                    && slots
                        .iter()
                        .flat_map(|slot| slot.inputs.iter().chain(&slot.outputs))
                        .filter_map(DeviceTensor::binding)
                        .any(|old| view.overlaps(&old))
                {
                    return Err(Error::Message("inference slots share writable IO".into()));
                }
            }
            if let Some(first) = slots.first()
                && (inputs
                    .iter()
                    .map(DeviceTensor::desc)
                    .ne(first.inputs.iter().map(DeviceTensor::desc))
                    || outputs
                        .iter()
                        .map(DeviceTensor::desc)
                        .ne(first.outputs.iter().map(DeviceTensor::desc)))
            {
                return Err(Error::Message("inference slot descriptors differ".into()));
            }
            slots.push(Slot {
                inputs,
                outputs,
                upload: Mutex::new(None),
                download: Mutex::new(None),
                graph,
            });
        }
        Ok(Self {
            pool: Arc::new(Pool {
                context: context.clone(),
                slots,
                state: Mutex::new(PoolState {
                    slots: (0..capacity).map(|_| SlotState::default()).collect(),
                    wakers: Vec::new(),
                }),
                changed: Condvar::new(),
            }),
        })
    }
    /// Number of resident inference slots.
    pub fn capacity(&self) -> usize {
        self.pool.slots.len()
    }
    /// Whether every slot is unleased and drained, suitable for idle cache eviction.
    pub fn is_idle(&self) -> bool {
        self.pool
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .slots
            .iter()
            .all(|slot| {
                !slot.leased && slot.completion.as_ref().is_none_or(Completion::is_complete)
            })
    }
    /// Reserve an idle slot without waiting. Retained outputs apply backpressure.
    pub fn try_acquire(&self) -> Result<ModelSlot> {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        self.acquire_locked(&mut state)
    }
    fn acquire_locked(&self, state: &mut PoolState) -> Result<ModelSlot> {
        let mut failure = None;
        for (index, slot) in state.slots.iter_mut().enumerate() {
            if slot.leased {
                continue;
            }
            if let Some(completion) = &slot.completion
                && !completion.is_complete()
            {
                continue;
            }
            // Cancellation and failed producers do not poison untouched storage.
            // Native failures do: leave that slot disabled, but try healthy peers.
            if let Err(error) = self.pool.slots[index].graph.usable() {
                failure = Some(error);
                continue;
            }
            slot.completion = None;
            slot.leased = true;
            return Ok(ModelSlot {
                lease: Arc::new(Lease {
                    pool: self.pool.clone(),
                    index,
                }),
            });
        }
        if state.slots.iter().all(|slot| {
            !slot.leased && slot.completion.as_ref().is_none_or(Completion::is_complete)
        }) && let Some(error) = failure
        {
            return Err(error);
        }
        Err(Error::Busy(
            "all inference slots are retained or running".into(),
        ))
    }
    /// Wait for capacity without growing the queue or duplicating weights.
    pub fn acquire_blocking(&self) -> Result<ModelSlot> {
        loop {
            let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
            match self.acquire_locked(&mut state) {
                Ok(slot) => return Ok(slot),
                Err(Error::Busy(_)) => {}
                Err(error) => return Err(error),
            }
            if let Some(completion) = state
                .slots
                .iter()
                .filter(|slot| !slot.leased)
                .filter_map(|slot| slot.completion.clone())
                .find(|completion| !completion.is_complete())
            {
                drop(state);
                // Recheck storage after draining, including safe cancellation.
                let _ = completion.wait();
            } else {
                drop(
                    self.pool
                        .changed
                        .wait(state)
                        .unwrap_or_else(|e| e.into_inner()),
                );
            }
        }
    }
    /// Wait asynchronously for a slot; polling never waits for a device.
    pub fn acquire(&self) -> AcquireSlot<'_> {
        AcquireSlot { model: self }
    }
    /// Reserve a slot and submit immediately, or return `Busy`.
    pub fn submit(&self, inputs: &[DeviceTensor]) -> Result<Inference> {
        self.try_acquire()?.submit(inputs)
    }
    /// Submit host bytes through preallocated upload storage, without a warm
    /// path allocation or graph preparation.
    pub fn submit_host(&self, inputs: &[&[u8]]) -> Result<Inference> {
        self.try_acquire()?.submit_host(inputs)
    }
}

/// Future yielding an exclusively reserved inference slot.
pub struct AcquireSlot<'a> {
    model: &'a PreparedModel,
}
impl Future for AcquireSlot<'_> {
    type Output = Result<ModelSlot>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .model
            .pool
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match self.model.acquire_locked(&mut state) {
            Ok(slot) => return Poll::Ready(Ok(slot)),
            Err(Error::Busy(_)) => {}
            Err(error) => return Poll::Ready(Err(error)),
        }
        let mut ready = false;
        for slot in state.slots.iter_mut().filter(|slot| !slot.leased) {
            if let Some(completion) = &mut slot.completion {
                ready |= Pin::new(completion).poll(cx).is_ready();
            }
        }
        if ready {
            cx.waker().wake_by_ref();
        }
        if !state.wakers.iter().any(|old| old.will_wake(cx.waker())) {
            state.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

/// An exclusively reserved slot. Dropping without submission releases capacity.
pub struct ModelSlot {
    lease: Arc<Lease>,
}
impl ModelSlot {
    /// Stage host inputs into this reserved slot and enqueue inference.
    pub fn submit_host(self, inputs: &[&[u8]]) -> Result<Inference> {
        let pool = &self.lease.pool;
        let slot = &pool.slots[self.lease.index];
        if inputs.len() != slot.inputs.len()
            || inputs
                .iter()
                .zip(&slot.inputs)
                .any(|(bytes, tensor)| bytes.len() != tensor.desc.bytes())
        {
            return Err(Error::Message(
                "host model input byte count mismatch".into(),
            ));
        }
        let upload = crate::cached_init(&slot.upload, || {
            Ok(Arc::new(Transfer::prepare(
                &pool.context,
                &slot.inputs,
                true,
            )?))
        })?;
        for (bytes, host) in inputs.iter().zip(&upload.buffers) {
            if let Some(host) = host {
                host.map_write()?.copy_from_slice(bytes);
            }
        }
        let dependencies = if let Some(upload) = &upload.graph {
            let completion = upload.submit()?;
            pool.state.lock().unwrap_or_else(|e| e.into_inner()).slots[self.lease.index]
                .completion = Some(completion.clone());
            vec![completion]
        } else {
            vec![]
        };
        self.execute(&dependencies)
    }
    /// Copy device inputs into fixed slot addresses and enqueue the model.
    /// Input producers are ordered and failures propagate without host waits.
    pub fn submit(self, inputs: &[DeviceTensor]) -> Result<Inference> {
        let pool = &self.lease.pool;
        let slot = &pool.slots[self.lease.index];
        if inputs.len() != slot.inputs.len() {
            return Err(Error::Message("model input count mismatch".into()));
        }
        for (input, expected) in inputs.iter().zip(&slot.inputs) {
            pool.context.validate(input)?;
            if input.desc != expected.desc {
                return Err(Error::Message("model input descriptor mismatch".into()));
            }
        }
        let producers = inputs
            .iter()
            .map(|input| input.producer.clone())
            .collect::<Vec<_>>();
        let mut copy = pool.context.runtime.graph();
        let mut copied = false;
        for (source, destination) in inputs.iter().zip(&slot.inputs) {
            if let (Some(source), Some(destination)) = (source.binding(), destination.binding()) {
                copy.copy(destination, source)?;
                copied = true;
            }
        }
        let dependencies = if copied {
            let completion = copy.prepare()?.submit_after(&producers)?;
            pool.state.lock().unwrap_or_else(|e| e.into_inner()).slots[self.lease.index]
                .completion = Some(completion.clone());
            vec![completion]
        } else {
            producers
        };
        self.execute(&dependencies)
    }
    fn execute(self, dependencies: &[Completion]) -> Result<Inference> {
        let pool = &self.lease.pool;
        let slot = &pool.slots[self.lease.index];
        let completion = slot.graph.submit_after(dependencies)?;
        pool.state.lock().unwrap_or_else(|e| e.into_inner()).slots[self.lease.index].completion =
            Some(completion.clone());
        let outputs = slot
            .outputs
            .iter()
            .cloned()
            .map(|mut output| {
                output.producer = completion.clone();
                output.lease = Some(self.lease.clone());
                output
            })
            .collect();
        Ok(Inference {
            outputs,
            completion,
            _lease: self.lease,
        })
    }
}

/// Asynchronous inference outputs and their completion. Dropping the handle
/// does not cancel execution; exported outputs keep the slot reserved.
pub struct Inference {
    outputs: Vec<DeviceTensor>,
    completion: Completion,
    _lease: Arc<Lease>,
}
impl Inference {
    /// Queue all outputs into reusable host storage, allocated on first readback.
    pub fn download(self) -> Result<InferenceReadback> {
        let pool = &self._lease.pool;
        let slot = &pool.slots[self._lease.index];
        let transfer = crate::cached_init(&slot.download, || {
            Ok(Arc::new(Transfer::prepare(
                &pool.context,
                &slot.outputs,
                false,
            )?))
        })?;
        let completion = if let Some(download) = &transfer.graph {
            download.submit_after(std::slice::from_ref(&self.completion))?
        } else {
            self.completion.clone()
        };
        pool.state.lock().unwrap_or_else(|e| e.into_inner()).slots[self._lease.index].completion =
            Some(completion.clone());
        Ok(InferenceReadback {
            completion,
            lease: self._lease,
            transfer,
        })
    }
    /// Device outputs, available for dependent submission before completion.
    pub fn outputs(&self) -> &[DeviceTensor] {
        &self.outputs
    }
    /// Completion observer; cancellation drains submitted work safely.
    pub fn completion(&self) -> &Completion {
        &self.completion
    }
    /// Wait for successful execution and return owned device outputs.
    pub fn wait(self) -> Result<Vec<DeviceTensor>> {
        self.completion.wait()?;
        Ok(self.outputs)
    }
}

/// Readbacks sharing their inference slot's preallocated host storage.
pub struct InferenceReadback {
    completion: Completion,
    lease: Arc<Lease>,
    transfer: Arc<Transfer>,
}
impl InferenceReadback {
    /// Completion of the final device-to-host copy.
    pub fn completion(&self) -> &Completion {
        &self.completion
    }
    /// Wait and copy outputs into owned host byte vectors.
    pub fn wait(self) -> Result<Vec<Vec<u8>>> {
        self.completion.wait()?;
        self.bytes()
    }
    /// Wait and copy into caller-owned output buffers without allocating.
    pub fn read_into(self, outputs: &mut [&mut [u8]]) -> Result<()> {
        let slot = &self.lease.pool.slots[self.lease.index];
        if outputs.len() != slot.outputs.len()
            || outputs
                .iter()
                .zip(&slot.outputs)
                .any(|(bytes, tensor)| bytes.len() != tensor.desc.bytes())
        {
            return Err(Error::Message("model output byte count mismatch".into()));
        }
        self.completion.wait()?;
        for (out, host) in outputs.iter_mut().zip(&self.transfer.buffers) {
            if let Some(host) = host {
                out.copy_from_slice(&host.map_read()?);
            }
        }
        Ok(())
    }
    fn bytes(&self) -> Result<Vec<Vec<u8>>> {
        self.transfer
            .buffers
            .iter()
            .map(|host| {
                host.as_ref()
                    .map_or_else(|| Ok(Vec::new()), |host| Ok(host.map_read()?.to_vec()))
            })
            .collect()
    }
}
impl Future for InferenceReadback {
    type Output = Result<Vec<Vec<u8>>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.completion).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => Poll::Ready(self.bytes()),
        }
    }
}
impl Future for Inference {
    type Output = Result<Vec<DeviceTensor>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.completion).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => Poll::Ready(Ok(std::mem::take(&mut self.outputs))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;
    #[test]
    fn empty_tensors_need_no_device_and_keep_context_identity() {
        let a = ModelContext::new(RuntimeOptions::default()).unwrap();
        let b = ModelContext::new(RuntimeOptions::default()).unwrap();
        let tensor = a
            .upload(TensorDesc::new(DType::F32, vec![0]).unwrap(), &[])
            .unwrap();
        assert!(tensor.binding().is_none());
        assert!(a.download(&tensor).unwrap().wait().unwrap().is_empty());
        assert!(b.download(&tensor).is_err());
        assert!(
            tensor
                .view(0, TensorDesc::new(DType::F32, vec![]).unwrap())
                .is_err()
        );
    }
}
