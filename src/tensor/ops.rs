use super::{DType, DeviceTensor, TensorDesc};
use crate::{
    Error, Result,
    inference::{Inference, ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, ModelFragment, ModelSession},
    plan_cache::PlanCache,
};
use std::sync::Arc;

/// Bounded shape-specialized device tensor operations in one shared context.
pub struct TensorOps {
    context: ModelContext,
    finite: PlanCache<TensorDesc, PreparedModel>,
    gather: PlanCache<(TensorDesc, usize), PreparedModel>,
}
impl TensorOps {
    /// Recordable row gather with directly bound source and U32 `[rows]`
    /// indices. Output is host-visible so final records can be mapped without
    /// a copy; GPU consumers can bind it directly. Out-of-range indices select
    /// row zero, so callers must validate indices before publication.
    pub fn gather_rows_fragment(&self, input: &TensorDesc, rows: usize) -> Result<ModelFragment> {
        self.gather_fragment(input, rows, true)
    }
    fn gather_fragment(
        &self,
        input: &TensorDesc,
        rows: usize,
        mapped: bool,
    ) -> Result<ModelFragment> {
        if input.shape().len() != 2
            || !input.is_contiguous()
            || input.is_empty()
            || input.bytes() > i32::MAX as usize
            || rows == 0
        {
            return Err(Error::Message(
                "gather requires nonempty contiguous rows".into(),
            ));
        }
        let output = TensorDesc::new(input.dtype(), vec![rows, input.shape()[1]])?
            .with_layout(input.layout())?;
        if output.bytes() > i32::MAX as usize {
            return Err(Error::Message(
                "gather exceeds kernel indexing limits".into(),
            ));
        }
        let ids_desc = TensorDesc::new(DType::U32, vec![rows])?;
        if ids_desc.bytes() > i32::MAX as usize {
            return Err(Error::Message(
                "gather index storage exceeds kernel limits".into(),
            ));
        }
        let mut model = ModelSession::in_context(&self.context)?;
        let source = model.allocate(input.bytes())?;
        let ids = model.allocate(ids_desc.bytes())?;
        let destination = if mapped {
            model.allocate_shared(output.bytes())?
        } else {
            model.allocate(output.bytes())?
        };
        let mut code = include_str!("gather.loom").to_owned();
        for (name, value) in [
            ("INPUT", input.bytes()),
            ("LAST_INPUT", input.bytes() - 1),
            ("ROWS", input.shape()[0]),
            ("LAST_ROW", input.shape()[0] - 1),
            ("ROW_BYTES", input.shape()[1] * input.dtype().bytes()),
            ("BUCKET", rows),
            ("LAST_INDEX", rows - 1),
            ("OUTPUT", output.bytes()),
            ("LAST_OUTPUT", output.bytes() - 1),
            ("GRID", output.bytes().div_ceil(256)),
        ] {
            code = code.replace(&format!("@{name}@"), &value.to_string());
        }
        // The embedded kernel checks indices and bounds every byte access.
        let kernel = unsafe { model.compile(&[(&code, Specialization::new("gather"))])? }[0];
        unsafe {
            model.freeze(&self.context)?.fragment(
                &[Command::Dispatch(Dispatch::indices(
                    kernel,
                    [0],
                    [output.bytes().div_ceil(256) as u32, 1, 1],
                    vec![source.read(), ids.read(), destination.write()],
                ))],
                &[(source, input.clone()), (ids, ids_desc)],
                &[(destination, output)],
            )
        }
    }
    /// Create lazy bounded caches, evicting only unleased and drained plans.
    pub fn new(context: &ModelContext, capacity: usize) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            finite: PlanCache::new(capacity, PreparedModel::is_idle)?,
            gather: PlanCache::new(capacity, PreparedModel::is_idle)?,
        })
    }
    /// Gather complete contiguous rows using checked host indices. Duplicates
    /// and arbitrary ordering are allowed. Only indices are uploaded; row bytes
    /// stay on device. Three private slots per power-of-two output-size bucket
    /// bound preparation churn; the returned view retains its slot until dropped.
    pub fn gather_rows(&self, input: &DeviceTensor, indices: &[usize]) -> Result<DeviceTensor> {
        self.context.validate(input)?;
        let desc = input.desc();
        if desc.shape().len() != 2
            || !desc.is_contiguous()
            || desc.bytes() > i32::MAX as usize
            || indices.iter().any(|&i| i >= desc.shape()[0])
        {
            return Err(Error::Message(
                "gather requires contiguous rows and in-range indices".into(),
            ));
        }
        let output = TensorDesc::new(desc.dtype(), vec![indices.len(), desc.shape()[1]])?
            .with_layout(desc.layout())?;
        if output.is_empty() {
            input.completion().wait()?;
            return self.context.allocate(output);
        }
        let bucket = indices
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| Error::Message("gather size overflow".into()))?;
        let plan = self.gather.get_or_prepare((desc.clone(), bucket), || {
            self.gather_fragment(desc, bucket, false)?.prepare(3)
        })?;
        let bytes: Vec<_> = indices
            .iter()
            .copied()
            .chain(std::iter::repeat(indices[0]))
            .take(bucket)
            .flat_map(|i| (i as u32).to_le_bytes())
            .collect();
        let ids = self
            .context
            .upload(TensorDesc::new(DType::U32, vec![bucket])?, &bytes)?;
        let result = plan.submit(&[input.clone(), ids])?;
        result.outputs()[0].view(0, output)
    }
    /// Reduce contiguous F32 to one I32 error flag: zero iff every value is
    /// finite. Empty tensors are not accepted. Multi-pass bounded reductions
    /// avoid atomics and leave input/output ownership with normal slot leases.
    pub fn prepare_finite(&self, input: &TensorDesc) -> Result<Arc<PreparedModel>> {
        if input.dtype() != DType::F32
            || !input.is_contiguous()
            || input.is_empty()
            || input.elements() > i32::MAX as usize
        {
            return Err(Error::Message(
                "finite check requires nonempty contiguous F32".into(),
            ));
        }
        self.finite.get_or_prepare(input.clone(), || {
            let mut model = ModelSession::in_context(&self.context)?;
            let src = model.allocate(input.bytes())?;
            let mut current = src;
            let mut count = input.elements();
            let mut commands = Vec::new();
            let mut first = true;
            loop {
                let out = count.div_ceil(256);
                let dst = model.allocate(out*4)?;
                let ty = if first {"f32"} else {"i32"};
                let bad = if first {
                    "%abs = scalar.absf %value : f32\n%max = scalar.constant 3.402823466e38 : f32\n%finite = scalar.cmpf ole, %abs, %max : f32\n%bad = scf.select %finite, %fzero, %fone : f32"
                } else { "%bad = scalar.uitofp %value : i32 to f32" };
                let mut source = include_str!("finite.loom").replace("@TYPE@",ty).replace("@CHECK@",bad);
                for (name,value) in [("GRID",out.div_ceil(64)),("INPUT",count),("OUTPUT",out),("LAST_INPUT",count-1),("LAST_OUTPUT",out-1)] {
                    source = source.replace(&format!("@{name}@"),&value.to_string());
                }
                // Static bounds above cover every load, reduction flag and store.
                let kernel = unsafe {model.compile(&[(&source,Specialization::new("finite"))])?}[0];
                commands.push(Command::Dispatch(Dispatch::indices(kernel,[0],[out.div_ceil(64) as u32,1,1],vec![current.read(),dst.write()])));
                current = dst;
                if out == 1 { break; }
                count = out;
                first = false;
            }
            unsafe {model.freeze(&self.context)?.prepare(&commands,&[(src,input.clone())],&[(current,TensorDesc::new(DType::I32,vec![1])?)],3)}
        })
    }
    /// Return a resident non-finite flag without downloading the input tensor.
    pub fn finite(&self, input: &DeviceTensor) -> Result<Inference> {
        self.context.validate(input)?;
        self.prepare_finite(input.desc())?
            .submit(std::slice::from_ref(input))
    }
}
