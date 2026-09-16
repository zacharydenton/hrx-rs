//! Device-resident image transformations with bounded reusable plans.

use crate::{
    Error, Result,
    inference::{Inference, ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, ModelFragment, ModelSession},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};
use std::sync::Arc;
mod affine;
mod composite;
mod normalize;
mod resize;
mod similarity;
mod views;
pub use affine::{RgbSampling, invert_affine};
pub use composite::RgbComposite;
pub use resize::RgbResize;
pub use similarity::fit_similarity_2d;
pub use views::RgbEncoding;

type NormalizationKey = (TensorDesc, [u32; 3], [u32; 3]);

/// GPU image operations sharing the caller's model context and compilation cache.
pub struct ImageOps {
    context: ModelContext,
    patch_plans: PlanCache<(TensorDesc, usize), PreparedModel>,
    normalize_plans: PlanCache<NormalizationKey, PreparedModel>,
    affine_plans: PlanCache<(TensorDesc, usize, usize, usize, RgbSampling), PreparedModel>,
    resize_plans: PlanCache<(TensorDesc, RgbResize), PreparedModel>,
    similarity_plans: PlanCache<(TensorDesc, Vec<u32>), PreparedModel>,
    view_plans: PlanCache<(TensorDesc, usize, RgbEncoding), PreparedModel>,
    composite_plans: PlanCache<(TensorDesc, TensorDesc, RgbComposite), PreparedModel>,
}
impl ImageOps {
    /// Create a bounded shape cache. Preparation is lazy, never per warm request.
    pub fn new(context: &ModelContext, cache_capacity: usize) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            patch_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            normalize_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            affine_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            resize_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            similarity_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            view_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
            composite_plans: PlanCache::new(cache_capacity, PreparedModel::is_idle)?,
        })
    }

    /// Prepare NCHW f32 patchification to `[batch, patches, channels*p*p]`.
    /// Within a patch, values are in channel, row, column order. No arithmetic
    /// changes the values, including non-finite payloads. Both axes must divide p.
    pub fn prepare_patchify(&self, input: &TensorDesc, patch: usize) -> Result<Arc<PreparedModel>> {
        self.patch_plans.get_or_prepare((input.clone(), patch), || {
            self.patchify_fragment(input, patch)?.prepare(3)
        })
    }

    /// Recordable patchification for direct composition with a model graph.
    pub fn patchify_fragment(&self, input: &TensorDesc, patch: usize) -> Result<ModelFragment> {
        if input.dtype() != DType::F32
            || input.layout() != Layout::Nchw
            || !input.is_contiguous()
            || input.is_empty()
        {
            return Err(Error::Message(
                "patchify requires nonempty contiguous NCHW f32".into(),
            ));
        }
        let [batch, channels, height, width]: [usize; 4] = input
            .shape()
            .try_into()
            .map_err(|_| Error::Message("patchify requires four axes".into()))?;
        if patch == 0
            || !height.is_multiple_of(patch)
            || !width.is_multiple_of(patch)
            || input.elements() > i32::MAX as usize
        {
            return Err(Error::Message("invalid patch or image size".into()));
        }
        let count = input.elements();
        let cells = (height / patch) * (width / patch);
        let patch_elements = channels * patch * patch;
        let output = TensorDesc::new(DType::F32, vec![batch, cells, patch_elements])?;
        let source = format!(
            r#"
amdgpu.target<gfx11-generic> @image_target {{subgroup_size = 32}}
kernel.def target(@image_target) export("patchify") @patchify(%unused: index) {{
  %one = index.constant 1 : index
  %threads = index.constant 256 : index
  %grid = index.constant {grid} : index
  kernel.launch.config workgroups(%grid, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%unused: index, %input: buffer, %output: buffer) {{
  %zero = index.constant 0 : offset
  %threads = index.constant 256 : index
  %lane = kernel.workitem.id<x> : index
  %group = kernel.workgroup.id<x> : index
  %base = index.mul %group, %threads : index
  %raw = index.add %base, %lane : index
  %count = index.constant {count} : index
  %p = index.constant {patch} : index
  %pp = index.constant {pp} : index
  %pe = index.constant {patch_elements} : index
  %ch = index.constant {channels} : index
  %pxs = index.constant {pxs} : index
  %cells = index.constant {cells} : index
  %h = index.constant {height} : index
  %w = index.constant {width} : index
  %inputg = buffer.assume.memory_space<global> %input : buffer
  %outputg = buffer.assume.memory_space<global> %output : buffer
  %src = buffer.view %inputg[%zero] : buffer -> view<{count}xf32>
  %dst = buffer.view %outputg[%zero] : buffer -> view<{count}xf32>
  %valid = index.cmp ult, %raw, %count : index
  scf.if %valid {{
    %i = index.assume %raw [range(%raw, 0, {count})] : index
    %dx = index.rem %i, %p : index
    %qdy = index.div %i, %p : index
    %dy = index.rem %qdy, %p : index
    %qc = index.div %i, %pp : index
    %c = index.rem %qc, %ch : index
    %cell = index.div %i, %pe : index
    %image = index.div %cell, %cells : index
    %incell = index.rem %cell, %cells : index
    %px = index.rem %incell, %pxs : index
    %py = index.div %incell, %pxs : index
    %x0 = index.mul %px, %p : index
    %x = index.add %x0, %dx : index
    %y0 = index.mul %py, %p : index
    %y = index.add %y0, %dy : index
    %ic0 = index.mul %image, %ch : index
    %ic = index.add %ic0, %c : index
    %iy0 = index.mul %ic, %h : index
    %iy = index.add %iy0, %y : index
    %ix0 = index.mul %iy, %w : index
    %ix = index.add %ix0, %x : index
    %j = index.assume %ix [range(%ix, 0, {count})] : index
    %value = view.load %src[%j] : view<{count}xf32> -> f32
    view.store %value, %dst[%i] : f32, view<{count}xf32>
  }}
  kernel.return
}}
"#,
            grid = count.div_ceil(256),
            pp = patch * patch,
            pxs = width / patch
        );
        let mut model = ModelSession::in_context(&self.context)?;
        let src = model.allocate(input.bytes())?;
        let dst = model.allocate(output.bytes())?;
        // The generated kernel's scalar and memory extents are fixed above.
        let kernel = unsafe { model.compile(&[(&source, Specialization::new("patchify"))])? }[0];
        let definition = model.freeze(&self.context)?;
        unsafe {
            definition.fragment(
                &[Command::Dispatch(Dispatch::indices(
                    kernel,
                    [0],
                    [count.div_ceil(256) as u32, 1, 1],
                    vec![src.read(), dst.write()],
                ))],
                &[(src, input.clone())],
                &[(dst, output)],
            )
        }
    }

    /// Patchify a resident image batch without a host wait or readback.
    pub fn patchify(&self, input: &DeviceTensor, patch: usize) -> Result<Inference> {
        self.context.validate(input)?;
        self.prepare_patchify(input.desc(), patch)?
            .submit(std::slice::from_ref(input))
    }
}
