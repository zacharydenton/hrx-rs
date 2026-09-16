use super::*;

/// Quantization contract for affine RGB compositing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbComposite {
    /// Interpolate the fractional crop, alpha-blend, then truncate to bytes.
    Truncate,
    /// Round sampled crop half-up, alpha-blend and truncate to bytes, then
    /// blend that byte result with the original using ties-to-even rounding.
    QuantizedBlend,
}

impl ImageOps {
    /// Prepare affine compositing of one F32 NHWC RGB crop and its F32
    /// `[height,width]` alpha mask into one U8 NHWC frame. RGB uses replicated
    /// borders; mask sampling uses zero borders and clamps alpha to `[0,1]`.
    ///
    /// Inputs: frame, crop, mask, F32 `[1,2,3]` frame-to-crop matrix, F32 `[5]`
    /// containing `[left,top,right,bottom,blend]`. Region bounds are exclusive
    /// on right/bottom. Pixels outside it (or with invalid coordinates) are
    /// unchanged. Blend is clamped to `[0,1]` and used only for QuantizedBlend.
    pub fn prepare_composite_rgb(
        &self,
        image: &TensorDesc,
        crop: &TensorDesc,
        mode: RgbComposite,
    ) -> Result<Arc<PreparedModel>> {
        for (desc, dtype) in [(image, DType::U8), (crop, DType::F32)] {
            if desc.dtype() != dtype
                || desc.layout() != Layout::Nhwc
                || !desc.is_contiguous()
                || desc.is_empty()
                || desc.shape().len() != 4
                || desc.shape()[0] != 1
                || desc.shape()[3] != 3
                || desc.elements() > i32::MAX as usize
                || desc.shape()[1..3].iter().any(|&n| n > 1 << 24)
            {
                return Err(Error::Message(
                    "composite requires one contiguous RGB frame and crop".into(),
                ));
            }
        }
        let mask = TensorDesc::new(DType::F32, crop.shape()[1..3].to_vec())?;
        let matrix = TensorDesc::new(DType::F32, vec![1, 2, 3])?;
        let region = TensorDesc::new(DType::F32, vec![5])?;
        self.composite_plans.get_or_prepare((image.clone(),crop.clone(),mode), || {
            let count = image.elements();
            let mut source = include_str!("composite_rgb.loom").to_owned();
            for (name,value) in [
                ("COUNT",count),("GRID",count.div_ceil(256)),("WIDTH",image.shape()[2]),
                ("LAST_OUTPUT",count-1),("LAST_CROP",crop.elements()-1),("LAST_MASK",mask.elements()-1),
                ("LAST_X",crop.shape()[2]-1),("LAST_Y",crop.shape()[1]-1),
                ("CROP",crop.elements()),("MASK",mask.elements()),
                ("CROP_WIDTH",crop.shape()[2]),("CROP_HEIGHT",crop.shape()[1]),
            ] { source = source.replace(&format!("@{name}@"),&value.to_string()); }
            source = source.replace("@CROP_ROUND@", if mode == RgbComposite::QuantizedBlend {
                "%half = scalar.constant 0.5 : f32\n%shift = scalar.addf %face0, %half : f32\n%face = scalar.floorf %shift : f32"
            } else { "%face = scalar.addf %face0, %fzero : f32" });
            source = source.replace("@FINAL_BLEND@", if mode == RgbComposite::QuantizedBlend {
                "%quantized = scalar.floorf %clamped : f32\n%blend_at = index.constant 4 : index\n%blend0 = view.load %rv[%blend_at] : view<5xf32> -> f32\n%blendlo = scalar.maxnumf %blend0, %fzero : f32\n%blend = scalar.minnumf %blendlo, %fone : f32\n%inverse = scalar.subf %fone, %blend : f32\n%before2 = scalar.mulf %original_float, %inverse : f32\n%after2 = scalar.mulf %quantized, %blend : f32\n%mixed = scalar.addf %before2, %after2 : f32\n%final = scalar.roundevenf %mixed : f32"
            } else { "%final = scalar.addf %clamped, %fzero : f32" });
            let mut model = ModelSession::in_context(&self.context)?;
            let descs = [image.clone(), crop.clone(), mask, matrix, region];
            let inputs = descs.into_iter().map(|d| Ok((model.allocate(d.bytes())?,d))).collect::<Result<Vec<_>>>()?;
            let dst = model.allocate(image.bytes())?;
            let mut bindings = inputs.iter().map(|(r,_)| r.read()).collect::<Vec<_>>();
            bindings.push(dst.write());
            // The source guards float-to-integer conversion and bounds every access.
            let kernel = unsafe { model.compile(&[(&source,Specialization::new("composite_rgb"))])? }[0];
            unsafe { model.freeze(&self.context)?.prepare(
                &[Command::Dispatch(Dispatch::indices(kernel,[0],[count.div_ceil(256) as u32,1,1],bindings))],
                &inputs, &[(dst,image.clone())], 3,
            ) }
        })
    }

    /// Composite resident pixels. Bounds and blend are small host metadata;
    /// image, crop, mask and matrix never cross the host boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn composite_rgb(
        &self,
        image: &DeviceTensor,
        crop: &DeviceTensor,
        mask: &DeviceTensor,
        matrix: &DeviceTensor,
        region: [usize; 4],
        blend: f32,
        mode: RgbComposite,
    ) -> Result<Inference> {
        for tensor in [image, crop, mask, matrix] {
            self.context.validate(tensor)?;
        }
        let plan = self.prepare_composite_rgb(image.desc(), crop.desc(), mode)?;
        let s = image.desc().shape();
        if region[0] > region[2]
            || region[1] > region[3]
            || region[2] > s[2]
            || region[3] > s[1]
            || !blend.is_finite()
            || !(0.0..=1.0).contains(&blend)
        {
            return Err(Error::Message("invalid composite region or blend".into()));
        }
        let bytes = region
            .into_iter()
            .map(|v| v as f32)
            .chain([blend])
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let metadata = self
            .context
            .upload(TensorDesc::new(DType::F32, vec![5])?, &bytes)?;
        plan.submit(&[
            image.clone(),
            crop.clone(),
            mask.clone(),
            matrix.clone(),
            metadata,
        ])
    }
}
