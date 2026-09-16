use super::*;

/// Explicit byte-image interpolation contracts. The multiplication order is
/// part of each contract; algebraic rearrangement can change a rounded byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbSampling {
    /// Black border, `(value * wx) * wy`, then ties-to-even rounding.
    BlackTiesEven,
    /// Replicated border, `value * (wx * wy)`, then `floor(value + 0.5)`.
    ReplicateHalfUp,
}

/// Invert a finite FP32 2×3 affine matrix, rejecting singular or overflowing maps.
pub fn invert_affine(t: [[f32; 3]; 2]) -> Result<[[f32; 3]; 2]> {
    let det = t[0][0] * t[1][1] - t[0][1] * t[1][0];
    if !t.iter().flatten().all(|v| v.is_finite()) || !det.is_finite() || det.abs() <= 1e-20 {
        return Err(Error::Message("degenerate affine transform".into()));
    }
    let a = t[1][1] / det;
    let b = -t[0][1] / det;
    let c = -t[1][0] / det;
    let d = t[0][0] / det;
    let inverse = [
        [a, b, -a * t[0][2] - b * t[1][2]],
        [c, d, -c * t[0][2] - d * t[1][2]],
    ];
    if !inverse.iter().flatten().all(|v| v.is_finite()) {
        return Err(Error::Message("non-finite inverse transform".into()));
    }
    Ok(inverse)
}

impl ImageOps {
    /// Prepare several affine crops from one packed RGB image `[1,h,w,3]`.
    /// The second input is contiguous F32 `[crops,2,3]` (General layout), mapping
    /// integer output pixel centers to source coordinates. Output is U8 NHWC.
    /// Sampling uses the explicit FP32 border/arithmetic/rounding contract.
    /// Non-finite or excessively large coordinates produce black, never an unsafe
    /// integer conversion. Matrices are runtime data, not part of the cache key.
    pub fn prepare_affine_rgb(
        &self,
        input: &TensorDesc,
        crops: usize,
        height: usize,
        width: usize,
        sampling: RgbSampling,
    ) -> Result<Arc<PreparedModel>> {
        self.affine_plans
            .get_or_prepare((input.clone(), crops, height, width, sampling), || {
                self.affine_rgb_fragment(input, crops, height, width, sampling)?
                    .prepare(3)
            })
    }

    /// Recordable affine crop sampling for composition with model kernels.
    pub fn affine_rgb_fragment(
        &self,
        input: &TensorDesc,
        crops: usize,
        height: usize,
        width: usize,
        sampling: RgbSampling,
    ) -> Result<ModelFragment> {
        let shape = input.shape();
        if input.dtype() != DType::U8
            || input.layout() != Layout::Nhwc
            || !input.is_contiguous()
            || input.is_empty()
            || shape.len() != 4
            || shape[0] != 1
            || shape[3] != 3
            || input.elements() > i32::MAX as usize
            || crops == 0
            || height == 0
            || width == 0
        {
            return Err(Error::Message(
                "affine requires one nonempty NHWC RGB image and nonempty crops".into(),
            ));
        }
        let output =
            TensorDesc::new(DType::U8, vec![crops, height, width, 3])?.with_layout(Layout::Nhwc)?;
        let matrices = TensorDesc::new(DType::F32, vec![crops, 2, 3])?;
        if output.elements() > i32::MAX as usize || matrices.elements() > i32::MAX as usize {
            return Err(Error::Message(
                "affine dimensions exceed kernel indexing limits".into(),
            ));
        }
        let count = output.elements();
        let mut source = include_str!("affine_rgb.loom").to_owned();
        for (name, value) in [
            ("COUNT", count),
            ("LAST_OUTPUT", count - 1),
            ("INPUT", input.elements()),
            ("LAST_INPUT", input.elements() - 1),
            ("MATRICES", matrices.elements()),
            ("LAST_MATRIX", matrices.elements() - 1),
            ("GRID", count.div_ceil(256)),
            ("WIDTH", width),
            ("PIXELS", height * width),
            ("SOURCE_WIDTH", shape[2]),
            ("SOURCE_HEIGHT", shape[1]),
        ] {
            source = source.replace(&format!("@{name}@"), &value.to_string());
        }
        source = source.replace(
            "@REPLICATE@",
            if sampling == RgbSampling::ReplicateHalfUp {
                "true"
            } else {
                "false"
            },
        );
        source = source.replace(
            "@WEIGHTED_VALUE@",
            match sampling {
                RgbSampling::BlackTiesEven => {
                    "%vx = scalar.mulf %v, %wx : f32\n%vxy = scalar.mulf %vx, %wy : f32"
                }
                RgbSampling::ReplicateHalfUp => {
                    "%weight = scalar.mulf %wx, %wy : f32\n%vxy = scalar.mulf %v, %weight : f32"
                }
            },
        );
        source = source.replace("@ROUND@", match sampling {
                    RgbSampling::BlackTiesEven => "%rounded = scalar.roundevenf %value : f32",
                    RgbSampling::ReplicateHalfUp => "%half = scalar.constant 0.5 : f32\n%shifted = scalar.addf %value, %half : f32\n%rounded = scalar.floorf %shifted : f32",
                });
        let mut model = ModelSession::in_context(&self.context)?;
        let src = model.allocate(input.bytes())?;
        let maps = model.allocate(matrices.bytes())?;
        let dst = model.allocate(output.bytes())?;
        // Kernel loads are guarded before conversion and bounded to these extents.
        let kernel = unsafe { model.compile(&[(&source, Specialization::new("affine_rgb"))])? }[0];
        unsafe {
            model.freeze(&self.context)?.fragment(
                &[Command::Dispatch(Dispatch::indices(
                    kernel,
                    [0],
                    [count.div_ceil(256) as u32, 1, 1],
                    vec![src.read(), maps.read(), dst.write()],
                ))],
                &[(src, input.clone()), (maps, matrices)],
                &[(dst, output)],
            )
        }
    }

    /// Sample resident RGB into resident crops without a host wait or readback.
    pub fn affine_rgb(
        &self,
        image: &DeviceTensor,
        inverse: &DeviceTensor,
        height: usize,
        width: usize,
        sampling: RgbSampling,
    ) -> Result<Inference> {
        self.context.validate(image)?;
        self.context.validate(inverse)?;
        let shape = inverse.desc().shape();
        if shape.len() != 3
            || shape[1..] != [2, 3]
            || inverse.desc().dtype() != DType::F32
            || inverse.desc().layout() != Layout::General
            || !inverse.desc().is_contiguous()
        {
            return Err(Error::Message(
                "affine matrices must be contiguous F32 [crops,2,3] with General layout".into(),
            ));
        }
        self.prepare_affine_rgb(image.desc(), shape[0], height, width, sampling)?
            .submit(&[image.clone(), inverse.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_validation() {
        assert_eq!(
            invert_affine([[2., 0., 4.], [0., 2., -6.]]).unwrap(),
            [[0.5, -0., -2.], [-0., 0.5, 3.]]
        );
        for t in [[[0.; 3]; 2], [[f32::NAN; 3]; 2], [[f32::MAX; 3]; 2]] {
            assert!(invert_affine(t).is_err());
        }
    }
}
