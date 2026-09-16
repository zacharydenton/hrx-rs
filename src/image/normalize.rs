use super::*;

impl ImageOps {
    /// Prepare packed RGB U8 NHWC to F32 NCHW conversion, computing
    /// `(pixel / 255 - mean[channel]) / std[channel]` in that order.
    pub fn prepare_normalize_rgb(
        &self,
        input: &TensorDesc,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Arc<PreparedModel>> {
        if input.dtype() != DType::U8
            || input.layout() != Layout::Nhwc
            || !input.is_contiguous()
            || input.is_empty()
            || input.shape().last() != Some(&3)
            || input.elements() > i32::MAX as usize
            || mean.iter().any(|v| !v.is_finite())
            || std.iter().any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(Error::Message(
                "normalize requires nonempty NHWC RGB and finite means/positive deviations".into(),
            ));
        }
        self.normalize_plans.get_or_prepare(
            (input.clone(), mean.map(f32::to_bits), std.map(f32::to_bits)),
            || {
                let [batch, height, width, _]: [usize; 4] = input
                    .shape()
                    .try_into()
                    .map_err(|_| Error::Message("expected NHWC input".into()))?;
                let count = input.elements();
                let output = TensorDesc::new(DType::F32, vec![batch, 3, height, width])?
                    .with_layout(Layout::Nchw)?;
                let source = format!(
                    r#"
amdgpu.target<gfx11-generic> @normalize_target {{subgroup_size = 32}}
kernel.def target(@normalize_target) export("normalize_rgb") @normalize_rgb(%unused: index) {{
  %one = index.constant 1 : index
  %threads = index.constant 256 : index
  %grid = index.constant {grid} : index
  kernel.launch.config workgroups(%grid, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%unused: index, %input: buffer, %output: buffer) {{
  %off = index.constant 0 : offset
  %zero = index.constant 0 : index
  %one = index.constant 1 : index
  %three = index.constant 3 : index
  %plane = index.constant {plane} : index
  %image_size = index.constant {image_size} : index
  %count = index.constant {count} : index
  %threads = index.constant 256 : index
  %lane = kernel.workitem.id<x> : index
  %group = kernel.workgroup.id<x> : index
  %base = index.mul %group, %threads : index
  %raw = index.add %base, %lane : index
  %ig = buffer.assume.memory_space<global> %input : buffer
  %og = buffer.assume.memory_space<global> %output : buffer
  %iv = buffer.view %ig[%off] : buffer -> view<{count}xi8>
  %ov = buffer.view %og[%off] : buffer -> view<{count}xf32>
  %valid = index.cmp ult, %raw, %count : index
  scf.if %valid {{
    %i = index.assume %raw [range(%raw, 0, {count})] : index
    %image = index.div %i, %image_size : index
    %pixel = index.rem %i, %plane : index
    %channel0 = index.div %i, %plane : index
    %channel = index.rem %channel0, %three : index
    %ip0 = index.mul %image, %plane : index
    %ip = index.add %ip0, %pixel : index
    %ic0 = index.mul %ip, %three : index
    %ic = index.add %ic0, %channel : index
    %at = index.assume %ic [range(%ic, 0, {count})] : index
    %byte = view.load %iv[%at] : view<{count}xi8> -> i8
    %uint = scalar.extui %byte : i8 to i32
    %value = scalar.uitofp %uint : i32 to f32
    %scale = scalar.constant 255.0 : f32
    %unit = scalar.divf %value, %scale : f32
    %m0 = scalar.constant {m0:.9e} : f32
    %m1 = scalar.constant {m1:.9e} : f32
    %m2 = scalar.constant {m2:.9e} : f32
    %s0 = scalar.constant {s0:.9e} : f32
    %s1 = scalar.constant {s1:.9e} : f32
    %s2 = scalar.constant {s2:.9e} : f32
    %red = index.cmp eq, %channel, %zero : index
    %green = index.cmp eq, %channel, %one : index
    %m12 = scf.select %green, %m1, %m2 : f32
    %m = scf.select %red, %m0, %m12 : f32
    %s12 = scf.select %green, %s1, %s2 : f32
    %s = scf.select %red, %s0, %s12 : f32
    %centered = scalar.subf %unit, %m : f32
    %normalized = scalar.divf %centered, %s : f32
    view.store %normalized, %ov[%i] : f32, view<{count}xf32>
  }}
  kernel.return
}}
"#,
                    grid = count.div_ceil(256),
                    plane = height * width,
                    image_size = height * width * 3,
                    m0 = mean[0],
                    m1 = mean[1],
                    m2 = mean[2],
                    s0 = std[0],
                    s1 = std[1],
                    s2 = std[2]
                );
                let mut model = ModelSession::in_context(&self.context)?;
                let src = model.allocate(input.bytes())?;
                let dst = model.allocate(output.bytes())?;
                // The generated source's accesses are bounded to these descriptors.
                let kernel =
                    unsafe { model.compile(&[(&source, Specialization::new("normalize_rgb"))])? }
                        [0];
                unsafe {
                    model.freeze(&self.context)?.prepare(
                        &[Command::Dispatch(Dispatch::indices(
                            kernel,
                            [0],
                            [count.div_ceil(256) as u32, 1, 1],
                            vec![src.read(), dst.write()],
                        ))],
                        &[(src, input.clone())],
                        &[(dst, output)],
                        3,
                    )
                }
            },
        )
    }

    /// Normalize resident RGB pixels and change layout without a host boundary.
    pub fn normalize_rgb(
        &self,
        input: &DeviceTensor,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Inference> {
        self.context.validate(input)?;
        self.prepare_normalize_rgb(input.desc(), mean, std)?
            .submit(std::slice::from_ref(input))
    }
}
