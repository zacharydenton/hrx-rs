use super::*;

/// Numerical contracts for RGB model inputs and outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbEncoding {
    /// Encode byte / 255; decode float * 255 without clipping or rounding.
    Unit,
    /// Encode byte / 127.5 - 1; decode round-even((clamp(x,-1,1)+1)*127.5).
    Symmetric,
}

impl ImageOps {
    /// Split packed U8 NHWC into interleaved F32 NCHW views. A factor of f
    /// produces f² views per image: view (dy,dx) samples image[y*f+dy,x*f+dx].
    /// This is a lossless layout permutation before the selected normalization,
    /// not a resize. Both spatial dimensions must divide the factor.
    pub fn prepare_encode_rgb_views(
        &self,
        input: &TensorDesc,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<Arc<PreparedModel>> {
        if input.dtype() != DType::U8 || input.layout() != Layout::Nhwc {
            return Err(Error::Message("RGB view encoding requires U8 NHWC".into()));
        }
        self.view_plans
            .get_or_prepare((input.clone(), factor, encoding), || {
                self.rgb_views_fragment(input, factor, encoding)?.prepare(3)
            })
    }

    /// Reassemble interleaved F32 NCHW views as F32 NHWC byte-level RGB.
    /// The view batch must be a multiple of factor². Unit values retain the
    /// network's fractional output; Symmetric values use the documented clamp.
    pub fn prepare_decode_rgb_views(
        &self,
        input: &TensorDesc,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<Arc<PreparedModel>> {
        if input.dtype() != DType::F32 || input.layout() != Layout::Nchw {
            return Err(Error::Message("RGB view decoding requires F32 NCHW".into()));
        }
        self.view_plans
            .get_or_prepare((input.clone(), factor, encoding), || {
                self.rgb_views_fragment(input, factor, encoding)?.prepare(3)
            })
    }

    /// Record RGB view encoding with directly bound input pixels.
    pub fn encode_rgb_views_fragment(
        &self,
        input: &TensorDesc,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<ModelFragment> {
        if input.dtype() != DType::U8 || input.layout() != Layout::Nhwc {
            return Err(Error::Message("RGB view encoding requires U8 NHWC".into()));
        }
        self.rgb_views_fragment(input, factor, encoding)
    }

    /// Record RGB view decoding with directly bound model output.
    pub fn decode_rgb_views_fragment(
        &self,
        input: &TensorDesc,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<ModelFragment> {
        if input.dtype() != DType::F32 || input.layout() != Layout::Nchw {
            return Err(Error::Message("RGB view decoding requires F32 NCHW".into()));
        }
        self.rgb_views_fragment(input, factor, encoding)
    }

    fn rgb_views_fragment(
        &self,
        input: &TensorDesc,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<ModelFragment> {
        if !input.is_contiguous()
            || input.is_empty()
            || input.shape().len() != 4
            || input.elements() > i32::MAX as usize
            || factor == 0
        {
            return Err(Error::Message("invalid RGB view dimensions".into()));
        }
        let ff = factor
            .checked_mul(factor)
            .ok_or_else(|| Error::Message("RGB view factor overflow".into()))?;
        let encode = input.dtype() == DType::U8;
        let s = input.shape();
        let (batch, height, width) = if encode {
            if s[3] != 3 || !s[1].is_multiple_of(factor) || !s[2].is_multiple_of(factor) {
                return Err(Error::Message(
                    "RGB axes must divide the view factor".into(),
                ));
            }
            (s[0], s[1], s[2])
        } else {
            if s[1] != 3 || !s[0].is_multiple_of(ff) {
                return Err(Error::Message(
                    "RGB view batch must divide factor squared".into(),
                ));
            }
            (s[0] / ff, s[2] * factor, s[3] * factor)
        };
        let vh = height / factor;
        let vw = width / factor;
        let output = if encode {
            TensorDesc::new(DType::F32, vec![batch * ff, 3, vh, vw])?.with_layout(Layout::Nchw)?
        } else {
            TensorDesc::new(DType::F32, vec![batch, height, width, 3])?.with_layout(Layout::Nhwc)?
        };
        {
            let count = input.elements();
            let address = if encode {
                format!(
                    r#"
    %x = index.rem %i, %vw : index
    %yy = index.div %i, %vw : index
    %y = index.rem %yy, %vh : index
    %cc = index.div %i, %plane : index
    %ch = index.rem %cc, %three : index
    %view = index.div %cc, %three : index
    %image = index.div %view, %ff : index
    %part = index.rem %view, %ff : index
    %dx = index.rem %part, %factor : index
    %dy = index.div %part, %factor : index
    %xf = index.mul %x, %factor : index
    %xx = index.add %xf, %dx : index
    %yf = index.mul %y, %factor : index
    %yp = index.add %yf, %dy : index
    %ih = index.mul %image, %height : index
    %row = index.add %ih, %yp : index
    %rw = index.mul %row, %width : index
    %pixel = index.add %rw, %xx : index
    %base = index.mul %pixel, %three : index
    %addr = index.add %base, %ch : index
    %at = index.assume %addr [range(%addr, 0, {count})] : index
    %byte = view.load %src[%at] : view<{count}xi8> -> i8
    %uint = scalar.extui %byte : i8 to i32
    %value = scalar.uitofp %uint : i32 to f32
"#
                )
            } else {
                format!(
                    r#"
    %ch = index.rem %i, %three : index
    %pixel = index.div %i, %three : index
    %x = index.rem %pixel, %width : index
    %yy = index.div %pixel, %width : index
    %y = index.rem %yy, %height : index
    %image = index.div %yy, %height : index
    %dx = index.rem %x, %factor : index
    %dy = index.rem %y, %factor : index
    %xx = index.div %x, %factor : index
    %yp = index.div %y, %factor : index
    %df = index.mul %dy, %factor : index
    %part = index.add %df, %dx : index
    %im = index.mul %image, %ff : index
    %view = index.add %im, %part : index
    %vc = index.mul %view, %three : index
    %channel = index.add %vc, %ch : index
    %hh = index.mul %channel, %vh : index
    %row = index.add %hh, %yp : index
    %rw = index.mul %row, %vw : index
    %addr = index.add %rw, %xx : index
    %at = index.assume %addr [range(%addr, 0, {count})] : index
    %value = view.load %src[%at] : view<{count}xf32> -> f32
"#
                )
            };
            let arithmetic = match (encode, encoding) {
                (true, RgbEncoding::Unit) => {
                    "%scale = scalar.constant 255.0 : f32\n%result = scalar.divf %value, %scale : f32"
                }
                (true, RgbEncoding::Symmetric) => {
                    "%scale = scalar.constant 127.5 : f32\n%onef = scalar.constant 1.0 : f32\n%scaled = scalar.divf %value, %scale : f32\n%result = scalar.subf %scaled, %onef : f32"
                }
                (false, RgbEncoding::Unit) => {
                    "%scale = scalar.constant 255.0 : f32\n%result = scalar.mulf %value, %scale : f32"
                }
                (false, RgbEncoding::Symmetric) => {
                    "%scale = scalar.constant 127.5 : f32\n%onef = scalar.constant 1.0 : f32\n%negative = scalar.constant -1.0 : f32\n%low = scalar.maxnumf %value, %negative : f32\n%clamped = scalar.minnumf %low, %onef : f32\n%shifted = scalar.addf %clamped, %onef : f32\n%scaled = scalar.mulf %shifted, %scale : f32\n%result = scalar.roundevenf %scaled : f32"
                }
            };
            let ty = if encode { "i8" } else { "f32" };
            let source = format!(
                r#"
amdgpu.target<gfx11-generic> @views_target {{subgroup_size = 32}}
kernel.def target(@views_target) export("rgb_views") @rgb_views(%unused: index) {{
  %one = index.constant 1 : index
  %threads = index.constant 256 : index
  %grid = index.constant {grid} : index
  kernel.launch.config workgroups(%grid, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%unused: index, %input: buffer, %output: buffer) {{
  %off = index.constant 0 : offset
  %threads = index.constant 256 : index
  %lane = kernel.workitem.id<x> : index
  %group = kernel.workgroup.id<x> : index
  %basei = index.mul %group, %threads : index
  %raw = index.add %basei, %lane : index
  %count = index.constant {count} : index
  %three = index.constant 3 : index
  %width = index.constant {width} : index
  %height = index.constant {height} : index
  %vw = index.constant {vw} : index
  %vh = index.constant {vh} : index
  %plane = index.constant {plane} : index
  %factor = index.constant {factor} : index
  %ff = index.constant {ff} : index
  %ig = buffer.assume.memory_space<global> %input : buffer
  %og = buffer.assume.memory_space<global> %output : buffer
  %src = buffer.view %ig[%off] : buffer -> view<{count}x{ty}>
  %dst = buffer.view %og[%off] : buffer -> view<{count}xf32>
  %valid = index.cmp ult, %raw, %count : index
  scf.if %valid {{
    %i = index.assume %raw [range(%raw, 0, {count})] : index
    {address}
    {arithmetic}
    view.store %result, %dst[%i] : f32, view<{count}xf32>
  }}
  kernel.return
}}
"#,
                grid = count.div_ceil(256),
                plane = vh * vw
            );
            let mut model = ModelSession::in_context(&self.context)?;
            let src = model.allocate(input.bytes())?;
            let dst = model.allocate(output.bytes())?;
            // Permutations above are bijections within checked input/output extents.
            let kernel =
                unsafe { model.compile(&[(&source, Specialization::new("rgb_views"))])? }[0];
            unsafe {
                model.freeze(&self.context)?.fragment(
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
    }

    /// Encode resident pixels without a host boundary.
    pub fn encode_rgb_views(
        &self,
        input: &DeviceTensor,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<Inference> {
        self.context.validate(input)?;
        self.prepare_encode_rgb_views(input.desc(), factor, encoding)?
            .submit(std::slice::from_ref(input))
    }
    /// Decode resident model views without a host boundary.
    pub fn decode_rgb_views(
        &self,
        input: &DeviceTensor,
        factor: usize,
        encoding: RgbEncoding,
    ) -> Result<Inference> {
        self.context.validate(input)?;
        self.prepare_decode_rgb_views(input.desc(), factor, encoding)?
            .submit(std::slice::from_ref(input))
    }
}
