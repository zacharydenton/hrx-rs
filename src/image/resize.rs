use super::*;

/// RGB resize into a rectangle on a black canvas. Uses half-pixel bilinear
/// interpolation, clamped borders, and separable integer rounding (vertical
/// before horizontal). No antialias filter is applied when shrinking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbResize {
    /// Canvas height.
    pub height: usize,
    /// Canvas width.
    pub width: usize,
    /// Destination rectangle `[left, top, width, height]`.
    pub region: [usize; 4],
}
impl RgbResize {
    /// Resize to the entire canvas.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            region: [0, 0, width, height],
        }
    }

    /// Fit within a canvas, using integer aspect-ratio truncation. Padding is
    /// bottom/right, or split equally (with the odd pixel on bottom/right).
    pub fn letterbox(
        source_height: usize,
        source_width: usize,
        height: usize,
        width: usize,
        centered: bool,
    ) -> Result<Self> {
        if [source_height, source_width, height, width].contains(&0) {
            return Err(Error::Message("resize dimensions must be nonzero".into()));
        }
        let horizontal = (width as u128) * source_height as u128;
        let vertical = (height as u128) * source_width as u128;
        let (w, h) = if horizontal <= vertical {
            (width, (horizontal / source_width as u128) as usize)
        } else {
            ((vertical / source_height as u128) as usize, height)
        };
        if w == 0 || h == 0 {
            return Err(Error::Message("image aspect ratio is too extreme".into()));
        }
        Ok(Self {
            height,
            width,
            region: [
                if centered { (width - w) / 2 } else { 0 },
                if centered { (height - h) / 2 } else { 0 },
                w,
                h,
            ],
        })
    }
}

// Half-pixel triangle weights normalized at boundaries. Integer coefficient
// precision and pass rounding match fast_image_resize's U8 bilinear contract.
// These are shape constants, prepared once, not per-image CPU processing.
fn coefficients(source: usize, output: usize) -> (Vec<u8>, usize) {
    let weights: Vec<_> = (0..output)
        .map(|i| {
            let x = (i as f64 + 0.5) * (source as f64 / output as f64) - 0.5;
            let start = x.floor();
            if start < 0. {
                (0, 0, 1., 0.)
            } else if start >= (source - 1) as f64 {
                (source - 1, source - 1, 1., 0.)
            } else {
                (
                    start as usize,
                    start as usize + 1,
                    1. - (x - start),
                    x - start,
                )
            }
        })
        .collect();
    let max = weights.iter().map(|w| w.2.max(w.3)).fold(0f64, f64::max);
    let precision = (0..22)
        .find(|p| (max * (1u32 << (p + 1)) as f64).round() >= 32768.)
        .unwrap_or(21);
    let scale = (1u32 << precision) as f64;
    let bytes = weights
        .iter()
        .flat_map(|&(a, b, x, y)| {
            [
                a as i32,
                b as i32,
                (x * scale).round() as i32,
                (y * scale).round() as i32,
            ]
            .into_iter()
            .flat_map(i32::to_le_bytes)
        })
        .collect();
    (bytes, precision)
}

impl ImageOps {
    /// Prepare a reusable three-slot RGB resize/letterbox plan for contiguous
    /// U8 NHWC `[batch,height,width,3]`. Only image bytes are runtime inputs.
    pub fn prepare_resize_rgb(
        &self,
        input: &TensorDesc,
        resize: RgbResize,
    ) -> Result<Arc<PreparedModel>> {
        let shape = input.shape();
        let [left, top, width, height] = resize.region;
        if input.dtype() != DType::U8
            || input.layout() != Layout::Nhwc
            || !input.is_contiguous()
            || input.is_empty()
            || shape.len() != 4
            || shape[3] != 3
            || input.elements() > i32::MAX as usize
            || width == 0
            || height == 0
            || left.checked_add(width).is_none_or(|v| v > resize.width)
            || top.checked_add(height).is_none_or(|v| v > resize.height)
        {
            return Err(Error::Message(
                "invalid NHWC RGB resize dimensions or rectangle".into(),
            ));
        }
        let output = TensorDesc::new(DType::U8, vec![shape[0], resize.height, resize.width, 3])?
            .with_layout(Layout::Nhwc)?;
        let intermediate = TensorDesc::new(DType::U8, vec![shape[0], height, shape[2], 3])?;
        if output.elements() > i32::MAX as usize || intermediate.elements() > i32::MAX as usize {
            return Err(Error::Message(
                "resize exceeds kernel indexing limits".into(),
            ));
        }
        self.resize_plans
            .get_or_prepare((input.clone(), resize), || {
                let mut model = ModelSession::in_context(&self.context)?;
                let src = model.allocate(input.bytes())?;
                let tmp = model.allocate(intermediate.bytes())?;
                let dst = model.allocate(output.bytes())?;
                let mut commands = Vec::new();
                for (vertical, from, to, sh, sw, dh, dw, x, y, innerw, innerh, elements) in [
                    (
                        true,
                        src,
                        tmp,
                        shape[1],
                        shape[2],
                        height,
                        shape[2],
                        0,
                        0,
                        shape[2],
                        height,
                        intermediate.elements(),
                    ),
                    (
                        false,
                        tmp,
                        dst,
                        height,
                        shape[2],
                        resize.height,
                        resize.width,
                        left,
                        top,
                        width,
                        height,
                        output.elements(),
                    ),
                ] {
                    let axis = if vertical { innerh } else { innerw };
                    let (bytes, precision) = coefficients(if vertical { sh } else { sw }, axis);
                    let table = model.weight(&bytes)?;
                    let mut source = include_str!("resize_rgb.loom")
                        .replace("@VERTICAL@", if vertical { "true" } else { "false" });
                    for (name, value) in [
                        ("COUNT", elements),
                        ("LAST", elements - 1),
                        ("INPUT", shape[0] * sh * sw * 3),
                        ("INPUT_LAST", shape[0] * sh * sw * 3 - 1),
                        ("COEFFS", axis * 4),
                        ("COEFFS_LAST", axis * 4 - 1),
                        ("AXIS_LAST", if vertical { sh - 1 } else { sw - 1 }),
                        ("GRID", elements.div_ceil(256)),
                        ("SH", sh),
                        ("SW", sw),
                        ("DH", dh),
                        ("DW", dw),
                        ("LEFT", x),
                        ("TOP", y),
                        ("RIGHT", x + innerw),
                        ("BOTTOM", y + innerh),
                        ("PRECISION", precision),
                        ("ROUND", 1 << (precision - 1)),
                    ] {
                        source = source.replace(&format!("@{name}@"), &value.to_string());
                    }
                    // All table entries are generated within the source extent;
                    // output coordinates are guarded before addressing the table.
                    let kernel =
                        unsafe { model.compile(&[(&source, Specialization::new("resize_rgb"))])? }
                            [0];
                    commands.push(Command::Dispatch(Dispatch::indices(
                        kernel,
                        [0],
                        [elements.div_ceil(256) as u32, 1, 1],
                        vec![from.read(), table.read(), to.write()],
                    )));
                }
                unsafe {
                    model.freeze(&self.context)?.prepare(
                        &commands,
                        &[(src, input.clone())],
                        &[(dst, output)],
                        3,
                    )
                }
            })
    }

    /// Resize a resident RGB batch; output padding is initialized to black on
    /// every replay. Retained outputs apply bounded per-shape backpressure.
    pub fn resize_rgb(&self, image: &DeviceTensor, resize: RgbResize) -> Result<Inference> {
        self.context.validate(image)?;
        self.prepare_resize_rgb(image.desc(), resize)?
            .submit(std::slice::from_ref(image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn letterbox_geometry_is_checked() {
        assert_eq!(
            RgbResize::letterbox(100, 200, 640, 640, false)
                .unwrap()
                .region,
            [0, 0, 640, 320]
        );
        assert_eq!(
            RgbResize::letterbox(200, 100, 640, 640, true)
                .unwrap()
                .region,
            [160, 0, 320, 640]
        );
        assert!(RgbResize::letterbox(0, 100, 640, 640, false).is_err());
        assert!(RgbResize::letterbox(1, 10000, 1, 1, false).is_err());
    }
}
