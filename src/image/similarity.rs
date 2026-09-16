use super::*;
use std::fmt::Write;

/// Least-squares orientation-preserving 2D similarity, mapping source points
/// to template points in FP32. No robust outlier rejection. Collinear points
/// are allowed if the fit has nonzero scale; collapsed/non-finite fits fail.
pub fn fit_similarity_2d(points: &[[f32; 2]], template: &[[f32; 2]]) -> Result<[[f32; 3]; 2]> {
    if points.len() < 2
        || points.len() != template.len()
        || !points
            .iter()
            .chain(template)
            .flatten()
            .all(|v| v.is_finite())
    {
        return Err(Error::Message(
            "similarity needs matching finite point sets".into(),
        ));
    }
    let mean = |points: &[[f32; 2]]| {
        [
            points.iter().map(|p| p[0]).sum::<f32>() / points.len() as f32,
            points.iter().map(|p| p[1]).sum::<f32>() / points.len() as f32,
        ]
    };
    let p = mean(points);
    let q = mean(template);
    let (mut den, mut a, mut b) = (0., 0., 0.);
    for (s, d) in points.iter().zip(template) {
        let (x, y, u, v) = (s[0] - p[0], s[1] - p[1], d[0] - q[0], d[1] - q[1]);
        den += x * x + y * y;
        a += x * u + y * v;
        b += x * v - y * u;
    }
    a /= den;
    b /= den;
    let result = [
        [a, -b, q[0] - a * p[0] + b * p[1]],
        [b, a, q[1] - b * p[0] - a * p[1]],
    ];
    invert_affine(result)?;
    Ok(result)
}

impl ImageOps {
    /// Prepare FP32 similarity fitting for `[batch,points,2]`. Positive
    /// non-overlapping strides allow direct landmark views of detection rows.
    /// Template coordinates are immutable shape-plan data (2..=64 points).
    /// Outputs: forward `[batch,2,3]`, inverse `[batch,2,3]`, status I32 `[batch]`.
    /// Status is zero for success and one for invalid/degenerate geometry; both
    /// matrices are NaN on failure. Consumers must preserve/check status, not
    /// treat a black crop from invalid geometry as a successful alignment.
    pub fn prepare_similarity_2d(
        &self,
        points: &TensorDesc,
        template: &[[f32; 2]],
    ) -> Result<Arc<PreparedModel>> {
        let key = (
            points.clone(),
            template.iter().flatten().map(|v| v.to_bits()).collect(),
        );
        self.similarity_plans.get_or_prepare(key, || {
            self.similarity_2d_fragment(points, template)?.prepare(3)
        })
    }

    /// Recordable fit producing forward matrices, inverse matrices and status.
    pub fn similarity_2d_fragment(
        &self,
        points: &TensorDesc,
        template: &[[f32; 2]],
    ) -> Result<ModelFragment> {
        let shape = points.shape();
        if points.dtype() != DType::F32
            || points.layout() != Layout::General
            || shape.len() != 3
            || shape[2] != 2
            || shape[0] == 0
            || !(2..=64).contains(&template.len())
            || shape[1] != template.len()
            || points.bytes() / 4 > i32::MAX as usize
            || !template.iter().flatten().all(|v| v.is_finite())
        {
            return Err(Error::Message(
                "similarity needs F32 [batch,points,2] and a finite template".into(),
            ));
        }
        let batch = shape[0];
        let count = template.len();
        let elements = points.bytes() / 4;
        let matrices = TensorDesc::new(DType::F32, vec![batch, 2, 3])?;
        let status = TensorDesc::new(DType::I32, vec![batch])?;
        if matrices.elements() > i32::MAX as usize {
            return Err(Error::Message("similarity output too large".into()));
        }
        let q = [0, 1].map(|axis| template.iter().map(|p| p[axis]).sum::<f32>() / count as f32);
        if !q.iter().all(|v| v.is_finite()) {
            return Err(Error::Message("template mean overflow".into()));
        }
        let mut body = String::new();
        for (i, point) in template.iter().enumerate() {
            for (axis, letter) in [(0, "x"), (1, "y")] {
                let offset = i * points.strides()[1] + axis * points.strides()[2];
                writeln!(
                    body,
                    r#"
    %off{letter}{i} = index.constant {offset} : index
    %raw{letter}{i} = index.add %inputbase, %off{letter}{i} : index
    %at{letter}{i} = index.assume %raw{letter}{i} [range(%raw{letter}{i}, 0, {last})] : index
    %{letter}{i} = view.load %src[%at{letter}{i}] : view<{elements}xf32> -> f32
    %abs{letter}{i} = scalar.absf %{letter}{i} : f32
    %finite{letter}{i} = scalar.cmpf ole, %abs{letter}{i}, %max : f32
    %sum{letter}{next} = scalar.addf %sum{letter}{i}, %{letter}{i} : f32
"#,
                    last = elements - 1,
                    next = i + 1
                )
                .unwrap();
            }
            writeln!(body, "    %xyfinite{i} = scalar.andi %finitex{i}, %finitey{i} : i1\n    %finite{next} = scalar.andi %finite{i}, %xyfinite{i} : i1",next=i+1).unwrap();
            for (axis, letter) in [(0, "u"), (1, "v")] {
                writeln!(
                    body,
                    "    %{letter}{i} = scalar.constant {:.9e} : f32",
                    point[axis] - q[axis]
                )
                .unwrap();
            }
        }
        writeln!(body,"    %px = scalar.divf %sumx{count}, %n : f32\n    %py = scalar.divf %sumy{count}, %n : f32").unwrap();
        for i in 0..count {
            writeln!(
                body,
                r#"
    %sx{i} = scalar.subf %x{i}, %px : f32
    %sy{i} = scalar.subf %y{i}, %py : f32
    %xx{i} = scalar.mulf %sx{i}, %sx{i} : f32
    %yy{i} = scalar.mulf %sy{i}, %sy{i} : f32
    %xy{i} = scalar.addf %xx{i}, %yy{i} : f32
    %den{next} = scalar.addf %den{i}, %xy{i} : f32
    %xu{i} = scalar.mulf %sx{i}, %u{i} : f32
    %yv{i} = scalar.mulf %sy{i}, %v{i} : f32
    %dot{i} = scalar.addf %xu{i}, %yv{i} : f32
    %dot{next}sum = scalar.addf %dot{i}sum, %dot{i} : f32
    %xv{i} = scalar.mulf %sx{i}, %v{i} : f32
    %yu{i} = scalar.mulf %sy{i}, %u{i} : f32
    %cross{i} = scalar.subf %xv{i}, %yu{i} : f32
    %cross{next}sum = scalar.addf %cross{i}sum, %cross{i} : f32
"#,
                next = i + 1
            )
            .unwrap();
        }
        let mut source = include_str!("similarity_2d.loom").replace("@BODY@", &body);
        for (name, value) in [
            ("BATCH", batch),
            ("BATCH_LAST", batch - 1),
            ("GRID", batch.div_ceil(64)),
            ("INPUT", elements),
            ("STRIDE", points.strides()[0]),
            ("COUNT", count),
            ("OUTPUT", batch * 6),
            ("OUTPUT_LAST", batch * 6 - 1),
        ] {
            source = source.replace(&format!("@{name}@"), &value.to_string());
        }
        source = source
            .replace("@QX@", &format!("{:.9e}", q[0]))
            .replace("@QY@", &format!("{:.9e}", q[1]));
        let mut model = ModelSession::in_context(&self.context)?;
        let input = model.allocate(points.bytes())?;
        let forward = model.allocate(matrices.bytes())?;
        let inverse = model.allocate(matrices.bytes())?;
        let flags = model.allocate(status.bytes())?;
        let kernel =
            unsafe { model.compile(&[(&source, Specialization::new("similarity_2d"))])? }[0];
        unsafe {
            model.freeze(&self.context)?.fragment(
                &[Command::Dispatch(Dispatch::indices(
                    kernel,
                    [0],
                    [batch.div_ceil(64) as u32, 1, 1],
                    vec![
                        input.read(),
                        forward.write(),
                        inverse.write(),
                        flags.write(),
                    ],
                ))],
                &[(input, points.clone())],
                &[
                    (forward, matrices.clone()),
                    (inverse, matrices),
                    (flags, status),
                ],
            )
        }
    }

    /// Fit resident landmarks; no landmark or matrix readback is performed.
    pub fn similarity_2d(&self, points: &DeviceTensor, template: &[[f32; 2]]) -> Result<Inference> {
        self.context.validate(points)?;
        self.prepare_similarity_2d(points.desc(), template)?
            .submit(std::slice::from_ref(points))
    }
}
