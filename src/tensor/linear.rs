use super::{DType, DeviceTensor, TensorDesc, TensorOps};
use crate::{
    Error, Result,
    loom::Specialization,
    model::{Command, Dispatch, ModelFragment, ModelSession},
};

impl TensorOps {
    /// Single-row BF16 `x * weightᵀ`, accumulating in F32. Weights are contiguous
    /// `[N, K]`; input is `[1, K]` with K a multiple of 128. Output may be BF16 or
    /// F32. This bandwidth-oriented decode operation does not accept biases.
    pub fn linear(
        &self,
        input: &DeviceTensor,
        weight: &DeviceTensor,
        output: DType,
    ) -> Result<DeviceTensor> {
        Ok(self
            .linear_many(input, std::slice::from_ref(weight), output)?
            .remove(0))
    }

    /// One dispatch for one, two or three single-row projections, e.g. Q/K/V or
    /// gate/up. Weights and output widths can differ; K and output dtype agree.
    /// Returned tensors retain a shared plan-slot lease. Three live results per
    /// shape are supported; release one before requesting a fourth.
    pub fn linear_many(
        &self,
        input: &DeviceTensor,
        weights: &[DeviceTensor],
        output: DType,
    ) -> Result<Vec<DeviceTensor>> {
        self.context.validate(input)?;
        for weight in weights {
            self.context.validate(weight)?;
        }
        let descriptions: Vec<_> = weights.iter().map(|w| w.desc().clone()).collect();
        let key = (input.desc().clone(), descriptions.clone(), output);
        let plan = self.linear.get_or_prepare(key, || {
            self.linear_fragment(input.desc(), &descriptions, output)?
                .prepare(3)
        })?;
        let mut bindings = vec![input.clone()];
        bindings.extend_from_slice(weights);
        Ok(plan.submit(&bindings)?.outputs().to_vec())
    }

    /// Record the same decode projections in a caller-owned inference graph.
    /// Fragment inputs are `[input, weights...]`, outputs follow weight order.
    pub fn linear_fragment(
        &self,
        input: &TensorDesc,
        weights: &[TensorDesc],
        output: DType,
    ) -> Result<ModelFragment> {
        let valid = input.dtype() == DType::BF16
            && input.is_contiguous()
            && input.shape().len() == 2
            && input.shape()[0] == 1
            && input.shape()[1] > 0
            && input.shape()[1].is_multiple_of(128)
            && (1..=3).contains(&weights.len())
            && matches!(output, DType::BF16 | DType::F32)
            && weights.iter().all(|w| {
                w.dtype() == DType::BF16
                    && w.is_contiguous()
                    && w.shape().len() == 2
                    && w.shape()[0] > 0
                    && w.shape()[1] == input.shape()[1]
                    && w.elements() <= 1073741824
            });
        if !valid {
            return Err(Error::Message("linear requires BF16 [1,K], one to three contiguous BF16 [N,K] weights, K divisible by 128, and BF16/F32 output".into()));
        }
        let k = input.shape()[1];
        let grid = weights
            .iter()
            .map(|w| w.shape()[0].div_ceil(8))
            .max()
            .unwrap();
        let mut model = ModelSession::in_context(&self.context)?;
        let x = model.allocate(input.bytes())?;
        let mut inputs = vec![(x, input.clone())];
        let mut outputs = Vec::new();
        let mut bindings = vec![x.read()];
        let mut source = "amdgpu.target<gfx11-generic> @target {subgroup_size = 32}\n".to_owned();
        let mut spec = Specialization::new("hrx_linear");
        for (key, value) in [("k", k), ("grid_x", grid), ("grid_y", weights.len())] {
            source += &format!(
                "config.decl @hrx.linear.{key} : %value: index where [range(%value, 1, 1073741824)]\n"
            );
            spec.set_config(format!("hrx.linear.{key}"), value.to_string());
        }
        for (i, w) in weights.iter().enumerate() {
            for (key, value) in [
                (format!("n{i}"), w.shape()[0]),
                (format!("wn{i}"), w.elements()),
            ] {
                source += &format!(
                    "config.decl @hrx.linear.{key} : %value: index where [range(%value, 1, 1073741824)]\n"
                );
                spec.set_config(format!("hrx.linear.{key}"), value.to_string());
            }
            let weight = model.allocate(w.bytes())?;
            let desc = TensorDesc::new(output, vec![1, w.shape()[0]])?;
            let out = model.allocate(desc.bytes())?;
            inputs.push((weight, w.clone()));
            outputs.push((out, desc));
            bindings.extend([weight.read(), out.write()]);
        }
        source += r#"kernel.def target(@target) export("hrx_linear") @hrx_linear(%count: index) {
 %one = index.constant 1 : index
 %threads = index.constant 256 : index
 %gx = config.get @hrx.linear.grid_x : index
 %gy = config.get @hrx.linear.grid_y : index
 kernel.launch.config workgroups(%gx, %gy, %one) workgroup_size(%threads, %one, %one) : index
} launch(%count: index, %x: buffer, "#;
        source += &(0..weights.len())
            .map(|i| format!("%w{i}: buffer, %out{i}: buffer"))
            .collect::<Vec<_>>()
            .join(", ");
        source += ") {\n %projection = kernel.workgroup.id<y> : index\n";
        let (dtype, convert) = if output == DType::BF16 {
            ("bf16", "scalar.fptrunc %sum1 : f32 to bf16")
        } else {
            ("f32", "scalar.addf %sum1, %zf : f32")
        };
        for i in 0..weights.len() {
            source += &format!(
                " %p{i} = index.constant {i} : index\n %is{i} = index.cmp eq, %projection, %p{i} : index\n scf.if %is{i} {{\n"
            );
            source += &include_str!("linear.loom")
                .replace("@P@", &i.to_string())
                .replace("@TYPE@", dtype)
                .replace("@CONVERT@", convert);
            source += "\n }\n";
        }
        source += " kernel.return\n}\n";
        // Checked shapes bound every input/output view; each wave writes one row.
        let kernel = unsafe { model.compile(&[(&source, spec)])? }[0];
        unsafe {
            model.freeze(&self.context)?.fragment(
                &[Command::Dispatch(Dispatch::indices(
                    kernel,
                    [1],
                    [grid as u32, weights.len() as u32, 1],
                    bindings,
                ))],
                &inputs,
                &outputs,
            )
        }
    }
}
