//! Mixed-resolution, batched analysis. CPU decoding/downscaling is explicit;
//! landmark fitting, crop sampling and all neural-network work stay resident.
use super::*;
use hrx::image::RgbSampling;

/// Caller-prepared pixels. Keeping downscaling explicit preserves the caller's
/// antialiasing/letterbox contract and its existing stored descriptor space.
#[derive(Clone, Copy)]
pub struct AnalysisImage<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub detector_rgb: &'a [u8],
    pub detector_scale: f32,
    pub descriptor_rgb: &'a [u8],
    pub descriptor_region: [usize; 4],
}

/// Shared immutable models, bounded prepared batch shapes and source storage.
/// One active result per batch shape; retained outputs return Busy rather than
/// being overwritten. Callers control admission, including their host images.
pub struct BatchedVision {
    context: ModelContext,
    dino: Arc<dinov3_hrx::DINOv3>,
    detector: Arc<scrfd_hrx::Scrfd>,
    recognizer: Arc<arcface_hrx::ArcFace>,
    plans: PlanCache<usize, BatchPlan>,
    max_batch: usize,
    source_bytes: usize,
}

struct BatchPlan {
    first: PreparedModel,
    candidates: scrfd_hrx::postprocess::RecordedDetections,
    faces: PlanCache<(usize, usize), PreparedModel>,
}

/// Completed resident outputs in input-image order. Descriptor rows are
/// `[batch,2,384]`; face rows and embeddings use `offsets` to preserve grouping.
pub struct BatchOutput {
    pub descriptors: DeviceTensor,
    /// Original resident RGB for photos with selected faces. Views retain the
    /// packed source lease and can feed crop, swap or enhancement graphs without
    /// republishing pixels. No-face photos deliberately have no source upload.
    pub images: Vec<Option<DeviceTensor>>,
    pub rows: Option<DeviceTensor>,
    pub embeddings: Vec<DeviceTensor>,
    pub offsets: Vec<usize>,
}

impl BatchOutput {
    pub fn readback(self) -> Result<Vec<VisionOutput>> {
        let descriptors = floats(&self.descriptors)?;
        let rows = self
            .rows
            .as_ref()
            .map(floats)
            .transpose()?
            .unwrap_or_default();
        let mut embeddings = Vec::new();
        for tensor in &self.embeddings {
            embeddings.extend(
                floats(tensor)?
                    .chunks_exact(512)
                    .map(|row| <[f32; 512]>::try_from(row).unwrap()),
            );
        }
        Ok(self
            .offsets
            .windows(2)
            .enumerate()
            .map(|(image, range)| {
                let detections = (range[0]..range[1])
                    .map(|i| {
                        let row = &rows[i * 16..][..16];
                        scrfd_hrx::Detection {
                            score: row[0],
                            bbox: row[1..5].try_into().unwrap(),
                            landmarks: std::array::from_fn(|j| [row[5 + 2 * j], row[6 + 2 * j]]),
                        }
                    })
                    .collect();
                VisionOutput {
                    descriptors: Some(descriptors[image * 768..][..768].to_vec()),
                    detections,
                    embeddings: embeddings[range[0]..range[1]].to_vec(),
                }
            })
            .collect())
    }
}

impl BatchedVision {
    pub fn new(
        context: &ModelContext,
        dino: Arc<dinov3_hrx::DINOv3>,
        detector: Arc<scrfd_hrx::Scrfd>,
        recognizer: Arc<arcface_hrx::ArcFace>,
        max_batch: usize,
        source_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=32).contains(&max_batch),
            "analysis batch must be 1..=32"
        );
        ensure!(
            source_bytes > 0 && source_bytes <= i32::MAX as usize,
            "invalid source byte budget"
        );
        ensure!(
            [dino.context(), detector.context(), recognizer.context()]
                .into_iter()
                .all(|model| context.runtime().same_domain(model.runtime())),
            "all models must share the analysis context"
        );
        Ok(Self {
            context: context.clone(),
            dino,
            detector,
            recognizer,
            max_batch,
            source_bytes,
            plans: PlanCache::new(2, |plan: &BatchPlan| plan.first.is_idle())?,
        })
    }

    fn prepare(&self, batch: usize) -> hrx::Result<Arc<BatchPlan>> {
        self.plans.get_or_prepare(batch, || {
            let mut candidates = None;
            let first = PreparedModel::prepare(&self.context, 1, |context| {
                let input = |dtype, shape: Vec<usize>, layout| {
                    context.allocate_with(
                        TensorDesc::new(dtype, shape)?.with_layout(layout)?,
                        MemoryPlacement::HostVisible,
                    )
                };
                let canvas = input(DType::U8, vec![batch, 640, 640, 3], Layout::Nhwc)?;
                let plane = input(DType::U8, vec![batch, 224, 224, 3], Layout::Nhwc)?;
                let mask = input(DType::U8, vec![batch, 196], Layout::General)?;
                let metadata = input(DType::F32, vec![batch, 3], Layout::General)?;
                let mut graph = context.runtime().graph();
                let descriptor = self
                    .dino
                    .record_descriptors(&mut graph, &plane, &mask)
                    .map_err(runtime_error)?;
                candidates = Some(
                    self.detector
                        .record_letterboxed(&mut graph, &canvas, &metadata, Default::default())
                        .map_err(runtime_error)?,
                );
                Ok(InferenceGraph {
                    inputs: vec![canvas, plane, mask, metadata],
                    outputs: vec![descriptor],
                    graph: graph.prepare()?,
                })
            })?;
            Ok(BatchPlan {
                first,
                candidates: candidates.unwrap(),
                faces: PlanCache::new(8, PreparedModel::is_idle)?,
            })
        })
    }

    fn prepare_faces(
        &self,
        plan: &BatchPlan,
        count: usize,
        source_bytes: usize,
    ) -> hrx::Result<Arc<PreparedModel>> {
        plan.faces.get_or_prepare((count, source_bytes), || {
            let tensors = TensorOps::new(&self.context, 1)?;
            let images = ImageOps::new(&self.context, 1)?;
            let gather = tensors.gather_rows_fragment(plan.candidates.rows().desc(), count)?;
            PreparedModel::prepare(&self.context, 1, |context| {
                let source = context.allocate_with(
                    TensorDesc::new(DType::U8, vec![source_bytes])?,
                    MemoryPlacement::HostVisible,
                )?;
                let ids = context.allocate_with(
                    TensorDesc::new(DType::U32, vec![count])?,
                    MemoryPlacement::HostVisible,
                )?;
                let geometry = context.allocate_with(
                    TensorDesc::new(DType::U32, vec![count, 3])?,
                    MemoryPlacement::HostVisible,
                )?;
                let mut graph = context.runtime().graph();
                let rows = gather
                    .record(&mut graph, &[plan.candidates.rows().clone(), ids.clone()])?
                    .remove(0);
                let mut outputs = vec![rows.clone(), source.clone()];
                for start in (0..count).step_by(32) {
                    let n = (count - start).min(32);
                    let points = rows.view(
                        start * 64 + 20,
                        TensorDesc::strided(DType::F32, vec![n, 5, 2], vec![16, 2, 1])?,
                    )?;
                    let transforms = images
                        .similarity_2d_fragment(points.desc(), &arcface_hrx::alignment::TEMPLATE)?
                        .record(&mut graph, &[points])?;
                    let meta =
                        geometry.view(start * 12, TensorDesc::new(DType::U32, vec![n, 3])?)?;
                    let crop = images
                        .affine_packed_rgb_fragment(
                            source.desc(),
                            n,
                            112,
                            112,
                            RgbSampling::BlackTiesEven,
                        )?
                        .record(&mut graph, &[source.clone(), transforms[1].clone(), meta])?
                        .remove(0);
                    outputs.push(
                        self.recognizer
                            .record(&mut graph, &crop)
                            .map_err(runtime_error)?,
                    );
                    outputs.push(transforms[2].clone());
                }
                Ok(InferenceGraph {
                    inputs: vec![ids, geometry, source],
                    outputs,
                    graph: graph.prepare()?,
                })
            })
        })
    }

    pub fn analyze(&self, images: &[AnalysisImage<'_>]) -> Result<BatchOutput> {
        ensure!(
            !images.is_empty() && images.len() <= self.max_batch,
            "invalid analysis batch"
        );
        // Validate before reserving/preparing or publishing any input.
        for image in images {
            ensure!(
                image.width <= 32767 && image.height <= 32767,
                "source axes exceed packed sampler limit"
            );
            let [x, y, w, h] = image.descriptor_region;
            ensure!(
                image.width > 0
                    && image.height > 0
                    && image
                        .width
                        .checked_mul(image.height)
                        .and_then(|n| n.checked_mul(3))
                        == Some(image.rgb.len())
                    && image.rgb.len() <= self.source_bytes,
                "invalid source image or source byte budget exceeded"
            );
            ensure!(
                image.detector_rgb.len() == 640 * 640 * 3
                    && image.descriptor_rgb.len() == 224 * 224 * 3
                    && image.detector_scale.is_finite()
                    && image.detector_scale > 0.,
                "invalid model pixels or scale"
            );
            ensure!(
                w > 0
                    && h > 0
                    && x.checked_add(w).is_some_and(|v| v <= 224)
                    && y.checked_add(h).is_some_and(|v| v <= 224),
                "invalid descriptor region"
            );
        }
        let plan = self.prepare(images.len())?;
        let first = plan.first.try_acquire()?.submit_host_with(|index, dst| {
            match index {
                0 | 1 => {
                    let mut offset = 0;
                    for image in images {
                        let src = if index == 0 {
                            image.detector_rgb
                        } else {
                            image.descriptor_rgb
                        };
                        dst[offset..offset + src.len()].copy_from_slice(src);
                        offset += src.len();
                    }
                }
                2 => {
                    for (i, image) in images.iter().enumerate() {
                        let [x, y, w, h] = image.descriptor_region;
                        for py in 0..14 {
                            for px in 0..14 {
                                dst[i * 196 + py * 14 + px] = u8::from(
                                    px * 16 + 8 >= x
                                        && px * 16 + 8 < x + w
                                        && py * 16 + 8 >= y
                                        && py * 16 + 8 < y + h,
                                );
                            }
                        }
                    }
                }
                3 => {
                    for (row, image) in dst.chunks_exact_mut(12).zip(images) {
                        for (slot, value) in row.chunks_exact_mut(4).zip([
                            image.detector_scale,
                            (image.width / 2) as f32,
                            (image.height / 2) as f32,
                        ]) {
                            slot.copy_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                _ => unreachable!(),
            }
            Ok(())
        })?;
        let shapes: Vec<_> = images
            .iter()
            .map(|image| [image.width, image.height])
            .collect();
        let selected = plan.candidates.select(first.completion(), &shapes)?;
        let mut offsets = vec![0];
        for ids in &selected {
            offsets.push(offsets.last().unwrap() + ids.len());
        }
        let count = *offsets.last().unwrap();
        let descriptors = first.outputs()[0].clone();
        if count == 0 {
            return Ok(BatchOutput {
                descriptors,
                images: vec![None; images.len()],
                rows: None,
                embeddings: vec![],
                offsets,
            });
        }
        let needed = images
            .iter()
            .zip(&selected)
            .filter(|(_, ids)| !ids.is_empty())
            .try_fold(0usize, |total, (image, _)| {
                total.checked_add(image.rgb.len())
            })
            .ok_or_else(|| anyhow::anyhow!("source size overflow"))?;
        ensure!(
            needed <= self.source_bytes,
            "selected images exceed source byte budget"
        );
        let mut records = Vec::with_capacity(count);
        {
            let mut offset = 0;
            for (image, ids) in images.iter().zip(&selected) {
                if ids.is_empty() {
                    continue;
                }
                records.extend(std::iter::repeat_n(
                    [offset as u32, image.width as u32, image.height as u32],
                    ids.len(),
                ));
                offset += image.rgb.len();
            }
        }
        let capacity = if count <= 32 {
            count.next_power_of_two()
        } else {
            count.div_ceil(8) * 8
        };
        let source_capacity = needed.next_power_of_two().min(self.source_bytes);
        let faces = self.prepare_faces(&plan, capacity, source_capacity)?;
        let result = faces.try_acquire()?.submit_host_with(|index, dst| {
            if index == 0 {
                for (slot, &id) in dst
                    .chunks_exact_mut(4)
                    .zip(selected.iter().flatten().cycle())
                {
                    slot.copy_from_slice(&(id as u32).to_le_bytes());
                }
            } else if index == 1 {
                for (slot, &value) in dst
                    .chunks_exact_mut(4)
                    .zip(records.iter().flatten().cycle())
                {
                    slot.copy_from_slice(&value.to_le_bytes());
                }
            } else {
                let mut offset = 0;
                for (image, ids) in images.iter().zip(&selected) {
                    if !ids.is_empty() {
                        dst[offset..offset + image.rgb.len()].copy_from_slice(image.rgb);
                        offset += image.rgb.len();
                    }
                }
            }
            Ok(())
        })?;
        // Drain every reader before the first-stage slot/candidate lease can be
        // reused, including geometry errors and abandoned downstream consumers.
        let output = result.wait()?;
        for status in output[3..].iter().step_by(2) {
            ensure!(
                status
                    .binding()
                    .unwrap()
                    .map_read()?
                    .chunks_exact(4)
                    .all(|word| word == [0, 0, 0, 0]),
                "invalid alignment geometry"
            );
        }
        let embeddings = output[2..]
            .iter()
            .step_by(2)
            .zip((0..capacity).step_by(32))
            .take_while(|(_, start)| *start < count)
            .map(|(tensor, start)| {
                tensor.view(
                    0,
                    TensorDesc::new(DType::F32, vec![(count - start).min(32), 512])?
                        .with_layout(Layout::Rows)?,
                )
            })
            .collect::<hrx::Result<Vec<_>>>()?;
        let mut offset = 0;
        let resident_images = images
            .iter()
            .zip(&selected)
            .map(|(image, ids)| {
                if ids.is_empty() {
                    return Ok(None);
                }
                let view = output[1].view(
                    offset,
                    TensorDesc::new(DType::U8, vec![1, image.height, image.width, 3])?
                        .with_layout(Layout::Nhwc)?,
                )?;
                offset += image.rgb.len();
                Ok(Some(view))
            })
            .collect::<hrx::Result<Vec<_>>>()?;
        Ok(BatchOutput {
            descriptors,
            images: resident_images,
            rows: Some(output[0].view(0, TensorDesc::new(DType::F32, vec![count, 16])?)?),
            embeddings,
            offsets,
        })
    }
}

fn runtime_error(error: anyhow::Error) -> hrx::Error {
    hrx::Error::Message(error.to_string())
}
