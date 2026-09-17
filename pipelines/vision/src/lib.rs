//! Cross-model resident vision pipelines using the shared HRX image operations.
//!
//! One fixed-shape RGB publication feeds DINO and SCRFD in one graph. CPU NMS
//! reads compact summaries and publishes selected row indices. A second graph
//! gathers resident landmarks and runs ArcFace against the same original image.
//! JPEG/PNG decoding belongs to the caller. Model-specific code stays in clients.
use anyhow::{Result, ensure};
use hrx::{
    execution::MemoryPlacement,
    image::{ImageOps, RgbResize},
    inference::{Inference, InferenceGraph, ModelContext, PreparedModel},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc, TensorOps},
};
use std::sync::Arc;
mod batch;
pub use batch::{AnalysisImage, BatchOutput, BatchedVision};

/// One fixed image shape with bounded face-count specialization and one active
/// image lease. Returned tensors retain their slots; another call returns Busy
/// rather than overwriting retained descriptors or embeddings.
pub struct ResidentVision {
    context: ModelContext,
    first: PreparedModel,
    image: DeviceTensor,
    candidates: scrfd_hrx::postprocess::RecordedDetections,
    faces: PlanCache<usize, PreparedModel>,
    recognizer: Arc<arcface_hrx::ArcFace>,
    tensors: TensorOps,
    shape: [usize; 2],
}

/// Validated, completed outputs. No descriptor, embedding or selected landmark
/// bytes are read by the CPU until `readback`; tensors can feed another GPU graph.
pub struct ResidentOutput {
    pub descriptors: Option<DeviceTensor>,
    pub rows: Option<DeviceTensor>,
    pub embeddings: Vec<DeviceTensor>,
    // Retain the first graph slot even for a no-face, face-only result.
    _input: DeviceTensor,
}

pub struct VisionOutput {
    pub descriptors: Option<Vec<f32>>,
    pub detections: Vec<scrfd_hrx::Detection>,
    pub embeddings: Vec<[f32; arcface_hrx::EMBEDDING]>,
}

fn floats(tensor: &DeviceTensor) -> Result<Vec<f32>> {
    tensor.completion().wait()?;
    let binding = tensor.binding().expect("nonempty output");
    let bytes = binding.map_read()?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

impl ResidentOutput {
    /// Original RGB, still resident and protected by the image slot lease.
    /// This can feed a subsequent crop, enhancement or compositing pipeline.
    pub fn image(&self) -> &DeviceTensor {
        &self._input
    }
    pub fn readback(self) -> Result<VisionOutput> {
        let descriptors = self.descriptors.as_ref().map(floats).transpose()?;
        let rows = self
            .rows
            .as_ref()
            .map(floats)
            .transpose()?
            .unwrap_or_default();
        let detections = rows
            .chunks_exact(16)
            .map(|row| scrfd_hrx::Detection {
                score: row[0],
                bbox: row[1..5].try_into().unwrap(),
                landmarks: std::array::from_fn(|i| [row[5 + 2 * i], row[6 + 2 * i]]),
            })
            .collect();
        let mut embeddings = Vec::new();
        for tensor in &self.embeddings {
            embeddings.extend(
                floats(tensor)?
                    .chunks_exact(arcface_hrx::EMBEDDING)
                    .map(|row| <[f32; arcface_hrx::EMBEDDING]>::try_from(row).unwrap()),
            );
        }
        Ok(VisionOutput {
            descriptors,
            detections,
            embeddings,
        })
    }
}

impl ResidentVision {
    /// Whether the source slot can be reused or this prepared shape evicted.
    pub fn is_idle(&self) -> bool {
        self.first.is_idle()
    }
    /// Prepare for one shape. The ArcFace model must support batches of 32.
    /// All models must use `context`; recording rejects foreign bindings.
    pub fn new(
        context: &ModelContext,
        dino: &dinov3_hrx::DINOv3,
        detector: &scrfd_hrx::Scrfd,
        recognizer: Arc<arcface_hrx::ArcFace>,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        Self::prepare(
            context,
            Some(dino),
            detector,
            recognizer,
            width,
            height,
            Default::default(),
        )
    }

    /// Face-only variant, sharing the same resident gather/alignment path.
    pub fn faces(
        context: &ModelContext,
        detector: &scrfd_hrx::Scrfd,
        recognizer: Arc<arcface_hrx::ArcFace>,
        width: usize,
        height: usize,
        options: scrfd_hrx::DetectionOptions,
    ) -> Result<Self> {
        Self::prepare(context, None, detector, recognizer, width, height, options)
    }

    fn prepare(
        context: &ModelContext,
        dino: Option<&dinov3_hrx::DINOv3>,
        detector: &scrfd_hrx::Scrfd,
        recognizer: Arc<arcface_hrx::ArcFace>,
        width: usize,
        height: usize,
        options: scrfd_hrx::DetectionOptions,
    ) -> Result<Self> {
        ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
        ensure!(
            context.runtime().same_domain(detector.context().runtime())
                && context
                    .runtime()
                    .same_domain(recognizer.context().runtime())
                && dino
                    .is_none_or(|model| context.runtime().same_domain(model.context().runtime())),
            "all vision models must share the pipeline context"
        );
        let desc =
            TensorDesc::new(DType::U8, vec![1, height, width, 3])?.with_layout(Layout::Nhwc)?;
        let image = context.allocate_with(desc, MemoryPlacement::HostVisible)?;
        let mut graph = context.runtime().graph();
        let mut outputs = vec![image.clone()];
        if let Some(dino) = dino {
            let images = ImageOps::new(context, 1)?;
            let resize = RgbResize::letterbox(height, width, 224, 224, true)?;
            let [x, y, w, h] = resize.region;
            let mask: Vec<u8> = (0..14)
                .flat_map(|py| {
                    (0..14).map(move |px| {
                        let (cx, cy) = (px * 16 + 8, py * 16 + 8);
                        u8::from(cx >= x && cx < x + w && cy >= y && cy < y + h)
                    })
                })
                .collect();
            // Geometry is a shape constant, published once rather than every image.
            let mask_tensor = context.allocate_with(
                TensorDesc::new(DType::U8, vec![1, 196])?,
                MemoryPlacement::HostVisible,
            )?;
            mask_tensor
                .binding()
                .unwrap()
                .map_write()?
                .copy_from_slice(&mask);
            let pixels = images
                .resize_rgb_fragment(image.desc(), resize)?
                .record(&mut graph, std::slice::from_ref(&image))?
                .remove(0);
            let descriptors = dino.record_descriptors(&mut graph, &pixels, &mask_tensor)?;
            outputs.push(descriptors);
        }
        let candidates = detector.record_image(&mut graph, &image, options)?;
        let mut graph = Some(graph.prepare()?);
        let first = PreparedModel::prepare(context, 1, |_| {
            Ok(InferenceGraph {
                inputs: vec![image.clone()],
                outputs: outputs.clone(),
                graph: graph.take().expect("one slot"),
            })
        })?;
        Ok(Self {
            context: context.clone(),
            first,
            image,
            candidates,
            faces: PlanCache::new(4, PreparedModel::is_idle)?,
            recognizer,
            tensors: TensorOps::new(context, 1)?,
            shape: [width, height],
        })
    }

    fn prepare_faces(&self, count: usize) -> hrx::Result<Arc<PreparedModel>> {
        self.faces.get_or_prepare(count, || {
            let gather = self
                .tensors
                .gather_rows_fragment(self.candidates.rows().desc(), count)?;
            PreparedModel::prepare(&self.context, 1, |context| {
                let ids = context.allocate_with(
                    TensorDesc::new(DType::U32, vec![count])?,
                    MemoryPlacement::HostVisible,
                )?;
                let mut graph = context.runtime().graph();
                let rows = gather
                    .record(&mut graph, &[self.candidates.rows().clone(), ids.clone()])?
                    .remove(0);
                let mut outputs = vec![rows.clone()];
                // Every face is retained. Chunk CNN work without uploading,
                // copying slot inputs, or synchronizing between chunks.
                for start in (0..count).step_by(32) {
                    let batch = (count - start).min(32);
                    let points = rows.view(
                        start * 64 + 20,
                        TensorDesc::strided(DType::F32, vec![batch, 5, 2], vec![16, 2, 1])?,
                    )?;
                    let (embedding, status) = self
                        .recognizer
                        .record_image(&mut graph, &self.image, &points)
                        .map_err(|e| hrx::Error::Message(e.to_string()))?;
                    outputs.extend([embedding, status]);
                }
                Ok(InferenceGraph {
                    inputs: vec![ids],
                    outputs,
                    graph: graph.prepare()?,
                })
            })
        })
    }

    /// Publish RGB once; execute both model branches, CPU selection, and the
    /// directly bound ArcFace graph. Only compact control data is read here.
    /// The synchronous geometry validation also drains all image readers before
    /// releasing the input lease, including on errors.
    pub fn analyze(&self, rgb: &[u8]) -> Result<ResidentOutput> {
        let first = self.first.try_acquire()?.submit_host(&[rgb])?;
        let indices = self
            .candidates
            .select(first.completion(), &[self.shape])?
            .remove(0);
        let input = first.outputs()[0].clone();
        let descriptors = first.outputs().get(1).cloned();
        if indices.is_empty() {
            return Ok(ResidentOutput {
                descriptors,
                rows: None,
                embeddings: vec![],
                _input: input,
            });
        }
        let ids: Vec<u8> = indices
            .iter()
            .flat_map(|&i| (i as u32).to_le_bytes())
            .collect();
        let face_plan = self.prepare_faces(indices.len())?;
        let faces: Inference = face_plan.try_acquire()?.submit_host(&[&ids])?;
        // Wait before dropping `first`: its image/candidates are captured by
        // the second graph, while returned outputs own independent slot leases.
        let outputs = faces.wait()?;
        for status in outputs[2..].iter().step_by(2) {
            let binding = status.binding().unwrap();
            ensure!(
                binding
                    .map_read()?
                    .chunks_exact(4)
                    .all(|v| v == [0, 0, 0, 0]),
                "invalid or degenerate alignment landmarks"
            );
        }
        Ok(ResidentOutput {
            descriptors,
            rows: Some(outputs[0].clone()),
            embeddings: outputs[1..].iter().step_by(2).cloned().collect(),
            _input: input,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151 and cached SCRFD/ArcFace models"]
    fn face_chunks_preserve_every_row_and_retained_outputs() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        let detector = scrfd_hrx::Scrfd::load_in(scrfd_hrx::hub::weights(true)?, &context, 1)?;
        let arc = Arc::new(arcface_hrx::ArcFace::load_in(
            arcface_hrx::hub::weights(true)?,
            &context,
            32,
        )?);
        let photo = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../scrfd-hrx/tests/fixtures/t1.png"
        ))?
        .into_rgb8();
        let plan = ResidentVision::faces(
            &context,
            &detector,
            arc.clone(),
            photo.width() as usize,
            photo.height() as usize,
            Default::default(),
        )?;
        let detections = detector.detect(
            scrfd_hrx::Image {
                rgb: &photo,
                width: photo.width() as usize,
                height: photo.height() as usize,
            },
            Default::default(),
        )?;
        assert!(!detections.is_empty());
        let first = plan.first.try_acquire()?.submit_host(&[&photo])?;
        let indices = plan
            .candidates
            .select(first.completion(), &[plan.shape])?
            .remove(0);
        // Duplicate/permuted selected indices exercise graph chunk boundaries
        // independently of the fixture's actual face count.
        for count in [1, 31, 32, 33] {
            let order = (0..count)
                .map(|i| (i * 5 + 1) % indices.len())
                .collect::<Vec<_>>();
            let ids: Vec<u8> = order
                .iter()
                .flat_map(|&i| (indices[i] as u32).to_le_bytes())
                .collect();
            let points = order
                .iter()
                .map(|&i| detections[i].landmarks)
                .collect::<Vec<_>>();
            let expected = arc.embed(
                &photo,
                photo.width() as usize,
                photo.height() as usize,
                &points,
            )?;
            let faces = plan.prepare_faces(count)?;
            let output = faces.try_acquire()?.submit_host(&[&ids])?.wait()?;
            assert!(
                faces.try_acquire().is_err(),
                "output lease must prevent overwrite"
            );
            let result = ResidentOutput {
                descriptors: None,
                rows: Some(output[0].clone()),
                embeddings: output[1..].iter().step_by(2).cloned().collect(),
                _input: first.outputs()[0].clone(),
            }
            .readback()?;
            assert_eq!(result.embeddings, expected);
            assert_eq!(result.detections.len(), count);
            for (actual, &i) in result.detections.iter().zip(&order) {
                assert_eq!(actual.landmarks, detections[i].landmarks);
                assert_eq!(actual.bbox, detections[i].bbox);
            }
            drop(output);
            let before = context.runtime().statistics();
            let replay = faces.try_acquire()?.submit_host(&[&ids])?.wait()?;
            let after = context.runtime().statistics();
            assert_eq!(after.copied_bytes, before.copied_bytes);
            assert_eq!(after.allocations, before.allocations);
            assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
            assert_eq!(after.submissions - before.submissions, 1);
            for status in replay[2..].iter().step_by(2) {
                assert!(
                    status
                        .binding()
                        .unwrap()
                        .map_read()?
                        .iter()
                        .all(|&v| v == 0)
                );
            }
        }
        Ok(())
    }
}
