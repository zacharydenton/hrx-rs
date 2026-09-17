use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{CroppedImageMut, Image, ImageRef},
};
use hrx::{benchmark::Distribution, inference::ModelContext};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum PipelineMode {
    #[default]
    Resident,
    Host,
}

#[derive(Parser)]
#[command(about = "Single-image DINOv3 → SCRFD → ArcFace end-to-end benchmark")]
struct Args {
    /// JPEG or PNG containing at least one detectable face.
    image: PathBuf,
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..))]
    samples: u32,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
    warmup: u32,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..))]
    device: i32,
    /// Resolve model weights from cache only. Use HRX_OFFLINE=1 for native artifacts too.
    #[arg(long)]
    offline: bool,
    /// Permit skipping ArcFace when SCRFD finds no faces (reported explicitly).
    #[arg(long)]
    allow_no_faces: bool,
    /// Emit the complete report, samples and outputs as JSON instead of a table.
    #[arg(long)]
    json: bool,
    /// Resident composed graphs (default), or the separate host API baseline.
    #[arg(long, value_enum, default_value_t = PipelineMode::Resident)]
    pipeline: PipelineMode,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Face {
    bbox: [f32; 4],
    score: f32,
    landmarks: [[f32; 2]; 5],
    embedding: Vec<f32>,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Outputs {
    dimensions: [u32; 2],
    descriptors: Vec<f32>,
    faces: Vec<Face>,
}
#[derive(Serialize)]
struct Run {
    milliseconds: BTreeMap<&'static str, f64>,
    allocations: u64,
    submissions: u64,
    native_graphs_prepared: u64,
    copied_bytes: u64,
}
struct Pipeline {
    context: ModelContext,
    dino: dinov3_hrx::DINOv3,
    detector: scrfd_hrx::Scrfd,
    recognizer: Arc<arcface_hrx::ArcFace>,
    resizer: Resizer,
    resident: Option<hrx_vision::ResidentVision>,
    mode: PipelineMode,
}

fn timed<T>(
    times: &mut BTreeMap<&'static str, f64>,
    name: &'static str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let start = Instant::now();
    let result = f().with_context(|| format!("benchmark stage {name}"))?;
    times.insert(name, start.elapsed().as_secs_f64() * 1000.);
    Ok(result)
}

/// Centered DINO letterbox, with patch-center masking of its black padding.
fn dino_input(
    rgb: &[u8],
    width: u32,
    height: u32,
    resizer: &mut Resizer,
) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(width > 0 && height > 0, "empty image");
    let (w, h) = if width >= height {
        (224, (224u64 * height as u64 / width as u64).max(1) as u32)
    } else {
        ((224u64 * width as u64 / height as u64).max(1) as u32, 224)
    };
    let (x, y) = ((224 - w) / 2, (224 - h) / 2);
    let source = ImageRef::new(width, height, rgb, PixelType::U8x3)?;
    let mut pixels = vec![0; 224 * 224 * 3];
    let mut image = Image::from_slice_u8(224, 224, &mut pixels, PixelType::U8x3)?;
    let mut crop = CroppedImageMut::new(&mut image, x, y, w, h)?;
    resizer.resize(
        &source,
        &mut crop,
        &ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear)),
    )?;
    let mask = (0..14)
        .flat_map(|py| {
            (0..14).map(move |px| {
                let (cx, cy) = (px * 16 + 8, py * 16 + 8);
                u8::from(cx >= x && cx < x + w && cy >= y && cy < y + h)
            })
        })
        .collect();
    Ok((pixels, mask))
}

impl Pipeline {
    fn faces(
        &self,
        image: &image::RgbImage,
        times: &mut BTreeMap<&'static str, f64>,
    ) -> Result<Vec<Face>> {
        let (width, height) = image.dimensions();
        let detections = timed(times, "scrfd", || {
            self.detector.detect(
                scrfd_hrx::Image {
                    rgb: image,
                    width: width as usize,
                    height: height as usize,
                },
                Default::default(),
            )
        })?;
        let landmarks = detections.iter().map(|d| d.landmarks).collect::<Vec<_>>();
        let embeddings = timed(times, "arcface", || {
            self.recognizer
                .embed(image, width as usize, height as usize, &landmarks)
        })?;
        ensure!(
            embeddings.len() == detections.len()
                && embeddings.iter().flatten().all(|v| v.is_finite()),
            "invalid face embeddings"
        );
        Ok(detections
            .into_iter()
            .zip(embeddings)
            .map(|(d, embedding)| Face {
                bbox: d.bbox,
                score: d.score,
                landmarks: d.landmarks,
                embedding: embedding.to_vec(),
            })
            .collect())
    }
    fn run(&mut self, encoded: &[u8]) -> Result<(Outputs, Run)> {
        let before = self.context.runtime().statistics();
        let start = Instant::now();
        let mut milliseconds = BTreeMap::new();
        let image = timed(&mut milliseconds, "decode", || {
            Ok(image::load_from_memory(encoded)?.into_rgb8())
        })?;
        let (width, height) = image.dimensions();
        let (descriptors, faces) = match self.mode {
            PipelineMode::Host => {
                let (pixels, mask) = timed(&mut milliseconds, "dino_preprocess", || {
                    dino_input(&image, width, height, &mut self.resizer)
                })?;
                let descriptors = timed(&mut milliseconds, "dino", || {
                    self.dino.describe_rgb(&pixels, &mask)
                })?;
                (descriptors, self.faces(&image, &mut milliseconds)?)
            }
            PipelineMode::Resident => timed(&mut milliseconds, "resident", || {
                if self.resident.is_none() {
                    self.resident = Some(hrx_vision::ResidentVision::new(
                        &self.context,
                        &self.dino,
                        &self.detector,
                        self.recognizer.clone(),
                        width as usize,
                        height as usize,
                    )?);
                }
                let output = self
                    .resident
                    .as_ref()
                    .unwrap()
                    .analyze(&image)?
                    .readback()?;
                ensure!(
                    output.detections.len() == output.embeddings.len(),
                    "face count mismatch"
                );
                let faces = output
                    .detections
                    .into_iter()
                    .zip(output.embeddings)
                    .map(|(d, embedding)| Face {
                        bbox: d.bbox,
                        score: d.score,
                        landmarks: d.landmarks,
                        embedding: embedding.to_vec(),
                    })
                    .collect();
                Ok((output.descriptors.expect("DINO enabled"), faces))
            })?,
        };
        milliseconds.insert("total", start.elapsed().as_secs_f64() * 1000.);
        let after = self.context.runtime().statistics();
        ensure!(
            descriptors.len() == 2 * dinov3_hrx::HIDDEN
                && descriptors.iter().all(|v| v.is_finite()),
            "invalid DINO descriptors"
        );
        ensure!(
            faces
                .iter()
                .all(|face| face.embedding.len() == arcface_hrx::EMBEDDING
                    && face.embedding.iter().all(|v| v.is_finite())),
            "invalid face embeddings"
        );
        Ok((
            Outputs {
                dimensions: [width, height],
                descriptors,
                faces,
            },
            Run {
                milliseconds,
                allocations: after.allocations - before.allocations,
                submissions: after.submissions - before.submissions,
                native_graphs_prepared: after.native_graphs_prepared
                    - before.native_graphs_prepared,
                copied_bytes: after.copied_bytes - before.copied_bytes,
            },
        ))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let encoded =
        std::fs::read(&args.image).with_context(|| format!("reading {}", args.image.display()))?;
    // Validate the input before resolving or loading any weights.
    image::load_from_memory(&encoded).context("decoding benchmark image")?;
    let mut setup_ms = BTreeMap::new();
    let paths = timed(&mut setup_ms, "resolve_weights", || {
        Ok([
            dinov3_hrx::hub::weights(args.offline)?,
            scrfd_hrx::hub::weights(args.offline)?,
            arcface_hrx::hub::weights(args.offline)?,
        ])
    })?;
    let context = ModelContext::new(hrx::execution::RuntimeOptions {
        gpu_index: args.device,
        ..Default::default()
    })?;
    let target = context.runtime().gpu()?.target().as_str().to_owned();
    let dino = timed(&mut setup_ms, "load_dino", || {
        dinov3_hrx::DINOv3::load_in(&paths[0], &context, 1)
    })?;
    let detector = timed(&mut setup_ms, "load_scrfd", || {
        scrfd_hrx::Scrfd::load_in(&paths[1], &context, 1)
    })?;
    let recognizer = timed(&mut setup_ms, "load_arcface", || {
        arcface_hrx::ArcFace::load_in(&paths[2], &context, 32)
    })?;
    let mut pipeline = Pipeline {
        context,
        dino,
        detector,
        recognizer: Arc::new(recognizer),
        resizer: Resizer::new(),
        resident: None,
        mode: args.pipeline,
    };
    let (reference, first_run) = pipeline.run(&encoded)?;
    ensure!(
        args.allow_no_faces || !reference.faces.is_empty(),
        "no faces detected: use an image containing a face, or --allow-no-faces to explicitly skip ArcFace"
    );
    for _ in 0..args.warmup {
        ensure!(
            pipeline.run(&encoded)?.0 == reference,
            "warmup output changed"
        );
    }
    let mut runs = Vec::new();
    for i in 0..args.samples {
        let (output, run) = pipeline.run(&encoded)?;
        ensure!(output == reference, "output changed in sample {i}");
        runs.push(run);
    }
    let distributions: BTreeMap<_, _> = first_run
        .milliseconds
        .keys()
        .map(|&name| {
            Ok((
                name,
                Distribution::from_samples(
                    runs.iter().map(|run| run.milliseconds[name]).collect(),
                )?,
            ))
        })
        .collect::<Result<_>>()?;
    let cpu_affinity = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .map(|cpus| cpus.trim().to_owned())
            })
        });
    let report = serde_json::json!({
        "schema_version": 2,
        "pipeline": args.pipeline,
        "image": args.image, "image_sha256": format!("{:x}", Sha256::digest(&encoded)),
        "device": args.device, "target": target,
        "cpu_affinity": cpu_affinity,
        "weights": {"dino": paths[0], "scrfd": paths[1], "arcface": paths[2]},
        "pinned_revisions": {"dino": dinov3_hrx::hub::REVISION, "scrfd": scrfd_hrx::hub::REVISION, "arcface": arcface_hrx::hub::REVISION},
        "samples": args.samples, "warmup": args.warmup, "setup_ms": setup_ms, "first_run": first_run,
        "stages": distributions, "runs": runs, "outputs": reference,
        "arcface_executed": !reference.faces.is_empty(),
        "runtime_after": pipeline.context.runtime().statistics(),
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "{}×{}, {} faces; {target}; {} warm samples",
            reference.dimensions[0],
            reference.dimensions[1],
            reference.faces.len(),
            args.samples
        );
        println!("{:<18} {:>12} {:>12}", "stage", "median ms", "p95 ms");
        for (name, d) in &distributions {
            println!("{name:<18} {:>12.3} {:>12.3}", d.median_ms, d.p95_ms);
        }
        if reference.faces.is_empty() {
            println!("ArcFace skipped: no detected faces.");
        }
        println!(
            "Setup and first-run preparation excluded; decode, preprocessing, transfers and output waits included."
        );
        println!("Use --json for raw samples, setup times, runtime counters and checked outputs.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151 and cached DINO/SCRFD/ArcFace models"]
    fn resident_pipeline_matches_host_and_replays_without_transfers() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        let dino = dinov3_hrx::DINOv3::load_in(dinov3_hrx::hub::weights(true)?, &context, 1)?;
        let detector = scrfd_hrx::Scrfd::load_in(scrfd_hrx::hub::weights(true)?, &context, 1)?;
        let recognizer = Arc::new(arcface_hrx::ArcFace::load_in(
            arcface_hrx::hub::weights(true)?,
            &context,
            32,
        )?);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../scrfd-hrx/tests/fixtures/t1.png");
        let photo = image::open(path)?.into_rgb8();
        let black = image::RgbImage::new(photo.width(), photo.height());
        let flipped = image::imageops::flip_horizontal(&photo);
        let resident = hrx_vision::ResidentVision::new(
            &context,
            &dino,
            &detector,
            recognizer.clone(),
            photo.width() as usize,
            photo.height() as usize,
        )?;
        for image in [&photo, &black, &flipped, &photo] {
            let (pixels, mask) =
                dino_input(image, image.width(), image.height(), &mut Resizer::new())?;
            let descriptor = dino.describe_rgb(&pixels, &mask)?;
            let detections = detector.detect(
                scrfd_hrx::Image {
                    rgb: image,
                    width: image.width() as usize,
                    height: image.height() as usize,
                },
                Default::default(),
            )?;
            let landmarks = detections.iter().map(|d| d.landmarks).collect::<Vec<_>>();
            let embeddings = recognizer.embed(
                image,
                image.width() as usize,
                image.height() as usize,
                &landmarks,
            )?;
            let held = resident.analyze(image)?;
            assert!(
                resident.analyze(image).is_err(),
                "retained descriptors must backpressure reuse"
            );
            let actual = held.readback()?;
            assert_eq!(actual.descriptors.as_ref().unwrap(), &descriptor);
            assert_eq!(
                serde_json::to_value(&actual.detections)?,
                serde_json::to_value(&detections)?
            );
            assert_eq!(actual.embeddings, embeddings);
            let before = context.runtime().statistics();
            let again = resident.analyze(image)?.readback()?;
            let after = context.runtime().statistics();
            assert_eq!(again.descriptors, actual.descriptors);
            assert_eq!(again.embeddings, actual.embeddings);
            assert_eq!(after.allocations, before.allocations);
            assert_eq!(after.copied_bytes, before.copied_bytes);
            assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
            assert_eq!(
                after.submissions - before.submissions,
                if detections.is_empty() { 1 } else { 2 }
            );
        }
        Ok(())
    }
    #[test]
    fn letterbox_masks_patch_centers_and_preserves_rgb() -> Result<()> {
        for (width, height, kept) in [(224, 224, 196), (224, 112, 98), (112, 224, 98)] {
            let rgb = [17, 83, 211].repeat((width * height) as usize);
            let (pixels, mask) = dino_input(&rgb, width, height, &mut Resizer::new())?;
            ensure!(pixels.len() == 224 * 224 * 3 && mask.len() == 196);
            ensure!(mask.iter().map(|&v| v as usize).sum::<usize>() == kept);
            for (p, &valid) in mask.iter().enumerate() {
                let offset = ((p / 14 * 16 + 8) * 224 + p % 14 * 16 + 8) * 3;
                ensure!(
                    &pixels[offset..offset + 3]
                        == if valid == 1 {
                            &[17, 83, 211]
                        } else {
                            &[0, 0, 0]
                        }
                );
            }
        }
        Ok(())
    }
    #[test]
    fn zero_dimensions_and_bad_buffers_are_rejected() {
        assert!(dino_input(&[], 0, 1, &mut Resizer::new()).is_err());
        assert!(dino_input(&[], 1, 1, &mut Resizer::new()).is_err());
    }
    #[test]
    fn cli_rejects_empty_sample_sets() {
        assert!(Args::try_parse_from(["bench", "image.png", "--samples", "0"]).is_err());
        assert!(Args::try_parse_from(["bench", "image.png", "--warmup", "0"]).is_err());
    }

    #[test]
    #[ignore = "requires gfx1151, cached models and the sibling SCRFD image fixture"]
    fn single_image_dino_descriptors_match_host_pooling() -> Result<()> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../scrfd-hrx/tests/fixtures/t1.png");
        let image = image::open(path)?.into_rgb8();
        let (rgb, mask) = dino_input(&image, image.width(), image.height(), &mut Resizer::new())?;
        let model = dinov3_hrx::DINOv3::load(
            dinov3_hrx::hub::weights(true)?,
            dinov3_hrx::Options {
                device: 0,
                max_batch: 1,
            },
        )?;
        let mut normalized = vec![0.; rgb.len()];
        for c in 0..3 {
            for p in 0..224 * 224 {
                normalized[c * 224 * 224 + p] = (rgb[p * 3 + c] as f32 / 255.
                    - [0.485, 0.456, 0.406][c])
                    / [0.229, 0.224, 0.225][c];
            }
        }
        let tokens = model.forward(&normalized)?;
        let mut expected = tokens[..384].to_vec();
        let mut mean = vec![0f32; 384];
        let mut kept = 0;
        for (p, &valid) in mask.iter().enumerate() {
            if valid != 0 {
                kept += 1;
                for c in 0..384 {
                    mean[c] += tokens[(p + 5) * 384 + c];
                }
            }
        }
        if kept > 0 {
            for v in &mut mean {
                *v /= kept as f32;
            }
        }
        expected.extend(mean);
        for row in expected.chunks_mut(384) {
            let norm = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() as f32;
            if norm > 0. {
                for value in row {
                    *value /= norm;
                }
            }
        }
        let actual = model.describe_rgb(&rgb, &mask)?;
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(&a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        ensure!(
            max_error < 2e-4,
            "single-image descriptor error: {max_error}"
        );
        Ok(())
    }
}
