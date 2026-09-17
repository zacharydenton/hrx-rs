use anyhow::Result;
use hrx::inference::ModelContext;
use hrx_vision::{AnalysisImage, BatchedVision};
use std::sync::Arc;

#[test]
#[ignore = "requires gfx1151 and pinned cached vision checkpoints"]
fn mixed_images_match_host_batch_and_replay_without_transfers() -> Result<()> {
    let context = ModelContext::new(Default::default())?;
    let batch = std::env::var("VISION_BATCH")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let dino = Arc::new(dinov3_hrx::DINOv3::load_in(
        dinov3_hrx::hub::weights(true)?,
        &context,
        batch,
    )?);
    let detector = Arc::new(scrfd_hrx::Scrfd::load_in(
        scrfd_hrx::hub::weights(true)?,
        &context,
        batch,
    )?);
    let arc = Arc::new(arcface_hrx::ArcFace::load_in(
        arcface_hrx::hub::weights(true)?,
        &context,
        32,
    )?);
    let pipeline = BatchedVision::new(
        &context,
        dino.clone(),
        detector.clone(),
        arc.clone(),
        batch,
        128 * 1024 * 1024,
    )?;
    let photo = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../scrfd-hrx/tests/fixtures/t1.png"
    ))?
    .into_rgb8();
    let pictures: Vec<_> = (0..batch)
        .map(|i| {
            if i % 3 == 1 {
                image::RgbImage::new(320, 480)
            } else {
                image::imageops::resize(
                    &photo,
                    640 - i as u32 * 4,
                    440 - i as u32 * 2,
                    image::imageops::FilterType::Triangle,
                )
            }
        })
        .collect();
    let canvases: Vec<_> = pictures
        .iter()
        .map(|image| {
            let mut canvas = image::RgbImage::new(640, 640);
            image::imageops::replace(&mut canvas, image, 0, 0);
            canvas
        })
        .collect();
    let planes: Vec<_> = pictures
        .iter()
        .map(|image| {
            image::imageops::resize(image, 224, 224, image::imageops::FilterType::Triangle)
        })
        .collect();
    let inputs: Vec<_> = pictures
        .iter()
        .zip(&canvases)
        .zip(&planes)
        .map(|((image, canvas), plane)| AnalysisImage {
            rgb: image.as_raw(),
            width: image.width() as usize,
            height: image.height() as usize,
            detector_rgb: canvas.as_raw(),
            detector_scale: 1.,
            descriptor_rgb: plane.as_raw(),
            descriptor_region: [0, 0, 224, 224],
        })
        .collect();
    let rgb: Vec<_> = planes
        .iter()
        .flat_map(|image| image.as_raw().iter().copied())
        .collect();
    let canvases: Vec<_> = canvases
        .iter()
        .flat_map(|image| image.as_raw().iter().copied())
        .collect();
    let shapes: Vec<_> = pictures
        .iter()
        .map(|image| [image.width() as usize, image.height() as usize])
        .collect();
    let host = || -> Result<_> {
        let descriptions = dino.describe_rgb(&rgb, &vec![1; batch * 196])?;
        let found = detector.detect_letterboxed(
            &canvases,
            &vec![1.; batch],
            &shapes,
            Default::default(),
        )?;
        let mut crops = Vec::new();
        for (image, faces) in pictures.iter().zip(&found) {
            for face in faces {
                crops.extend(arcface_hrx::alignment::crop(
                    image.as_raw(),
                    image.width() as usize,
                    image.height() as usize,
                    &face.landmarks,
                )?);
            }
        }
        Ok((descriptions, found, arc.embeddings(&crops)?))
    };
    let (expected_dino, expected_faces, expected_arc) = host()?;
    assert!(!expected_arc.is_empty());
    let first = pipeline.analyze(&inputs)?;
    for (input, resident) in inputs.iter().zip(&first.images) {
        if let Some(resident) = resident {
            let binding = resident.binding().unwrap();
            assert_eq!(&*binding.map_read()?, input.rgb);
        }
    }
    assert!(
        pipeline.analyze(&inputs).is_err(),
        "retained results must hold the input lease"
    );
    let retained_image = first.images.iter().flatten().next().unwrap().clone();
    let baseline = first.readback()?;
    assert!(
        pipeline.analyze(&inputs).is_err(),
        "resident source views must retain the face slot"
    );
    drop(retained_image);
    let mut offset = 0;
    for (i, result) in baseline.iter().enumerate() {
        assert_eq!(
            result.descriptors.as_ref().unwrap(),
            &expected_dino[i * 768..][..768]
        );
        assert_eq!(records(&result.detections), records(&expected_faces[i]));
        for (actual, expected) in result.embeddings.iter().zip(&expected_arc[offset..]) {
            let dot: f64 = actual
                .iter()
                .zip(expected)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let na: f64 = actual.iter().map(|&v| (v as f64).powi(2)).sum();
            let nb: f64 = expected.iter().map(|&v| (v as f64).powi(2)).sum();
            assert!(
                dot / (na * nb).sqrt() > 0.99999,
                "resident affine embedding drift"
            );
        }
        offset += result.embeddings.len();
    }
    // Publication validation does not poison a plan or leave its slot reserved.
    assert!(pipeline.analyze(&[]).is_err());
    let before = context.runtime().statistics();
    for _ in 0..3 {
        let actual = pipeline.analyze(&inputs)?.readback()?;
        for (a, b) in actual.iter().zip(&baseline) {
            assert_eq!(a.descriptors, b.descriptors);
            assert_eq!(records(&a.detections), records(&b.detections));
            assert_eq!(a.embeddings, b.embeddings);
        }
    }
    let after = context.runtime().statistics();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.copied_bytes, before.copied_bytes);
    assert_eq!(after.native_graphs_prepared, before.native_graphs_prepared);
    assert_eq!(after.submissions - before.submissions, 6);
    let black_source = vec![0u8; 3 * 24 * 16];
    let black_canvas = vec![0u8; 3 * 640 * 640];
    let black_plane = vec![0u8; 3 * 224 * 224];
    let black = AnalysisImage {
        rgb: &black_source,
        width: 24,
        height: 16,
        detector_rgb: &black_canvas,
        detector_scale: 1.,
        descriptor_rgb: &black_plane,
        descriptor_region: [0, 0, 224, 224],
    };
    let before_blank = context.runtime().statistics();
    let blank = pipeline.analyze(&vec![black; batch])?;
    assert!(blank.images.iter().all(Option::is_none));
    assert!(blank.embeddings.is_empty());
    assert!(blank.rows.is_none());
    assert!(
        blank
            .readback()?
            .iter()
            .all(|image| image.detections.is_empty())
    );
    let after_blank = context.runtime().statistics();
    assert_eq!(after_blank.submissions - before_blank.submissions, 1);
    assert_eq!(after_blank.allocations, before_blank.allocations);
    assert_eq!(
        after_blank.native_graphs_prepared,
        before_blank.native_graphs_prepared
    );
    // Changing image dimensions and returning from a no-face batch must not
    // expose stale rows, source pixels or padded embeddings.
    let again = pipeline.analyze(&inputs)?.readback()?;
    for (a, b) in again.iter().zip(&baseline) {
        assert_eq!(a.embeddings, b.embeddings);
    }
    // Optional alternating paired smoke measurements, with identical model
    // pixels and all final outputs read back; no decode/file IO is timed.
    if let Ok(samples) = std::env::var("VISION_SAMPLES") {
        let samples: usize = samples.parse()?;
        let mut host_ms = Vec::new();
        let mut resident_ms = Vec::new();
        for i in 0..samples {
            for resident in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let at = std::time::Instant::now();
                if resident {
                    std::hint::black_box(pipeline.analyze(&inputs)?.readback()?);
                } else {
                    std::hint::black_box(host()?);
                }
                let ms = at.elapsed().as_secs_f64() * 1000.;
                if resident {
                    resident_ms.push(ms);
                } else {
                    host_ms.push(ms);
                }
            }
        }
        host_ms.sort_by(f64::total_cmp);
        resident_ms.sort_by(f64::total_cmp);
        eprintln!("batch={batch} faces={offset} host_ms={host_ms:?} resident_ms={resident_ms:?}");
    }
    Ok(())
}

fn records(faces: &[scrfd_hrx::Detection]) -> Vec<(f32, [f32; 4], [[f32; 2]; 5])> {
    faces
        .iter()
        .map(|face| (face.score, face.bbox, face.landmarks))
        .collect()
}
