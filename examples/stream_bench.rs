//! End-to-end stream costs on a prepared GPU. Run with --release --features loom.
use hrx::{Constants, Result, Stream};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

fn measure(
    stream: &mut Stream,
    samples: usize,
    operations: usize,
    mut run: impl FnMut(&mut Stream) -> Result<()>,
) -> Result<(f64, f64)> {
    let mut enqueue = Vec::new();
    let mut complete = Vec::new();
    for sample in 0..samples + 3 {
        stream.synchronize()?;
        let start = Instant::now();
        run(stream)?;
        let recorded = start.elapsed();
        stream.synchronize()?;
        let finished = start.elapsed();
        if sample >= 3 {
            enqueue.push(recorded.as_secs_f64() * 1e9 / operations as f64);
            complete.push(finished.as_secs_f64() * 1e9 / operations as f64);
        }
    }
    enqueue.sort_by(f64::total_cmp);
    complete.sort_by(f64::total_cmp);
    Ok((enqueue[samples / 2], complete[samples / 2]))
}

fn main() -> Result<()> {
    let mut stream = Stream::open()?;
    let mut metrics = BTreeMap::new();
    let samples = 9;
    let (ns, _) = measure(&mut stream, samples, 50_000, |stream| {
        for _ in 0..50_000 {
            let buffer = stream.scratch(1024 * 1024)?;
            black_box(&buffer);
            stream.recycle(buffer)?;
        }
        Ok(())
    })?;
    metrics.insert("scratch_reuse_ns", ns);
    let (ns, _) = measure(&mut stream, samples, 2_000, |stream| {
        for _ in 0..2_000 {
            black_box(stream.allocate(1024 * 1024)?);
        }
        Ok(())
    })?;
    metrics.insert("allocate_drop_ns", ns);

    let bytes = 256 * 1024 * 1024;
    let chunk = vec![0x3cu8; 4 * 1024 * 1024];
    let weights = stream.allocate(bytes)?;
    let (_, ns) = measure(&mut stream, samples, 1, |stream| {
        for offset in (0..bytes).step_by(chunk.len()) {
            stream.upload(weights.binding().slice(offset, chunk.len())?, &chunk)?;
        }
        Ok(())
    })?;
    metrics.insert(
        "queued_upload_gib_s",
        bytes as f64 / (1024f64.powi(3) * ns * 1e-9),
    );
    let mut check = [0u8; 16];
    stream.read_blocking(
        weights.try_slice(bytes - check.len(), check.len())?,
        &mut check,
    )?;
    assert_eq!(check, [0x3c; 16]);
    drop(weights);

    let compiler = hrx::loom::Compiler::with_options(
        None,
        hrx::loom::CompilerOptions {
            target: stream.target().clone(),
            ..Default::default()
        },
    )?;
    let module = compiler.module(include_str!("../tests/kernels/euler.loom"));
    let mut spec = hrx::loom::Specialization::new("krea2_euler");
    spec.config.insert("krea2.euler.grid_x".into(), "1".into());
    spec.config.insert("krea2.euler.grid_y".into(), "1".into());
    let cache = tempfile::tempdir()?;
    let artifact = module.compile(&spec, cache.path())?;
    // This repository owns the source; use its exact dimensions and scalar layout.
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    let sample = stream.allocate(512)?;
    let velocity = stream.allocate(512)?;
    let ones: Vec<u8> = (0..256).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    stream.upload_blocking(sample.binding(), &ones)?;
    stream.upload_blocking(velocity.binding(), &ones)?;
    let mut constants = Constants::new();
    match kernel.info().constant_byte_length {
        8 => constants.push(256u32)?,
        12 => constants.push(256u64)?,
        n => return Err(hrx::Error::Message(format!("unexpected constant size {n}"))),
    }
    constants.push(0f32)?;
    let bindings = [sample.binding(), velocity.binding()];
    let (enqueue, complete) = measure(&mut stream, samples, 2048, |stream| {
        for _ in 0..2048 {
            // The source accesses exactly the 256 bf16 elements in each buffer.
            unsafe {
                stream.dispatch(&kernel, [1; 3], [256, 1, 1], &constants, &bindings)?;
            }
        }
        Ok(())
    })?;
    metrics.insert("dispatch_enqueue_ns", enqueue);
    metrics.insert("dispatch_complete_ns", complete);

    let mut builder = stream.sequence()?;
    for _ in 0..32 {
        unsafe {
            builder.dispatch(&kernel, [1; 3], [256, 1, 1], &constants, &bindings)?;
        }
    }
    let mut sequence = builder.finish()?;
    let (enqueue, ns) = measure(&mut stream, samples, 256 * 32, |stream| {
        for _ in 0..256 {
            stream.launch_sequence(&mut sequence)?;
        }
        Ok(())
    })?;
    metrics.insert("graph_enqueue_ns_per_replay", enqueue * 32.0);
    metrics.insert("graph_complete_ns_per_kernel", ns);
    let mut output = vec![0; 512];
    stream.read_blocking(sample.binding(), &mut output)?;
    assert_eq!(output, ones);
    println!("{}", serde_json::to_string(&metrics)?);
    Ok(())
}
