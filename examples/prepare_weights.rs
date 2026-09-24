//! Measure page preparation plus a first host read or completed device upload.
//! Use saved fresh processes in alternating order; this does not evict caches.
use hrx::artifacts::safetensors::FileView;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let upload = args.last().is_some_and(|arg| arg == "--upload");
    if upload {
        args.pop();
    }
    if args.len() != 3 || !matches!(args[2].as_str(), "none" | "advice" | "populate") {
        return Err("usage: prepare_weights FILE TENSOR none|advice|populate [--upload]".into());
    }
    // The command requires an immutable checkpoint for the mapping's lifetime.
    let file = unsafe { FileView::map(&args[0]) }?;
    let tensor = file.get(&args[1])?;
    let mut stream = if upload {
        Some(hrx::Stream::open()?)
    } else {
        None
    };
    let buffer = stream
        .as_ref()
        .map(|stream| stream.allocate(tensor.bytes.len()))
        .transpose()?;
    let start = Instant::now();
    let prepared = match args[2].as_str() {
        "populate" => format!("{:?}", file.prepare_bytes(tensor.bytes)?),
        "advice" => {
            file.will_need(tensor);
            "Advised".into()
        }
        _ => "None".into(),
    };
    let preparation_ms = start.elapsed().as_secs_f64() * 1000.;
    let checksum: u64;
    let total_ms;
    if let (Some(stream), Some(buffer)) = (&mut stream, &buffer) {
        stream.upload_blocking(buffer.binding(), tensor.bytes)?;
        total_ms = start.elapsed().as_secs_f64() * 1000.;
        // Validation is outside timing, after the upload fence has retired.
        let mut actual = vec![0; tensor.bytes.len()];
        stream.read_blocking(buffer.binding(), &mut actual)?;
        if actual != tensor.bytes {
            return Err("upload bytes differ from checkpoint".into());
        }
        checksum = actual.iter().map(|&v| u64::from(v)).sum();
    } else {
        checksum = tensor.bytes.iter().map(|&v| u64::from(v)).sum();
        total_ms = start.elapsed().as_secs_f64() * 1000.;
    }
    println!(
        "{}",
        serde_json::json!({"mode":args[2], "preparation":prepared,
        "bytes":tensor.bytes.len(), "preparation_ms":preparation_ms,
        "total_ms":total_ms, "checksum":checksum, "upload":upload,
        "allocation_policy":"preallocated destination; fresh process",
        "timed_scope":if upload { "page preparation plus completed upload" } else { "page preparation plus complete host read" }, "cache_state":"externally controlled"})
    );
    Ok(())
}
