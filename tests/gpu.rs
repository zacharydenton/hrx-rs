#![cfg(feature = "loom")]
//! Explicit hardware suite: cargo test --all-features --test gpu -- --ignored
use hrx::{Constants, Stream};
#[test]
#[ignore = "requires gfx1151 and prepared runtime bundle"]
fn queued_storage_views_and_replay() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let source = stream.allocate(4096)?;
    let result = stream.allocate(4096)?;
    let host: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    stream.upload_queued(&source, 0, &host)?;
    drop(host);
    stream.copy(&result, 0, &source, 0, 4096)?;
    let mut submission = stream.submit()?;
    let _ = submission.is_complete()?;
    // Completion tokens are optional. Forgetting one cannot free staging.
    #[allow(clippy::forget_non_drop)] // Protect the contract if Submission later gains Drop.
    std::mem::forget(submission);
    drop(source);
    let mut bytes = vec![0; 4096];
    stream.read(result.binding(), &mut bytes)?;
    assert_eq!(
        bytes,
        (0..4096).map(|i| (i % 251) as u8).collect::<Vec<_>>()
    );
    let readback = stream.read_queued(result.binding())?;
    assert_eq!(readback.wait(&mut stream)?, bytes);
    let abandoned = stream.read_queued(result.binding())?;
    drop(abandoned);
    stream.synchronize()?;
    assert!(result.try_slice(usize::MAX, 1).is_err());
    assert!(stream.upload_queued(&result, usize::MAX, &[1]).is_err());
    let other = stream.allocate(4096)?;
    let mut sequence = stream.sequence()?;
    sequence
        .fill(result.binding(), 0x3c)?
        .copy(other.binding(), result.binding())?;
    let mut sequence = sequence.finish()?;
    for _ in 0..4 {
        stream.launch_sequence(&mut sequence)?;
        stream.read(other.binding(), &mut bytes)?;
        assert!(bytes.iter().all(|b| *b == 0x3c));
    }
    Ok(())
}
#[test]
#[ignore = "requires gfx1151 and Loom compiler"]
fn prepared_binding_kernel_and_graph_match() -> hrx::Result<()> {
    let compiler = hrx::loom::Compiler::resolve(None)?;
    let module = compiler.module(include_str!("kernels/euler.loom"));
    let mut request = hrx::loom::Specialization::new("krea2_euler");
    request
        .config
        .insert("krea2.euler.grid_x".into(), "1".into());
    request
        .config
        .insert("krea2.euler.grid_y".into(), "1".into());
    let cache = tempfile::tempdir()?;
    let artifact = module.compile(&request, cache.path())?;
    let path = artifact.path().to_path_buf();
    let mut stream = Stream::open()?;
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    drop(artifact);
    drop(module);
    drop(compiler);
    let sample = stream.allocate(512)?;
    let velocity = stream.allocate(512)?;
    let ones: Vec<u8> = (0..256).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    stream.upload(&sample, &ones)?;
    stream.upload(&velocity, &ones)?;
    let mut constants = Constants::new();
    // This source has precisely (index, f32). Loom's index legalization is
    // reflected in its aggregate byte count, not a general width inference.
    match kernel.info().constant_byte_length {
        8 => {
            constants.push(256u32)?;
        }
        12 => {
            constants.push(256u64)?;
        }
        n => panic!("unexpected Euler constants: {n}"),
    }
    constants.push(0.5f32)?;
    unsafe {
        stream.dispatch(
            &kernel,
            [1; 3],
            [256, 1, 1],
            &constants,
            &[sample.binding(), velocity.binding()],
        )?;
    }
    let mut output = vec![0; 512];
    stream.read(sample.binding(), &mut output)?;
    assert!(
        output
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3fc0)
    );
    let mut sequence = stream.sequence()?;
    unsafe {
        let copied_constants = constants.clone();
        sequence.dispatch(
            &kernel,
            [1; 3],
            [256, 1, 1],
            &copied_constants,
            &[sample.binding(), velocity.binding()],
        )?;
    }
    let mut sequence = sequence.finish()?;
    stream.launch_sequence(&mut sequence)?;
    stream.read(sample.binding(), &mut output)?;
    assert!(
        output
            .chunks_exact(2)
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x4000)
    );
    let mut foreign = Stream::open()?;
    unsafe {
        assert!(
            foreign
                .dispatch(
                    &kernel,
                    [1; 3],
                    [256, 1, 1],
                    &constants,
                    &[sample.binding(), velocity.binding()]
                )
                .is_err()
        );
        let mut graph = foreign.sequence()?;
        assert!(
            graph
                .dispatch(
                    &kernel,
                    [1; 3],
                    [256, 1, 1],
                    &constants,
                    &[sample.binding(), velocity.binding()]
                )
                .is_err()
        );
    }
    #[cfg(feature = "runner")]
    {
        let input = cache.path().join("runner-input.bin");
        let velocity = cache.path().join("runner-velocity.bin");
        std::fs::write(&input, &ones)?;
        std::fs::write(&velocity, &ones)?;
        let locks = tempfile::tempdir()?;
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_loomrun"))
            .arg("--hsaco")
            .arg(&path)
            .args(["--kernel", "krea2_euler", "--block", "256"])
            .args([
                if kernel.info().constant_byte_length == 8 {
                    "--i32"
                } else {
                    "--i64"
                },
                "256",
                "--f32",
                "0.5",
            ])
            .arg("--inout")
            .arg(&input)
            .arg("--in")
            .arg(&velocity)
            .env("XDG_RUNTIME_DIR", locks.path())
            .stdout(std::process::Stdio::null())
            .spawn()?;
        let pid = child.id();
        assert!(child.wait()?.success());
        assert!(!locks.path().join(format!("hrx-{pid}.lock")).exists());
        assert!(
            std::fs::read(input)?
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3fc0)
        );
    }
    #[cfg(feature = "compat")]
    {
        let device = hrx::compat::Device::open()?;
        let _scope = device.enter();
        let sample = device.allocate(512)?;
        let velocity = device.allocate(512)?;
        device.copy_from_host(sample.ptr(), &ones)?;
        device.copy_from_host(velocity.ptr(), &ones)?;
        let direct = unsafe { hrx::compat::Kernel::load(&path, "krea2_euler")? };
        let mut arguments = hrx::compat::Args::new();
        if kernel.info().constant_byte_length == 8 {
            arguments.u32(256);
        } else {
            arguments.i64(256);
        }
        arguments.f32(0.5).ptr(sample.ptr()).ptr(velocity.ptr());
        unsafe {
            direct.launch([1; 3], [256, 1, 1], &arguments)?;
        }
        drop(velocity);
        drop(direct);
        device.copy_to_host(&mut output, sample.ptr())?;
        assert!(
            output
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3fc0)
        );
    }
    Ok(())
}
#[cfg(feature = "compat")]
#[test]
#[ignore = "requires gfx1151"]
fn independent_compat_streams_retain_dropped_allocations() -> hrx::Result<()> {
    let mut workers = Vec::new();
    for id in 0..4u8 {
        workers.push(std::thread::spawn(move || -> hrx::Result<()> {
            let device = hrx::compat::Device::open()?;
            let _scope = device.enter();
            let source = device.allocate(1024)?;
            let output = device.allocate(1024)?;
            device.copy_from_host(source.ptr(), &vec![id; 1024])?;
            device.copy_device_to_device(output.ptr(), source.ptr(), 1024)?;
            drop(source);
            let mut readback = vec![0; 1024];
            device.copy_to_host(&mut readback, output.ptr())?;
            assert!(readback.iter().all(|b| *b == id));
            let nested = hrx::compat::Device::open()?;
            {
                let _inner = nested.enter();
                assert!(std::sync::Arc::ptr_eq(&nested, &hrx::compat::device()));
            }
            assert!(std::sync::Arc::ptr_eq(&device, &hrx::compat::device()));
            Ok(())
        }));
    }
    for worker in workers {
        worker.join().unwrap()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn buffers_reject_every_foreign_stream_access() -> hrx::Result<()> {
    let mut a = Stream::open()?;
    let mut b = Stream::open()?;
    let source = a.allocate(16)?;
    let destination = b.allocate(16)?;
    a.fill(&source, 7)?;
    assert!(b.upload(&source, &[1; 16]).is_err());
    assert!(b.read(source.binding(), &mut [0; 16]).is_err());
    assert!(b.fill(&source, 1).is_err());
    assert!(b.copy(&destination, 0, &source, 0, 16).is_err());
    assert!(b.copy(&source, 0, &destination, 0, 16).is_err());
    assert!(b.upload_queued(&source, 0, &[1; 16]).is_err());
    assert!(b.read_queued(source.binding()).is_err());
    let mut graph = b.sequence()?;
    assert!(graph.fill(source.binding(), 1).is_err());
    assert!(graph.copy(destination.binding(), source.binding()).is_err());
    drop(graph);
    let mut graph = a.sequence()?;
    graph.fill(source.binding(), 9)?;
    let mut graph = graph.finish()?;
    assert!(b.launch_sequence(&mut graph).is_err());
    a.launch_sequence(&mut graph)?;
    let mut bytes = [0; 16];
    a.read(source.binding(), &mut bytes)?;
    assert_eq!(bytes, [9; 16]);

    let a = hrx::Gpu::open()?;
    let b = hrx::Gpu::open()?;
    let buffer = a.alloc(16)?;
    assert!(b.h2d(&buffer, &[0; 16]).is_err());
    assert!(b.h2d_at(&buffer, 0, &[]).is_err());
    assert!(b.d2h(&buffer, &mut bytes).is_err());
    assert!(b.d2h_ref(buffer.binding(), &mut bytes).is_err());
    assert!(b.memset(&buffer, 0, 16).is_err());
    assert!(b.d2d(&buffer, &buffer, 16).is_err());
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn instantiated_sequences_own_their_resources() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let buffer = stream.allocate(1024)?;
    let output = stream.allocate(1024)?;
    let mut builder = stream.sequence()?;
    builder
        .fill(buffer.binding(), 42)?
        .copy(output.binding(), buffer.binding())?;
    let mut sequence = builder.finish()?;
    drop(buffer);
    stream.launch_sequence(&mut sequence)?;
    let mut bytes = [0; 1024];
    stream.read(output.binding(), &mut bytes)?;
    assert_eq!(bytes, [42; 1024]);
    // Even an empty executable keeps the stream/device alive after escape.
    drop(output);
    drop(stream);
    drop(sequence);
    Ok(())
}

#[cfg(feature = "compat")]
#[test]
#[ignore = "requires gfx1151"]
fn compatibility_addresses_reject_foreign_streams() -> hrx::Result<()> {
    let a = hrx::compat::Device::open()?;
    let b = hrx::compat::Device::open()?;
    let source = a.allocate(16)?;
    let destination = b.allocate(16)?;
    a.zero(source.ptr(), 16)?;
    assert!(b.copy_from_host(source.ptr(), &[1; 16]).is_err());
    assert!(b.copy_to_host(&mut [0; 16], source.ptr()).is_err());
    assert!(b.zero(source.ptr(), 16).is_err());
    assert!(
        b.copy_device_to_device(destination.ptr(), source.ptr(), 16)
            .is_err()
    );
    assert!(
        b.copy_device_to_device(source.ptr(), destination.ptr(), 16)
            .is_err()
    );
    a.synchronize()?;
    Ok(())
}
