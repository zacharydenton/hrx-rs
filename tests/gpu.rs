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
    let submission = stream.submit()?;
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
    let mut request = hrx::loom::Request::new(include_str!("kernels/euler.loom"), "krea2_euler");
    request
        .config
        .insert("krea2.euler.grid_x".into(), "1".into());
    request
        .config
        .insert("krea2.euler.grid_y".into(), "1".into());
    let cache = tempfile::tempdir()?;
    let path = compiler.compile(&request, cache.path())?;
    let mut stream = Stream::open()?;
    let kernel = unsafe { stream.load(&path, "krea2_euler")? };
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
        sequence.dispatch(
            &kernel,
            [1; 3],
            [256, 1, 1],
            &constants,
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
    #[cfg(feature = "compat")]
    {
        let device = hrx::compat::Device::open()?;
        let _scope = device.enter();
        let sample = device.allocate(512)?;
        let velocity = device.allocate(512)?;
        device.copy_from_host(sample.ptr(), &ones)?;
        device.copy_from_host(velocity.ptr(), &ones)?;
        let direct = hrx::compat::Kernel::load(&path, "krea2_euler")?;
        let mut arguments = hrx::compat::Args::new();
        if kernel.info().constant_byte_length == 8 {
            arguments.u32(256);
        } else {
            arguments.i64(256);
        }
        arguments.f32(0.5).ptr(sample.ptr()).ptr(velocity.ptr());
        direct.launch([1; 3], [256, 1, 1], &arguments)?;
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
