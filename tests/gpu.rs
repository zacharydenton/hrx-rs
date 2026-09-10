//! Explicit hardware suite: cargo test --all-features --test gpu -- --ignored
use hrx::{Constants, Device, Stream};

/// Counts heap allocations while armed, so a test can assert that a hot path
/// stays on the stack instead of only intending to.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    thread_local! { static ARMED: Cell<bool> = const { Cell::new(false) }; }
    pub static COUNT: AtomicUsize = AtomicUsize::new(0);
    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ARMED.try_with(|armed| armed.get()).unwrap_or(false) {
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    /// Allocations performed by `body`.
    pub fn count(body: impl FnOnce()) -> usize {
        COUNT.store(0, Ordering::Relaxed);
        ARMED.with(|armed| armed.set(true));
        body();
        ARMED.with(|armed| armed.set(false));
        COUNT.load(Ordering::Relaxed)
    }
}
#[global_allocator]
static ALLOCATOR: counting::Counting = counting::Counting;

#[test]
#[ignore = "requires gfx1151"]
fn nested_views_bound_stream_and_graph_operations() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let source = stream.allocate(64)?;
    let destination = stream.allocate(64)?;
    let src = source.binding().slice(8, 32)?.slice(4, 16)?;
    let dst = destination.binding().slice(24, 24)?.slice(8, 16)?;
    assert_eq!(src.offset(), 12);
    assert_eq!(dst.offset(), 32);
    assert_eq!(src.len(), 16);
    assert!(std::ptr::eq(src.owner(), &source));
    assert_eq!(src.slice(16, 0)?.offset(), 28);
    assert!(src.slice(16, 0)?.is_empty());
    // These would fit in the allocation, but escape the parent view.
    assert!(src.slice(15, 2).is_err());
    assert!(src.slice(17, 0).is_err());
    assert!(src.slice(usize::MAX, 2).is_err());
    assert!(src.slice(1, usize::MAX).is_err());
    assert!(stream.upload(src, &[0; 17]).is_err());
    assert!(stream.upload_blocking(src, &[0; 17]).is_err());
    assert!(stream.copy(dst.slice(0, 15)?, src).is_err());
    let empty = src.slice(16, 0)?;
    assert!(stream.fill(empty, 1).is_err());
    assert!(stream.copy(empty, empty).is_err());
    stream.upload(empty, &[])?;

    stream.fill(source.binding(), 0x11)?;
    stream.fill(destination.binding(), 0x22)?;
    stream.fill(src, 0x33)?;
    stream.upload(src.slice(4, 4)?, &[1, 2, 3, 4])?;
    stream.upload_blocking(src.slice(8, 4)?, &[9, 10, 11, 12])?;
    stream.copy(dst, src)?;
    let mut expected = [0x11; 64];
    expected[12..28].fill(0x33);
    expected[16..20].copy_from_slice(&[1, 2, 3, 4]);
    expected[20..24].copy_from_slice(&[9, 10, 11, 12]);
    let mut actual = [0; 64];
    stream.read_blocking(source.binding(), &mut actual)?;
    assert_eq!(actual, expected);
    let mut copied = [0x22; 64];
    copied[32..48].copy_from_slice(&expected[12..28]);
    stream.read_blocking(destination.binding(), &mut actual)?;
    assert_eq!(actual, copied);

    let mut graph = stream.graph()?;
    assert!(graph.fill(&[], empty, 1).is_err());
    assert!(graph.copy(&[], empty, empty).is_err());
    assert!(graph.copy(&[], dst.slice(0, 15)?, src).is_err());
    // The copy reads what the fill writes, so the edge is declared, not implied.
    let filled = graph.fill(&[], src, 0x44)?;
    graph.copy(&[filled], dst, src)?;
    let mut exec = graph.finish()?;
    stream.launch(&mut exec)?;
    expected[12..28].fill(0x44);
    copied[32..48].fill(0x44);
    stream.read_blocking(source.binding(), &mut actual)?;
    assert_eq!(actual, expected);
    stream.read_blocking(destination.binding(), &mut actual)?;
    assert_eq!(actual, copied);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn independent_streams_move_between_threads() -> hrx::Result<()> {
    let device = hrx::Device::open(0)?;
    let streams = (0..4)
        .map(|_| device.stream())
        .collect::<hrx::Result<Vec<_>>>()?;
    let workers: Vec<_> = streams
        .into_iter()
        .enumerate()
        .map(|(index, mut stream)| {
            std::thread::spawn(move || -> hrx::Result<_> {
                let source = stream.allocate(4096)?;
                let destination = stream.allocate(4096)?;
                let expected = vec![index as u8 + 1; 4096];
                stream.upload(source.binding(), &expected)?;
                stream.copy(destination.binding(), source.binding())?;
                drop(source);
                let readback = stream.read(destination.binding())?;
                drop(destination);
                // Move pending commands and their retained storage back to the caller.
                Ok((stream, readback, expected))
            })
        })
        .collect();
    for worker in workers {
        let (mut stream, readback, expected) = worker.join().expect("stream worker panicked")?;
        assert_eq!(readback.wait(&mut stream)?, expected);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn copy_and_compute_streams_exchange_event_ordered_buffers() -> hrx::Result<()> {
    let device = hrx::Device::open(0)?;
    let mut upload = device.stream()?;
    let mut compute = device.stream()?;
    let source = upload.allocate(4096)?;
    // All reads and overwrites below are ordered in both directions by events.
    let result = compute.allocate(4096)?;
    let mut readbacks = Vec::new();
    let mut last = None;
    for value in 1..=16 {
        upload.upload(source.binding(), &[value; 4096])?;
        let ready = upload.record_event()?;
        compute.wait_event(&ready)?;
        drop(ready); // The queued dependency owns its native semaphore.
        compute.copy(result.binding(), source.binding())?;
        let consumed = compute.record_event()?;
        upload.wait_event(&consumed)?;
        last = Some(consumed);
        readbacks.push(compute.read(result.binding())?);
    }
    last.as_ref().unwrap().synchronize()?;
    assert!(last.as_ref().unwrap().is_complete()?);
    for (i, readback) in readbacks.into_iter().enumerate() {
        assert_eq!(readback.wait(&mut compute)?, vec![(i + 1) as u8; 4096]);
    }
    drop(source);
    upload.synchronize()?;
    compute.synchronize()?;
    Ok(())
}
#[test]
#[ignore = "requires gfx1151 and prepared runtime bundle"]
fn queued_storage_views_and_replay() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let source = stream.allocate(4096)?;
    let result = stream.allocate(4096)?;
    let host: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    stream.upload(source.binding(), &host)?;
    drop(host);
    stream.copy(result.binding(), source.binding())?;
    let mut submission = stream.submit()?;
    let _ = submission.is_complete()?;
    // Completion tokens are optional. Forgetting one cannot free staging.
    #[allow(clippy::forget_non_drop)] // Protect the contract if Submission later gains Drop.
    std::mem::forget(submission);
    drop(source);
    let mut bytes = vec![0; 4096];
    stream.read_blocking(result.binding(), &mut bytes)?;
    assert_eq!(
        bytes,
        (0..4096).map(|i| (i % 251) as u8).collect::<Vec<_>>()
    );
    let readback = stream.read(result.binding())?;
    assert_eq!(readback.wait(&mut stream)?, bytes);
    let abandoned = stream.read(result.binding())?;
    drop(abandoned);
    stream.synchronize()?;
    assert!(result.try_slice(usize::MAX, 1).is_err());
    assert!(result.binding().slice(usize::MAX, 1).is_err());
    let other = stream.allocate(4096)?;
    let mut graph = stream.graph()?;
    let filled = graph.fill(&[], result.binding(), 0x3c)?;
    graph.copy(&[filled], other.binding(), result.binding())?;
    let mut exec = graph.finish()?;
    for _ in 0..4 {
        stream.launch(&mut exec)?;
        stream.read_blocking(other.binding(), &mut bytes)?;
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
    let artifact = module.compile(&request)?;
    #[cfg(feature = "runner")]
    let path = artifact.path().to_path_buf();
    let mut stream = Stream::open()?;
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    drop(artifact);
    drop(module);
    drop(compiler);
    let sample = stream.allocate(512)?;
    let velocity = stream.allocate(512)?;
    let ones: Vec<u8> = (0..256).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    stream.upload_blocking(sample.binding(), &ones)?;
    stream.upload_blocking(velocity.binding(), &ones)?;
    let mut constants = Constants::new();
    // This source has precisely (index, f32). Loom's index legalization is
    // reflected in its aggregate byte count, not a general width inference.
    match kernel.info().constant_byte_length {
        8 => constants.push(256u32)?,
        12 => constants.push(256u64)?,
        n => panic!("unexpected Euler constants: {n}"),
    }
    constants.push(0.5f32)?;
    unsafe {
        assert!(
            stream
                .dispatch(
                    &kernel,
                    [1; 3],
                    [128, 1, 1],
                    &constants,
                    &[sample.binding(), velocity.binding()],
                )
                .unwrap_err()
                .to_string()
                .contains("compiled size")
        );
        stream.dispatch(
            &kernel,
            [1; 3],
            [256, 1, 1],
            &constants,
            &[sample.binding(), velocity.binding()],
        )?;
    }
    let mut output = vec![0; 512];
    stream.read_blocking(sample.binding(), &mut output)?;
    assert!(
        output
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3fc0)
    );
    let mut graph = stream.graph()?;
    unsafe {
        assert!(
            graph
                .dispatch(
                    &[],
                    &kernel,
                    [1; 3],
                    [128, 1, 1],
                    &constants,
                    &[sample.binding(), velocity.binding()],
                )
                .unwrap_err()
                .to_string()
                .contains("compiled size")
        );
        let copied_constants = constants.clone();
        graph.dispatch(
            &[],
            &kernel,
            [1; 3],
            [256, 1, 1],
            &copied_constants,
            &[sample.binding(), velocity.binding()],
        )?;
    }
    let mut exec = graph.finish()?;
    stream.launch(&mut exec)?;
    stream.read_blocking(sample.binding(), &mut output)?;
    assert!(
        output
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x4000)
    );
    // Buffers are device-scoped: a second stream on the same device may dispatch
    // against them, ordered by an event rather than refused by an ownership check.
    let mut foreign = Stream::open()?;
    let ready = stream.record_event()?;
    foreign.wait_event(&ready)?;
    unsafe {
        foreign.dispatch(
            &kernel,
            [1; 3],
            [256, 1, 1],
            &constants,
            &[sample.binding(), velocity.binding()],
        )?;
        let mut graph = foreign.graph()?;
        graph.dispatch(
            &[],
            &kernel,
            [1; 3],
            [256, 1, 1],
            &constants,
            &[sample.binding(), velocity.binding()],
        )?;
        let mut recorded = graph.finish()?;
        foreign.launch(&mut recorded)?;
    }
    let done = foreign.record_event()?;
    stream.wait_event(&done)?;
    stream.read_blocking(sample.binding(), &mut output)?;
    // Two more half-steps of 0.5 on top of 0x4000 (2.0): 2.5, then 3.0.
    assert!(
        output
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x4040)
    );
    // A kernel is still refused on a device that did not load it; with one GPU
    // present the device check is exercised through Buffer, not Kernel.
    assert!(Device::open(1).is_err());
    #[cfg(feature = "runner")]
    {
        let scratch = tempfile::tempdir()?;
        let input = scratch.path().join("runner-input.bin");
        let velocity = scratch.path().join("runner-velocity.bin");
        std::fs::write(&input, &ones)?;
        std::fs::write(&velocity, &ones)?;
        let locks = tempfile::tempdir()?;
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_hrx"))
            .arg("run")
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
                .as_chunks::<2>()
                .0
                .iter()
                .all(|b| u16::from_le_bytes([b[0], b[1]]) == 0x3fc0)
        );
    }
    Ok(())
}
#[test]
#[ignore = "requires gfx1151"]
fn buffers_are_device_scoped_but_graphs_stay_stream_scoped() -> hrx::Result<()> {
    let mut a = Stream::open()?;
    let mut b = Stream::open()?;
    let source = a.allocate(16)?;
    let destination = b.allocate(16)?;

    // Every safe operation accepts an allocation from either stream, because the
    // allocator is asked for any queue and executables are device-scoped. Each
    // step below is ordered against the previous one by an event, so the reads
    // observe a defined value rather than a race.
    a.fill(source.binding(), 7)?;
    let filled = a.record_event()?;
    b.wait_event(&filled)?;

    let mut bytes = [0; 16];
    b.read_blocking(source.binding(), &mut bytes)?;
    assert_eq!(bytes, [7; 16], "a foreign stream reads the allocation");

    b.copy(destination.binding(), source.binding())?;
    b.upload(source.binding(), &[1; 16])?;
    b.upload_blocking(source.binding(), &[2; 16])?;
    b.fill(source.binding(), 3)?;
    let readback = b.read(source.binding())?;
    assert_eq!(readback.wait(&mut b)?, vec![3; 16]);
    let mut graph = b.graph()?;
    let filled = graph.fill(&[], source.binding(), 4)?;
    graph.copy(&[filled], destination.binding(), source.binding())?;
    drop(graph);

    // A recorded graph is not device-scoped: it names the stream it will replay
    // on, and native replay elsewhere would reorder against that queue.
    let mut graph = a.graph()?;
    graph.fill(&[], source.binding(), 9)?;
    let mut exec = graph.finish()?;
    assert!(
        b.launch(&mut exec)
            .unwrap_err()
            .to_string()
            .contains("another stream")
    );
    let done = b.record_event()?;
    a.wait_event(&done)?;
    a.launch(&mut exec)?;
    a.read_blocking(source.binding(), &mut bytes)?;
    assert_eq!(bytes, [9; 16]);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn instantiated_graphs_own_their_resources() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let buffer = stream.allocate(1024)?;
    let output = stream.allocate(1024)?;
    let mut graph = stream.graph()?;
    let filled = graph.fill(&[], buffer.binding(), 42)?;
    graph.copy(&[filled], output.binding(), buffer.binding())?;
    let mut exec = graph.finish()?;
    drop(buffer);
    stream.launch(&mut exec)?;
    let mut bytes = [0; 1024];
    stream.read_blocking(output.binding(), &mut bytes)?;
    assert_eq!(bytes, [42; 1024]);
    // Even an empty executable keeps the stream/device alive after escape.
    drop(output);
    drop(stream);
    drop(exec);
    Ok(())
}

/// Dropping a handle is safe with work pending everywhere else, so it must be
/// safe here. It was not: releasing an executable graph mid-replay freed native
/// structures the device was still reading, and the next unrelated submission
/// died with an AMDGPU memory access fault rather than an error.
#[test]
#[ignore = "requires gfx1151"]
fn a_graph_released_mid_replay_does_not_fault_later_work() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let buffer = stream.allocate(64 << 20)?;
    let mut graph = stream.graph()?;
    // A chain long enough that the replay is still running when the handle goes.
    let mut last = graph.fill(&[], buffer.binding(), 1)?;
    for value in 2..64u8 {
        last = graph.fill(&[last], buffer.binding(), value)?;
    }
    let mut exec = graph.finish()?;
    stream.launch(&mut exec)?;
    // Both go while the replay runs, as a session dropping its graph beside the
    // buffers that graph recorded.
    drop(exec);
    drop(buffer);

    // Unrelated work on the same stream must still complete.
    for _ in 0..8 {
        let other = stream.allocate(4 << 20)?;
        stream.fill(other.binding(), 7)?;
        let mut bytes = vec![0u8; 4 << 20];
        stream.read_blocking(other.binding(), &mut bytes)?;
        assert!(bytes.iter().all(|b| *b == 7));
    }
    Ok(())
}

/// Distinct streams are distinguishable, and a clone of a handle is not a
/// distinct stream. Pools that reuse allocations need this: reuse is ordered by
/// the queue, so a block may only go back to the stream it came from.
/// The cache every consumer of this crate wrote for itself: a loaded kernel per
/// artifact, built eagerly or in one deferred batch.
#[test]
#[ignore = "requires gfx1151 and Loom compiler"]
fn a_kernel_cache_loads_once_and_builds_a_batch_together() -> hrx::Result<()> {
    let source = include_str!("kernels/euler.loom");
    let spec = |grid: &str| {
        let mut request = hrx::loom::Specialization::new("krea2_euler");
        request
            .config
            .insert("krea2.euler.grid_x".into(), grid.into());
        request
            .config
            .insert("krea2.euler.grid_y".into(), "1".into());
        request
    };
    let mut stream = Stream::open()?;
    let kernels = hrx::loom::Kernels::new(hrx::loom::Compiler::shared(
        None,
        hrx::loom::CompilerOptions {
            target: stream.target().clone(),
            ..Default::default()
        },
    )?);
    assert!(kernels.is_empty());

    // Safety: a checked-in Loom source compiled by this crate's own compiler.
    let first = unsafe { kernels.get(&stream, source, &spec("1")) }?;
    let again = unsafe { kernels.get(&stream, source, &spec("1")) }?;
    assert_eq!(first.symbol(), again.symbol());
    assert_eq!(kernels.len(), 1, "the same specialization loads once");

    // Deferred: named now, compiled together on build.
    let two = kernels.request(source, &spec("2"))?;
    let four = kernels.request(source, &spec("4"))?;
    assert_eq!(kernels.len(), 1, "requesting builds nothing");
    assert!(two.get().is_err(), "and yields nothing until built");
    // Safety: as above.
    unsafe { kernels.build(&stream) }?;
    assert_eq!(two.get()?.symbol(), "krea2_euler");
    assert_eq!(four.get()?.symbol(), "krea2_euler");
    assert_eq!(kernels.len(), 3);

    // A second cache over the same compiler still gets its own kernels, and the
    // compiler itself is the same one, not a second resolve of the library.
    let shared = hrx::loom::Compiler::shared(
        None,
        hrx::loom::CompilerOptions {
            target: stream.target().clone(),
            ..Default::default()
        },
    )?;
    assert_eq!(shared.identity(), kernels.compiler().identity());

    stream.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn keyed_kernel_hits_skip_the_factory_and_allocate_nothing() -> hrx::Result<()> {
    let stream = Stream::open()?;
    let source = include_str!("kernels/euler.loom");
    let spec = || {
        let mut spec = hrx::loom::Specialization::new("krea2_euler");
        spec.config.insert("krea2.euler.grid_x".into(), "1".into());
        spec.config.insert("krea2.euler.grid_y".into(), "1".into());
        spec
    };
    let reports = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = reports.clone();
    let kernels = hrx::loom::Kernels::new(hrx::loom::Compiler::shared(
        None,
        hrx::loom::CompilerOptions {
            target: stream.target().clone(),
            ..Default::default()
        },
    )?)
    .reporting(move |_| {
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    let keyed = kernels.clone().keyed();
    // Failed factories must leave the key available for a retry.
    let error = unsafe {
        keyed.get_or_insert_with(&stream, 1u32, |_| {
            Err(hrx::Error::Message("temporary request failure".into()))
        })
    };
    assert!(error.is_err());
    // Safety: trusted test source. Each key consistently selects this spec.
    unsafe { keyed.get_or_insert_with(&stream, 1, |_| Ok((source, spec()))) }?;
    let allocations = counting::count(|| {
        for _ in 0..100 {
            let kernel = unsafe {
                keyed.get_or_insert_with(&stream, 1, |_| panic!("factory called on a hit"))
            }
            .unwrap();
            assert_eq!(kernel.symbol(), "krea2_euler");
        }
    });
    assert_eq!(allocations, 0, "a warm key lookup must not allocate");
    // Two caller keys for one artifact still load and report only once.
    unsafe { keyed.get_or_insert_with(&stream, 2, |_| Ok((source, spec()))) }?;
    assert_eq!(kernels.len(), 1);
    assert_eq!(reports.load(std::sync::atomic::Ordering::Relaxed), 1);
    // A compilation failure also leaves no alias that could turn into a hit.
    for _ in 0..2 {
        assert!(
            unsafe {
                keyed.get_or_insert_with(&stream, 3, |_| {
                    Ok((source, hrx::loom::Specialization::new("missing_export")))
                })
            }
            .is_err()
        );
    }
    assert_eq!(kernels.len(), 1);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn streams_are_identifiable_for_per_stream_state() -> hrx::Result<()> {
    let device = hrx::Device::open(0)?;
    let first = device.stream()?;
    let second = device.stream()?;
    assert_ne!(first.id(), second.id());
    assert_eq!(first.id(), first.id());
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn graphs_fork_join_and_reject_foreign_nodes() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let left = stream.allocate(1024)?;
    let right = stream.allocate(1024)?;
    let merged = stream.allocate(2048)?;
    let mut graph = stream.graph()?;

    // Two independent fills; neither names the other, so the runtime may overlap
    // them. The copies that read them are ordered against the right producer only.
    let fill_left = graph.fill(&[], left.binding(), 0xa1)?;
    let fill_right = graph.fill(&[], right.binding(), 0xb2)?;
    assert_ne!(
        fill_left, fill_right,
        "distinct nodes have distinct handles"
    );

    // Naming a node twice is collapsed, not rejected.
    let ready = graph.join(&[fill_left, fill_right, fill_left])?;
    graph.copy(&[ready], merged.binding().slice(0, 1024)?, left.binding())?;
    graph.copy(
        &[ready],
        merged.binding().slice(1024, 1024)?,
        right.binding(),
    )?;

    // A node from another recording names nothing here.
    let mut other = stream.graph()?;
    let foreign = other.fill(&[], left.binding(), 0)?;
    assert!(
        graph
            .fill(&[foreign], right.binding(), 0)
            .unwrap_err()
            .to_string()
            .contains("another graph")
    );
    drop(other);

    let mut exec = graph.finish()?;
    stream.launch(&mut exec)?;
    let mut bytes = vec![0; 2048];
    stream.read_blocking(merged.binding(), &mut bytes)?;
    assert!(bytes[..1024].iter().all(|b| *b == 0xa1), "left branch ran");
    assert!(bytes[1024..].iter().all(|b| *b == 0xb2), "right branch ran");
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn small_fan_in_recording_does_not_allocate() -> hrx::Result<()> {
    let mut stream = Stream::open()?;
    let a = stream.allocate(1024)?;
    let b = stream.allocate(1024)?;
    let mut graph = stream.graph()?;
    let first = graph.fill(&[], a.binding(), 1)?;
    let second = graph.fill(&[], b.binding(), 2)?;
    // Warm the node vector so its growth is not attributed to dependency resolution.
    for _ in 0..8 {
        graph.fill(&[first], a.binding(), 3)?;
    }
    // A 16-entry dependency list is the documented stack capacity; the duplicate
    // forces the dedupe path, which is where the allocation used to live.
    let deps = [first, second, first];
    let mut recorded = Ok(first);
    let allocations = counting::count(|| {
        recorded = graph.fill(&deps, b.binding(), 4);
    });
    recorded?;
    assert_eq!(allocations, 0, "small fan-in must stay on the stack");
    let mut exec = graph.finish()?;
    stream.launch(&mut exec)?;
    let mut bytes = [0; 1024];
    stream.read_blocking(b.binding(), &mut bytes)?;
    assert_eq!(bytes, [4; 1024]);
    Ok(())
}
