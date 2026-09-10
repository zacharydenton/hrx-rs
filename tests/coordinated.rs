//! Native tests for coordinated ownership and the pending kernel index.
use hrx::{
    Runtime, Stream,
    execution::{Access, GpuAccess, MemoryPlacement},
};

#[test]
#[ignore = "requires the GPU runtime, no NPU"]
fn host_visible_transfers_and_scoped_handoffs() -> hrx::Result<()> {
    let runtime = Runtime::new()?;
    let input = runtime.allocate(4096, MemoryPlacement::HostVisible)?;
    let output = runtime.allocate(4096, MemoryPlacement::HostVisible)?;
    let local = runtime.allocate(4096, MemoryPlacement::GpuLocal)?;
    input.map_write()?.fill(19);
    let mut graph = runtime.graph();
    graph.copy(local.view(), input.view())?;
    graph.copy(output.view(), local.view())?;
    let graph = graph.prepare()?;
    graph.submit()?.wait()?;
    assert!(output.map_read()?.iter().all(|b| *b == 19));

    let mut stream = Stream::open()?;
    let accesses = [GpuAccess {
        view: input.view(),
        access: Access::Write,
    }];
    let host = input.map_read()?;
    assert!(matches!(
        unsafe { runtime.with_gpu_access(&mut stream, &accesses, |_, _| Ok(())) },
        Err(hrx::Error::Busy(_))
    ));
    drop(host);
    let failed: hrx::Result<()> = unsafe {
        runtime.with_gpu_access(&mut stream, &accesses, |stream, views| {
            assert!(matches!(input.try_map_read(), Err(hrx::Error::Busy(_))));
            assert!(matches!(graph.submit(), Err(hrx::Error::Busy(_))));
            stream.fill(views[0], 27)?;
            Err(hrx::Error::Message("callback failed after enqueue".into()))
        })
    };
    assert!(failed.is_err());
    assert!(input.map_read()?.iter().all(|b| *b == 27));

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        runtime.with_gpu_access(&mut stream, &accesses, |stream, views| -> hrx::Result<()> {
            stream.fill(views[0], 31)?;
            panic!("callback panic after enqueue");
        })
    }));
    assert!(panic.is_err());
    assert!(input.map_read()?.iter().all(|b| *b == 31));
    let foreign = Runtime::new()?;
    assert!(unsafe { foreign.with_gpu_access(&mut stream, &accesses, |_, _| Ok(())) }.is_err());

    let owned = stream.allocate(4096)?;
    stream.fill(owned.binding(), 47)?;
    stream.synchronize()?;
    // No recorded work or raw aliases remain after this transfer.
    let adopted = unsafe { runtime.adopt_gpu_buffer(owned, MemoryPlacement::GpuLocal) }?;
    let mut copy = runtime.graph();
    copy.copy(output.view(), adopted.view())?;
    let copy = copy.prepare()?;
    drop(adopted);
    drop(stream);
    copy.submit()?.wait()?;
    assert!(output.map_read()?.iter().all(|b| *b == 47));
    Ok(())
}

#[cfg(feature = "loom")]
#[test]
#[ignore = "requires the GPU runtime and Loom compiler"]
fn keyed_pending_hits_preserve_batch_building() -> hrx::Result<()> {
    let stream = Stream::open()?;
    let compiler = hrx::loom::Compiler::resolve(None)?;
    let cache = hrx::loom::Kernels::new(compiler).keyed();
    let request = || {
        let mut spec = hrx::loom::Specialization::new("krea2_euler");
        spec.config.insert("krea2.euler.grid_x".into(), "1".into());
        spec.config.insert("krea2.euler.grid_y".into(), "1".into());
        Ok((include_str!("kernels/euler.loom"), spec))
    };
    let first = unsafe { cache.request_or_insert_with(&stream, 0, |_| request()) }?;
    let again = unsafe {
        cache.request_or_insert_with(
            &stream,
            0,
            |_| -> hrx::Result<(&str, hrx::loom::Specialization)> { panic!("hit called factory") },
        )
    }?;
    assert!(first.built().is_none());
    unsafe { cache.kernels().build(&stream) }?;
    assert!(first.built().is_some());
    assert!(again.built().is_some());
    unsafe {
        cache.get_or_insert_with(
            &stream,
            0,
            |_| -> hrx::Result<(&str, hrx::loom::Specialization)> {
                panic!("built hit called factory")
            },
        )
    }?;
    assert_eq!(cache.kernels().len(), 1);
    Ok(())
}

#[cfg(feature = "loom")]
#[test]
#[ignore = "requires the GPU runtime and Loom compiler"]
fn direct_requests_publish_once_to_the_pending_batch() -> hrx::Result<()> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let stream = Stream::open()?;
    let reports = Arc::new(AtomicUsize::new(0));
    let reported = reports.clone();
    let cache = hrx::loom::Kernels::new(hrx::loom::Compiler::resolve(None)?).reporting(move |_| {
        reported.fetch_add(1, Ordering::Relaxed);
    });
    let mut spec = hrx::loom::Specialization::new("krea2_euler");
    spec.config.insert("krea2.euler.grid_x".into(), "1".into());
    spec.config.insert("krea2.euler.grid_y".into(), "1".into());
    let source = include_str!("kernels/euler.loom");
    let pending = cache.request(source, &spec)?;
    unsafe { cache.get(&stream, source, &spec) }?;
    unsafe { cache.build(&stream) }?;
    assert!(pending.built().is_some());
    assert_eq!(reports.load(Ordering::Relaxed), 1);
    assert_eq!(cache.len(), 1);
    Ok(())
}
