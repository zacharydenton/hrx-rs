//! Run explicitly to check short-job progress under sustained native GPU work.
use hrx::{
    Result,
    execution::{MemoryPlacement, NativeSession, Runtime},
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires native GPU"]
fn short_graphs_complete_during_a_long_native_stage() -> Result<()> {
    let runtime = Runtime::new()?;
    let mut stream = runtime.stream(None)?;
    let source = stream.allocate(8 << 20)?;
    let destination = stream.allocate(8 << 20)?;
    stream.fill(source.binding(), 0x3c)?;
    stream.copy(destination.binding(), source.binding())?;
    stream.synchronize()?;
    let output = runtime.allocate(64, MemoryPlacement::HostVisible)?;
    let mut graph = runtime.graph();
    graph.fill(output.view(), 0x5a)?;
    let graph = graph.prepare()?;
    graph.submit()?.wait()?;
    // SAFETY: the session owns disjoint private buffers and their only stream;
    // initialization is drained, and its fence covers every native submission.
    let mut native = unsafe {
        NativeSession::new(&runtime, (stream, source, destination), |state| {
            state.0.synchronize()
        })
    };
    let stop = AtomicBool::new(false);
    let (started, starting) = mpsc::channel();
    let mut latencies = Vec::new();
    let mut completed = true;
    std::thread::scope(|scope| -> Result<()> {
        let worker = scope.spawn(|| -> Result<usize> {
            // SAFETY: only the session's private buffers are used by this stage.
            unsafe {
                native.run_cooperative(
                    |(stream, source, destination), cooperate| -> Result<usize> {
                        let deadline = Instant::now() + Duration::from_secs(10);
                        let mut batches = 0;
                        while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
                            for _ in 0..4 {
                                stream.copy(destination.binding(), source.binding())?;
                            }
                            cooperate.checkpoint(|| stream.synchronize())?;
                            batches += 1;
                            if batches == 1 {
                                started.send(()).unwrap();
                            }
                        }
                        assert!(
                            stop.load(Ordering::Acquire),
                            "native load reached its deadline before short jobs finished"
                        );
                        let mut bytes = [0; 64];
                        stream.read_blocking(destination.slice(0, 64), &mut bytes)?;
                        assert_eq!(bytes, [0x3c; 64]);
                        Ok(batches)
                    },
                )
            }?
        });
        starting.recv_timeout(Duration::from_secs(5)).unwrap();
        let result = (|| -> Result<()> {
            for _ in 0..20 {
                let start = Instant::now();
                let done = graph.submit()?;
                if !done.wait_timeout(Duration::from_secs(2))? {
                    completed = false;
                    break;
                }
                latencies.push(start.elapsed());
                assert_eq!(&*output.map_read()?, &[0x5a; 64]);
            }
            Ok(())
        })();
        stop.store(true, Ordering::Release);
        assert!(worker.join().unwrap()? > 0);
        result
    })?;
    assert!(completed, "short graph waited for the native stage to end");
    eprintln!(
        "{} short graphs during native copies: max latency {:?}",
        latencies.len(),
        latencies.iter().max().unwrap()
    );
    Ok(())
}
