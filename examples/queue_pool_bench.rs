//! Queue-pool throughput and latency on a prepared GPU. Run with --release.
//! Usage: queue_pool_bench QUEUES [SAMPLES], one fresh process per queue count.
use hrx::{Buffer, Device, Result, Stream, benchmark::Distribution};
use serde_json::json;
use std::{sync::Barrier, time::Instant};

const STREAMS: usize = 16;
const WARMUP: usize = 3;

struct Lane {
    stream: Stream,
    source: Buffer,
    destination: Buffer,
}
impl Lane {
    fn new(device: &Device, bytes: usize) -> Result<Self> {
        let mut stream = device.stream()?;
        let source = stream.allocate(bytes)?;
        let destination = stream.allocate(bytes)?;
        stream.fill(source.binding(), 0x3c)?;
        stream.copy(destination.binding(), source.binding())?;
        stream.synchronize()?;
        Ok(Self {
            stream,
            source,
            destination,
        })
    }
    fn enqueue(&self) -> Result<()> {
        self.stream
            .copy(self.destination.binding(), self.source.binding())
    }
    fn check(&mut self) -> Result<()> {
        let mut bytes = [0; 64];
        self.stream
            .read_blocking(self.destination.slice(0, 64), &mut bytes)?;
        assert_eq!(bytes, [0x3c; 64]);
        Ok(())
    }
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn throughput(lanes: &mut [Lane], samples: usize, batch: usize) -> Result<Distribution> {
    let mut timings = Vec::new();
    for round in 0..samples + WARMUP {
        let start = Instant::now();
        for _ in 0..batch {
            for lane in &*lanes {
                lane.enqueue()?;
            }
        }
        for lane in &mut *lanes {
            lane.stream.synchronize()?;
        }
        if round >= WARMUP {
            timings.push(milliseconds(start));
        }
    }
    Distribution::from_samples(timings)
}

fn concurrent(lanes: &mut [Lane], samples: usize, batch: usize) -> Result<Distribution> {
    let mut timings = Vec::new();
    let barrier = Barrier::new(lanes.len());
    for round in 0..samples + WARMUP {
        // Creation and joining are excluded from each worker's elapsed time.
        // The slowest worker determines the batch's throughput.
        let elapsed = std::thread::scope(|scope| {
            let workers: Vec<_> = lanes
                .iter_mut()
                .map(|lane| {
                    let barrier = &barrier;
                    scope.spawn(move || -> Result<f64> {
                        barrier.wait();
                        let start = Instant::now();
                        for _ in 0..batch {
                            lane.enqueue()?;
                        }
                        lane.stream.synchronize()?;
                        Ok(milliseconds(start))
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("benchmark worker panicked"))
                .collect::<Result<Vec<_>>>()
        })?;
        if round >= WARMUP {
            timings.push(elapsed.into_iter().fold(0.0, f64::max));
        }
    }
    Distribution::from_samples(timings)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let parse = |value: &str| value.parse::<usize>().ok().filter(|&n| n > 0);
    let queues = args.first().and_then(|arg| parse(arg));
    let samples = args.get(1).map_or(Some(15), |arg| parse(arg));
    let (queues, samples) = match (queues, samples, args.len()) {
        (Some(queues), Some(samples), 1..=2) => (queues, samples),
        _ => {
            return Err(hrx::Error::Message(
                "usage: queue_pool_bench QUEUES [SAMPLES] (positive integers)".into(),
            ));
        }
    };
    let device = Device::open(0)?;
    device.set_stream_queue_count(queues)?;
    let start = Instant::now();
    let streams = (0..STREAMS)
        .map(|_| device.stream())
        .collect::<Result<Vec<_>>>()?;
    let creation_ms = milliseconds(start);
    // Keep these streams alive so every case uses the same already-created queues.
    let mut small = (0..STREAMS)
        .map(|_| Lane::new(&device, 4096))
        .collect::<Result<Vec<_>>>()?;
    let small_batch = 1024;
    let small_times = throughput(&mut small, samples, small_batch)?;
    let threaded_times = concurrent(&mut small, samples, small_batch)?;
    let mut idle_latency = Vec::new();
    for round in 0..samples + WARMUP {
        for lane in &mut small {
            let start = Instant::now();
            lane.enqueue()?;
            lane.stream.synchronize()?;
            if round >= WARMUP {
                idle_latency.push(milliseconds(start));
            }
        }
    }
    // A 1 GiB aggregate working set exceeds the on-device caches on gfx1151.
    let bulk_bytes = 32 * 1024 * 1024;
    let bulk_batch = 16;
    let mut bulk = (0..STREAMS)
        .map(|_| Lane::new(&device, bulk_bytes))
        .collect::<Result<Vec<_>>>()?;
    let bulk_times = throughput(&mut bulk, samples, bulk_batch)?;
    let mut loaded_latency = Vec::new();
    for round in 0..samples + WARMUP {
        // Probe every queue position. Enqueue a bulk backlog on all streams,
        // then measure one small copy's completion before draining the backlog.
        for probe in &mut small {
            for _ in 0..4 {
                for lane in &bulk {
                    lane.enqueue()?;
                }
            }
            let start = Instant::now();
            probe.enqueue()?;
            probe.stream.synchronize()?;
            if round >= WARMUP {
                loaded_latency.push(milliseconds(start));
            }
            for lane in &mut bulk {
                lane.stream.synchronize()?;
            }
        }
    }
    for lane in small.iter_mut().chain(bulk.iter_mut()) {
        lane.check()?;
    }
    let idle = Distribution::from_samples(idle_latency)?;
    let loaded = Distribution::from_samples(loaded_latency)?;
    println!(
        "{}",
        json!({
            "target": device.target().as_str(), "queues": queues, "streams": STREAMS,
            "samples": samples, "warmup": WARMUP, "stream_creation_ms": creation_ms,
            "small_commands_per_second": STREAMS as f64 * small_batch as f64 * 1_000.0 / small_times.median_ms,
            "threaded_commands_per_second": STREAMS as f64 * small_batch as f64 * 1_000.0 / threaded_times.median_ms,
            "bulk_copy_gb_s": 2.0 * bulk_bytes as f64 * STREAMS as f64 * bulk_batch as f64 / (bulk_times.median_ms * 1e6),
            "small_batch_median_ms": small_times.median_ms, "small_batch_p95_ms": small_times.p95_ms,
            "threaded_batch_median_ms": threaded_times.median_ms, "threaded_batch_p95_ms": threaded_times.p95_ms,
            "bulk_batch_median_ms": bulk_times.median_ms, "bulk_batch_p95_ms": bulk_times.p95_ms,
            "idle_latency_median_us": idle.median_ms * 1_000.0, "idle_latency_p95_us": idle.p95_ms * 1_000.0,
            "loaded_latency_median_us": loaded.median_ms * 1_000.0, "loaded_latency_p95_us": loaded.p95_ms * 1_000.0,
        })
    );
    drop(streams);
    Ok(())
}
