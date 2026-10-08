# Stream queue pool measurements

The pool size is configurable per live device:

```rust
let device = hrx::Device::open(0)?;
device.set_stream_queue_count(8)?;
let stream = device.stream()?;
```

Set it before creating the device's first stream. Clones, reopened handles and
`Stream::open()` share the setting. Changing the size after first use returns
`Error::Busy`, including after all the streams have been dropped while the device
remains alive. Repeating the current value succeeds. Zero is invalid. Queues are
created lazily; choosing a size beyond the hardware's capacity can fail during
stream creation. An explicitly created `fabric::Queue` is outside this pool.

## Reproduction

```sh
cargo build --locked --release --example queue_pool_bench
# Each command runs in a fresh process; run these sequentially, not concurrently.
for queues in 1 2 4 8 16; do
    target/release/examples/queue_pool_bench "$queues" 15
done
```

The example emits JSON with median and p95 timings. Every case has three warmup
rounds followed by fifteen measured rounds. Setup, compilation, allocations and
correctness checks are outside the steady-state timing. Sixteen logical streams
have disjoint source and destination buffers, and a 64-byte output prefix is checked.

- Small-command throughput: one host thread submits 1,024 copies of 4 KiB per
  stream, round-robin, then waits for all streams.
- Concurrent throughput: sixteen host threads each submit 1,024 copies of 4 KiB
  and wait for their own stream. A barrier releases the threads together; the
  slowest worker's duration determines throughput. Thread creation and joining
  are excluded, but scheduling and wakeup noise remain.
- Bulk throughput: sixteen copies of 32 MiB per stream, with a 1 GiB aggregate
  source/destination working set. GB/s counts both reads and writes.
- Idle latency: one 4 KiB copy followed immediately by its completion wait,
  sampled at every stream position.
- Loaded latency: enqueue four 32 MiB copies on every bulk stream, then time a
  4 KiB copy and wait on one small stream. Drain the backlog and repeat for every
  stream position. This measures latency during a finite bulk backlog, not
  latency under a fixed arrival rate.

The benchmark retains sixteen empty streams to keep all pool queues alive for
all cases. The sixteen-queue case gives each of the sixteen active streams its
own queue. Each case measures the same implementation with a different queue count.

## Recorded environment

Measurements were taken on 2026-09-29 with a Ryzen AI MAX+ 395 / Radeon 8060S
(`gfx1151`), 32 logical CPUs, about 125 GiB RAM, Linux `7.2.7-arch1-1`, and
`rustc 1.95.0-nightly (5fb2ff861 2026-02-21)`. The release build used the working
changes on top of commit `982376d75c55479e43ba1b74f7d6af9516d5b66d`.

Three fresh-process repetitions use orders `1,2,4,8,16`, `16,8,4,2,1`, and
`4,1,16,2,8` to reduce ordering bias. Raw per-process summaries are stored in
[gfx1151.jsonl](gfx1151.jsonl). These are transfer and submission microbenchmarks
on one GPU, not model inference or compute-heavy kernels. CPU/GPU clocks and OS
scheduling are not pinned; small differences should not be treated as decisive.

## Results and default

The default is **eight queues**. These measurements predate the timeline-lock
fix. Memory pressure was not recorded, and the host ran out of memory shortly
afterward. Rerun the benchmark before using these numbers to tune current builds.

The table uses the median of
three per-process summaries; p95 columns are medians of the three per-run p95s,
not pooled percentiles. Command rates are millions of completed copies per second.

| Queues | One host thread, Mcommands/s | 16 host threads, Mcommands/s | Bulk GB/s | Idle p95, µs | Loaded p95, ms | Create 16 streams, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.593 | 0.586 | 204.3 | 3.69 | 21.87 | 6.55 |
| 2 | 0.756 | 1.043 | 201.2 | 4.16 | 21.69 | 12.28 |
| 4 | 1.017 | 1.646 | 194.3 | 3.72 | 22.01 | 22.23 |
| 8 | 0.998 | 1.933 | 196.6 | 3.70 | 22.23 | 46.89 |
| 16 | 1.007 | 1.972 | 196.1 | 573.75 | 21.86 | 79.37 |

Eight queues had approximately 17% higher median concurrent submission throughput
than four, similar single-host throughput, and similar bulk bandwidth and idle
latency. Sixteen added only about 2% median concurrent throughput over eight,
while its idle p95 latency increased from about 3.7 µs to 574 µs. The cause of that
latency increase was not isolated. Four remains a reasonable lower-resource choice;
it also created the sixteen stream handles faster. A one-stream application still
creates only one queue regardless of the configured maximum.

Run-to-run variation is substantial: concurrent throughput ranged from
1.24–1.92 Mcommands/s with four queues, 1.51–2.18 with eight, and 1.97–2.25 with
sixteen. Eight did not outperform four in every repetition. Bulk-loaded
latency stayed around 22 ms for all sizes: increasing the pool does not remove
contention for the GPU and memory bandwidth. Benchmark the actual workload when
choosing an override, especially for compute-heavy kernels or a different GPU.

Pool assignment is round-robin with no priority or load awareness. More queues
can reduce sharing, but cannot guarantee that interactive work will avoid a long
kernel or device-side wait. Each stream remains on its assigned queue. Event
waits only target already-published native work; scheduled graph completions are
host dependencies, including graphs with `gpu_scoped` callbacks.
