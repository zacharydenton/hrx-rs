# GPU/NPU implementation validation

Validated locally on Strix Halo (gfx1151 + XDNA2), 2026-09-10. These results
qualify the paths below on this host; they are not a claim of production
qualification across driver versions, devices, or arbitrary kernel contracts.

| Check | Result |
| --- | --- |
| CPU tests with all features, offline | 49 passed, including doctests |
| Miri ownership and synchronization | 10 passed; deliberate quarantine leak excluded |
| Native GPU/compiler/heterogeneous tests | 23 passed |
| Prepared arithmetic GPU→NPU→GPU replay | Every element checked across 16 alternating inputs, using both blocking waits and Future polling; zero Rust heap allocations during replay |
| Feature combinations | All 64 checked, including all targets |
| MSRV | Rust 1.88, all features and targets |
| Clippy and rustdoc | Warnings denied |
| Cargo packaging | Packaged source builds without sibling checkouts or XRT headers |
| Independent NPU component | Hashed archive installed into an empty cache and executed the arithmetic pipeline offline |

The NPU passthrough fixture is compiled through the public Rust compiler API
with a pinned installed Peano/IRON toolchain. It exercises real NPU DMA. The
heterogeneous integration test adds real GPU BF16 arithmetic on both sides.
The representative GEMM example additionally ran a trusted BF16→F32 GEMM image
with M=2048, K=1024, N=1024 and GPU preparation/epilogue. It checked full output,
then measured p50 1.773 ms, p95 1.953 ms and 565.74 pipelined requests/s. Its six
shared allocations/imports occupied 30 MiB and reported zero explicit copy
bytes. The GEMM artifact was supplied separately; it is not shipped as an
unverified opaque binary in this crate.

## Dispatch overhead

`scripts/qualify-npu-performance.py` runs at least five fresh processes. Each
process alternates 100 coordinated submissions with 100 direct executions of
the same prepared native regions after warmup. It reports whether median ratios
exceed the 1.05 reference target and checks output and stable allocation/import
counts. Per the intended performance policy, 5% is guidance: normal runs report
measurements without failing on a small deviation. `--strict` opts into enforcement.
The fixture is a GPU fill followed by NPU DMA, using shared allocations.

Final blocking-wait scheduler results (the caller can execute its own ready
regions through the same scheduler):

| Process | Direct p50, µs | Coordinated p50, µs | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 190.735 | 192.795 | 1.011 |
| 2 | 188.985 | 191.805 | 1.015 |
| 3 | 187.755 | 189.075 | 1.007 |
| 4 | 187.855 | 190.244 | 1.013 |
| 5 | 190.385 | 193.285 | 1.015 |

Earlier scheduler versions exceeded the target in repeated measurements. The
final version wakes one worker for a serial chain, wakes its peer for independent
regions, and uses separate condition variables for host mappings and workers.
A blocking wait can claim ready work from its own submission before a worker
claims it, avoiding a round trip through another thread. Future polling and
timed waits never execute native work on the caller; their wakeup overhead is
not included in this blocking-wait performance claim.
These measurements do not establish a latency bound. Significant slowdowns
should be investigated; small variations around 5% are acceptable. Tail latency on this host varied.

## Existing GPU API

Compared against commit `3f61554` using the same native runtime, five alternating
process pairs and 31 samples per process. Only the benchmark sample count was
made configurable in the baseline checkout; its library code was unchanged.
The table reports medians across processes, with throughput ratios inverted so
larger always means worse. Shorter runs showed substantial host enqueue noise;
these are measurements, not a worst-case guarantee.

| Metric | Candidate / baseline cost |
| --- | ---: |
| Allocation/drop | 0.989 |
| Dispatch enqueue | 0.989 |
| Dispatch completion | 1.001 |
| Graph enqueue | 0.808 |
| Dependent graph completion/kernel | 1.014 |
| Independent graph completion/kernel | 0.999 |
| Upload throughput, inverse ratio | 0.997 |
| Scratch reuse | 1.018 |

The new native error message uses a boxed slice to avoid enlarging the previous
`Error`/`Result` representation on success paths. `stream_bench` accepts
`HRX_BENCH_SAMPLES`; `scripts/compare-gpu-performance.py BASELINE CANDIDATE`
alternates executable order and reports all measurements against the 1.05
reference ratio. Add `--strict` to enforce it. Both executables must support the sample-count setting.

## Reproduce

```bash
HRX_NATIVE_BUILD=/absolute/matching/hrx-build \
HRX_NPU_TOOLCHAIN=/absolute/toolchain.json \
  bash scripts/test-npu-hardware.sh

HRX_RUNTIME_DIR=/absolute/interop-runtime \
HRX_NPU_RUNTIME_DIR=/absolute/npu-runtime \
HRX_TEST_NPU_DIR=/absolute/passthrough-artifact \
  python3 scripts/qualify-npu-performance.py

cargo test --locked --all-features
cargo +nightly miri test --lib --no-default-features execution:: -- --test-threads=1
bash scripts/check-feature-matrix.sh
cargo package --allow-dirty --all-features
```

Set `HRX_QUALIFY_PERFORMANCE=1` to include the performance comparison in the hardware
script; manual GPU/NPU CI dispatches enable it.

The hardware script rebuilds the local interop extension and NPU shim, compiles
the fixture, diagnoses the configured host and runs the native test suite.
The source package, component archive and detailed local logs are retained in
`target/package/` and `artifacts/` for review.

## Distribution and remaining qualification

The published default GPU bundle still predates interop ABI 1. Shared execution
currently requires the documented matching runtime override plus the separate
NPU shim; Cargo builds remain independent of either native runtime. The native
patch, its digest, the shim sources, component staging tool and offline installer
are included. A release must rebuild and publish the matching GPU bundle, assign
the new release version, and verify anonymous installation before claiming an
out-of-the-box shared runtime.

Native compiler installation remains explicit: the library verifies and invokes
an installed, inventoried IRON/Peano or Chess environment. Chess was not tested on
this host. Driver-loss behavior is covered by scheduler fault injection and
resource quarantine tests; destructive device-reset testing and a wider native
failure-injection matrix remain release qualification work. Shared cache
maintenance is allocation-wide and cross-submission dependencies reserve whole
graphs, as documented in the API guide.
