# GPU/NPU implementation validation

Validated locally on Strix Halo (gfx1151 + XDNA2), 2026-09-10. These results
qualify the paths below on this host; they are not a claim of production
qualification across driver versions, devices, or arbitrary kernel contracts.

| Check | Result |
| --- | --- |
| CPU tests with all features, offline | 54 passed, including doctests |
| Miri ownership and synchronization | 13 passed; deliberate quarantine leak excluded |
| Native GPU/compiler/heterogeneous tests | 24 passed |
| Prepared arithmetic GPU→NPU→GPU replay | Every element checked across 16 alternating inputs, using both blocking waits and Future polling; zero Rust heap allocations during replay |
| Feature combinations | All 64 checked, including all targets |
| MSRV | Rust 1.88, all features and targets |
| Clippy and rustdoc | Warnings denied |
| Cargo packaging | Packaged source builds without sibling checkouts or XRT headers |
| Independent NPU component | Hashed archive installed into an empty cache and executed the arithmetic pipeline offline |
| Ubuntu 26.04 without system XRT | Preparation, offline reuse, doctor, Loom compilation and both heterogeneous tests passed; loader traces confirm bundled XRT libraries |
| Partial provisioning failure | GPU cache remains verified and its path is printed; unavailable or offline NPU provisioning returns failure |

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

Blocking-wait scheduler results with the final Ubuntu 26.04 bundles (the caller
can execute its own ready regions through the same scheduler):

| Process | Direct p50, µs | Coordinated p50, µs | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 244.283 | 248.813 | 1.019 |
| 2 | 243.394 | 245.933 | 1.010 |
| 3 | 245.114 | 243.804 | 0.995 |
| 4 | 270.483 | 272.683 | 1.008 |
| 5 | 291.133 | 296.662 | 1.019 |

The median ratio is 1.010 (+1.0%). These measurements do not establish a speedup
from changing the build distribution.

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

Both GPU and NPU runtime bundles are built and pinned for Ubuntu 26.04 LTS,
the selected current-LTS baseline. This is a support policy, not a claim that
Ubuntu 24.04 cannot build the sources. Bundled GPU/Loom libraries require glibc
2.43; stock Ubuntu 24.04 is outside the supported baseline. The runtime manifests,
source inventories and published archive hashes describe the same 26.04 builds.

Both native binary archives and their corresponding source archives are published
in `native-20260910-gpu-npu`. The archive hashes match the crate manifests and
source inventories. Local archive installation has been verified, including
installation from the Cargo package. Anonymous preparation from the public URLs
into an empty cache, offline diagnostics, GPU→NPU→GPU arithmetic and cross-context
execution also pass without runtime-directory overrides.

Native compiler installation remains explicit: the library verifies and invokes
an installed, inventoried IRON/Peano or Chess environment. Chess was not tested on
this host. Driver-loss behavior is covered by scheduler fault injection and
resource quarantine tests; destructive device-reset testing and a wider native
failure-injection matrix remain release qualification work. Shared cache
maintenance is allocation-wide and cross-submission dependencies reserve whole
graphs, as documented in the API guide.

## Ownership and context coverage

Host writes flush before GPU or NPU consumption. Mapping reserves a host lease
under the scheduler lock, releases that lock before cache maintenance, and rolls
back the lease on failure. Fault-injection tests cover rollback; Miri covers
ownership and synchronization. The cached interop API retains its native library
for the lifetime of its function pointers, including cached resolution failures.

XRT group IDs encode a memory bank in bits 0–15 and a context slot in bits 16–23.
Bindings compare device ordinal and bank while allocations retain the full group
ID. The hardware test drops the cached program before loading a second context,
while its buffers retain their owning context. Both NPU-local and shared buffers
execute through that second context. This qualifies distinct contexts using the
same xclbin, not every pair of different xclbins.

## Application throughput

The independent SCRFD + DINOv3 harness has nine passing CPU tests covering
preprocessing, pipeline slot reuse, pipe framing and checkpoint provenance.
Its Lena run observed +2.5% median mixed-pipeline throughput, with overlapping
ranges, unequal queue depths and one repeated image. This does not establish a
reliable general speedup. See [results and limitations](../scripts/vision-bench/RESULTS.md).
