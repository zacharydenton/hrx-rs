# Validation

## Reviewed native bundle, 2026-09-09

The replacement bundle uses freshly built HRX/Loom and the AMD runtime from
TheRock build 26672984641. Its 13 library files, license texts, inventory, and
source-archive reference are pinned in `bundle.json`.

```text
06000605de874bc972e17e90f94a3816302fdd2a17040cfaf1df1f0b70483cc4
```

The library bytes passed all 27 CPU tests, both Rust doctests, and all 18 ignored
GPU/compiler tests on gfx1151. Installation started with an empty cache and used
no runtime, compiler, or HSA overrides. Every dynamic dependency resolves to the
bundle or a documented host library. The Arch library chain and `loom-compile`
are absent. Source hashes match the recorded upstream manifests; this does not
claim byte-identical reconstruction of AMD's CI build.

The final archive also passed all 13 native tests from an isolated checkout of
commit `2530451`. The public default URL installed into a second empty cache
without GitHub authentication, and `HRX_OFFLINE=1 hrx info` initialized gfx1151.
The source archive returns HTTP 200 anonymously; GitHub reports matching SHA-256
digests for both uploaded archives.

Cargo's publishing dry run passed, packaging 70 files (185.1 KiB compressed) and
successfully compiling the packaged crate. No crates.io upload was performed.

See [native/RELEASE.md](native/RELEASE.md) for source and build evidence.

The View API tests also check nested offsets, parent bounds, empty and overflowing
regions, and unchanged surrounding bytes after stream transfers and graph replay.
A release benchmark run after this change measured 24.3 GiB/s queued uploads,
130 ns host dispatch recording, and 806 ns host recording per 32-kernel replay.
This run uses the reviewed bundle; the historical comparison below uses the
original bundle. Each metric is the median of nine samples after three warmups.

## Original bundle results, 2026-09-09

Tested on Linux x86_64 with a gfx1151 GPU. Archive installation was checked with
an empty cache. Native tests use the verified bundle cache with `HRX_OFFLINE=1`
and without runtime, compiler or HSA library overrides.

| Check | Result |
| --- | --- |
| CPU tests | 27 passed |
| Rust API doctests | 2 passed |
| Ignored runtime and compiler tests | 18 passed |
| Feature matrix | All 8 subsets build, including on Rust 1.88 |
| Clippy | Passes with warnings denied on stable and Rust 1.88 |
| Rustdoc | Passes with warnings denied |
| Package contents | Internal review documents excluded |

The native tests cover queued transfers, staging batching and reuse, event
ordering between streams, moving pending streams between threads, rejection of
shared scratch allocations, argument packing, workgroup validation,
resource retention, graph replay, compiler caching and diagnostics.

The tested archive SHA-256 is:

```text
c151a978eff1c7def5c54b9acfd595cc1e8ca21b793b4613864a18281d846bb1
```

A cached directory matched every file digest in `bundle.json`. Repacking it
reproduced this archive hash and replaced the stale `34591d78…` archive in
`artifacts/`. The anonymous release URL returned 404 during the initial bundle
check. After the repository became public, an unauthenticated HEAD request to
the pinned archive returned HTTP 200 on 2026-09-09. This availability check did
not repeat installation or GPU tests. Dependency provenance and notices remain
incomplete; see [THIRD-PARTY.md](THIRD-PARTY.md).

To repeat the bundle check with an empty cache:

```sh
export XDG_CACHE_HOME="$(mktemp -d)"
unset HRX_RUNTIME_DIR HRX_LOOM_LIBRARY HRX_BUNDLE_MANIFEST
cargo run --all-features --bin hrx -- prepare /path/to/hrx-linux-x86_64-gfx1151.tar.gz
HRX_OFFLINE=1 cargo run --all-features --bin hrx -- info
HRX_OFFLINE=1 cargo test --all-features -- --ignored --test-threads=1
```

These results cover this crate. Consumer model quality and performance were not
retested. Only gfx1151 was exercised; GPU timestamp profiling is not implemented.

## Stream performance

Release builds on the same Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), using
Rust 1.95.0-nightly (5fb2ff861), Linux 7.2.3 and the bundle above. The baseline is
commit `af2f7c7`; the candidate removes `Gpu`, `compat` and the application C API
helpers. Both run the same [benchmark](examples/stream_bench.rs) through `Stream`.

| Operation | Baseline | Stream after removal |
| --- | ---: | ---: |
| Queued upload, 256 MiB in 4 MiB chunks | 23.80 GiB/s | 24.08 GiB/s |
| Dispatch recording, host time per kernel | 131 ns | 131 ns |
| Dispatch batch through completion, per kernel | 2.158 µs | 2.154 µs |
| Graph replay, host time per 32-kernel replay | 815 ns | 815 ns |
| Graph batch through completion, per kernel | 2.200 µs | 2.194 µs |
| Scratch acquire/recycle, 1 MiB | 43.1 ns | 43.8 ns |
| Allocate/drop, 1 MiB | 367 ns | 361 ns |

Each process discards three warmups and takes the median of nine samples. The
table takes the median of five processes per version, alternating execution
order. Compilation, loading and initial data setup are outside the timed sections.
Each dispatch sample records 2,048 kernels; each graph sample replays 32 kernels
256 times. Samples end with stream synchronization. Upload and kernel outputs
are checked. Scratch samples contain 50,000 acquire/recycle pairs; allocation
samples contain 2,000 allocate/drop pairs.

The measured medians differ by less than 2%. Graph replay reduces host recording
work to roughly 25 ns per kernel here.

## Graph dependency cost, 2026-09-09

Declared edges are not free, which is why the graph API makes every dependency
explicit rather than chaining. Recording the same 32 Euler dispatches as a
declared chain and as independent nodes, on the bundle and host above:

| Recording | Per kernel through completion |
| --- | ---: |
| Chained (each node after the previous) | 2.197 µs |
| Independent (no declared dependencies) | 1.664 µs |
| Difference | 0.533 µs per edge |

`examples/stream_bench.rs` reports these as `graph_complete_ns_per_kernel`,
`graph_independent_ns_per_kernel` and `graph_edge_cost_ns`. A latency-bound
variant in `runtime::dag_probe` isolates the cost further: 64 tiny fill nodes
replay in ~151 µs chained and ~88 µs independent, about 0.95 µs per edge. The
Euler figure is smaller because a 2.2 µs kernel hides part of the scheduling
cost. The runtime engages additional workstreams only once a schedulable run
reaches 16 nodes, so larger graphs benefit more.

The independent recording is a scheduling measurement only: a zero timestep makes
the Euler result order-independent, so it is not a claim that these dispatches may
be reordered in general. These are wall-clock measurements, including
native runtime costs, not GPU timestamps or model benchmarks. The upload result
includes the host staging copy on this integrated GPU; it is not PCIe bandwidth.

To run after preparing the bundle:

```sh
HRX_OFFLINE=1 cargo run --release --example stream_bench
```

To compare another revision, copy `examples/stream_bench.rs` into its checkout,
build both with the same Rust toolchain, and alternate their release binaries.
For older revisions, adapt the transfer signatures using [CHANGELOG.md](CHANGELOG.md).
Keep other GPU work idle and use the same native bundle for both.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests using stable
Rust and Rust 1.88. It checks formatting, builds all targets, runs clippy, CPU tests
and rustdoc, checks every feature subset, and lists package contents.

`.github/workflows/gpu.yml` runs on pushes or manual dispatch. It needs a
registered runner with labels `self-hosted`, `linux`, `x64`, and `gfx1151`, access
to `/dev/kfd` and the render node, rustup, curl, and HTTPS access. Runner
registration is still outstanding.

The GPU job provisions an empty cache and runs every ignored test, including
compiler and library tests. It fails if provisioning fails. The optional
repository variable `HRX_BUNDLE_MANIFEST` is an HTTPS URL to a mirror manifest;
the workflow downloads it and sets the process environment variable to its local
path. Without a mirror, the job uses the pinned release.
