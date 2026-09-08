# Rust code review

Reviewed on 2026-09-08 for correctness, idiomatic Rust, and performance. Scope included all Rust modules, tests, and supporting scripts. Five actionable issues were identified. No implementation changes were made.

## Findings

### 1. [P1] Compatibility kernel execution is unsound as a safe API

Location: [`src/compat/kernel.rs`](src/compat/kernel.rs), lines 72–84.

`launch` and `launch_2d` are safe functions, but callers control dimensions, scalars, and raw argument bytes. Retaining allocations cannot prevent an oversized kernel access or an arbitrary address passed through `Args::raw`. Trusting the kernel does not establish that a particular invocation is safe.

Mark loading and dispatch unsafe, document their contracts, and keep safe operations in validated model wrappers. See the [Rustonomicon’s safe API requirements](https://doc.rust-lang.org/nomicon/safe-unsafe-meaning.html).

### 2. [P1] Separate streams can access shared buffers without synchronization

Locations: [`src/runtime.rs`](src/runtime.rs), lines 479–485; [`src/compat/device.rs`](src/compat/device.rs), lines 254–285.

The `Buffer::Sync` justification claims concurrent access is impossible because `Gpu` is not `Sync`. Different GPU/stream instances can nevertheless operate on the same buffer. Each transfer synchronizes only its own stream; compatibility transfers also use independent per-stream locks. This permits data races through safe calls.

A sequential probe demonstrated stale data: stream B read `3` after stream A queued a fill with `7`. After synchronizing A, B read `7`. This probe demonstrated missing ordering; it did not intentionally execute a concurrent data race.

Enforce stream ownership with explicit handoff, or track dependencies and synchronization per allocation.

### 3. [P2] Completed upload staging accumulates during synchronous reads

Location: [`src/runtime.rs`](src/runtime.rs), lines 672–681 and 846–854.

`read()` and `upload()` synchronize through `Gpu` but bypass `Stream::synchronize()`, which clears staging. Successful completion polling also retains staging.

Repeating 16 uploads of 8 MiB followed by synchronous reads retained 128 MiB of staging. Explicit synchronization immediately released that memory.

Centralize completion cleanup and reclaim staging whenever completion is established.

### 4. [P2] Public Manifest construction bypasses archive path validation

Location: [`src/bundle.rs`](src/bundle.rs), lines 126–150.

Validation occurs in `Manifest::parse`, while public fields and derived `Deserialize` permit unchecked manifests. `install()` trusts those names before joining them to the extraction directory.

A probe using a modified public manifest reproduced a `../escaped` archive entry writing outside the extraction directory despite valid archive and file hashes. The existing `Manifest::parse` path rejects this input correctly.

Validate at installation entry and before extraction, or make validated manifests an invariant-preserving type.

### 5. [P2] Compatibility synchronization scales with every session’s allocations

Location: [`src/compat/device.rs`](src/compat/device.rs), lines 163–172.

Every drain locks and scans the entire global allocation registry, including unrelated live buffers. Host reads and writes invoke this too.

A release-build probe measured the average time for 100 idle synchronizations while another compatibility device owned the allocations:

| Unrelated live allocations | Time per idle synchronization |
| ---: | ---: |
| 0 | 0.025 µs |
| 1,000 | 0.818 µs |
| 10,000 | 14.740 µs |

These are local microbenchmark measurements, not inference throughput measurements. They demonstrate the cost of the global scan.

Reclaim stale entries incrementally or periodically instead of scanning everything on each transfer.

## Idiomatic modernization

- Preserve error categories and underlying sources instead of flattening everything into `Error(String)` in [`src/lib.rs`](src/lib.rs), lines 19–28.
- Replace constant `CString` allocations with C string literals in [`src/runtime.rs`](src/runtime.rs), lines 331–332.
- Consider standard [`File::lock`](https://doc.rust-lang.org/std/fs/struct.File.html#method.lock) if raising the minimum Rust version from 1.88 to 1.89 is acceptable.
- Make the bounds test exercise the real `checked_span`; [`src/runtime.rs`](src/runtime.rs), lines 591–603, currently tests a duplicate implementation.

The main binding-dispatch path already avoids per-call heap allocation and uses fixed-size constant storage.

## Validation

The following checks passed during the review:

- All 21 CPU tests: `cargo test --all-features`.
- All 3 explicit GPU tests: `HRX_OFFLINE=1 cargo test --all-features --test gpu -- --ignored --test-threads=1`.
- Formatting: `cargo fmt --all -- --check`.
- Clippy: `cargo clippy --all-features --all-targets -- -D warnings`.
- Minimal feature build: `cargo check --no-default-features`.
- CPU tests and Clippy on installed stable Rust, using `cargo +stable`.
- Release build: `cargo build --release --all-features`.

Targeted probes additionally exercised staging retention, cross-stream ordering, manifest path traversal, and allocation-registry scan costs. Probe files were created outside the repository.

Rust 1.88 itself was not tested. Passing tests do not cover the safety and edge cases identified above. File references describe the code at review time.
