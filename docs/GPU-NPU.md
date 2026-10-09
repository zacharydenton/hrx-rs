# GPU and NPU execution

HRX uses libamdf for both devices. Enable the `npu` feature for XDNA execution;
Loom compilation for both targets is always available. Supported hardware is
`gfx1151` and Strix Halo NPU5 (`17f0`, revision `11`) on Linux x86_64.

## Native setup

```sh
cargo install hrx-rs --version 0.10.1 --locked --features npu
hrx prepare
hrx doctor
```

The native bundle supplies libamdf, the executable bridge and Loom. The host
supplies drivers, firmware, device permissions, glibc 2.43+ and compatible C/C++
runtimes. See [installation](../README.md#install-the-cli) for offline setup and
library overrides.

## NPU programs

`loom::Compiler::for_target(None, &Target::xdna())` selects the NPU compiler
profile. Compile a pipeline export with `Specialization` and load its `.xdna`
artifact through `execution::NpuDevice::load_artifact`. Supply the column count
and a `KernelContract` describing buffer lengths, alignment and access modes.
Each invocation establishes device state, including when contexts time-share
the array. One program supports up to 128 pending invocations with immutable
bindings.

Workers can use C23/C++26 translation units linked with Loom pipelines.
See the [compiler guide](COMPILER.md) for sources, reports and diagnostics.

Loading native code and defining its memory contract are unsafe. Image and
binding checks do not validate arbitrary program behavior.

## Shared memory

Use `MemoryPlacement::Shared(device)` for backing accessible by both engines,
or `NpuLocal(device)` for NPU storage. Adopting an owned GPU allocation exports
and imports a dma-buf. Aliases retain the original allocation and share host
access guards and visibility transitions.

Ordinary GPU buffers use cacheable system memory with explicit host cache
maintenance. `Stream::allocate_shared` uses coherent memory. On gfx1151,
floating-point atomic max does not update this coherent backing: use an integer
compare/exchange loop over the float bits instead of `view.atomic.reduce<maxnumf>`.
Ordinary loads, stores and arithmetic are unaffected.

`Fabric::allocation_profiles` queries access for the complete device set before
allocation. A profile describes visibility and creates backing with that profile.
For existing backing, `Buffer::visibility(producer, consumer)` describes reach,
release/acquire actions and width-specific atomic scopes. These describe memory
visibility; execution dependencies still need ordering.

`prepare_release_host` and `prepare_acquire_host` merge ranges and retain the
required host cache operations. Host mapping guards reject conflicting device
use. Shared allocation profiles cover new allocations, not external imports or
host-page registration.

## Scheduling and ownership

`execution::Runtime` runs at most one region per upload, compute, download and
NPU lane, including across independent graphs. Independent lanes can overlap;
buffer hazards order conflicting work. `RuntimeOptions::max_submissions` sets
queue capacity. Completion handles support blocking waits and Rust futures.

A timeout retains accepted work and its storage. Failed retirement or teardown
quarantines the owners. Memory budgets track buffers, model reservations and
executable image charges; include private workspace in model reservations.
See [memory budgets](EXECUTION.md#memory-budgets-and-files) for accounting rules.

## Resident GPU–NPU exchange

`fabric::ResidentSession` owns a `ResidentStartup`, a prepared PM4/AQL GPU
participant and an immutable `XdnaProgram`. Both programs must implement the
same WAIT/RUN/ABORT protocol. Submit them in either order and call `start()`
after both are accepted. They can then exchange payloads without host round
trips. `abort()` is available before RUN; a timeout does not cancel running work.

Choose coherent attachments with `Fabric::allocation_profiles(devices, true)`
and check each directional memory contract. The test fixture uses system-scope
GPU release/acquire operations and chained NPU DMA completion.
`XdnaProgram::wrap_transaction` retains extra buffers and command storage around
the compiler's invocation. Custom prefix/suffix records must use disjoint native
resources and drain their DMA traffic.

```sh
cargo test --all-features --test resident_sessions -- --ignored --test-threads=1
```

For GPU-authored file I/O, see [native storage](STORAGE.md).

## Examples

Run hardware tests serially:

```sh
bash scripts/test-npu-hardware.sh
cargo run --release --features npu --example shared_roundtrip
cargo run --release --features npu --example shared_bench
cargo run --release --features npu --example gemm_pipeline
cargo run --release --features npu --example gpu_npu_parallel -- 16777216 1024 21 trace.json
```

[`gpu_npu_parallel`](../examples/gpu_npu_parallel.rs) puts two independent
branches in one graph: a GPU integer-vector transform and batched NPU 8×8 BF16
matrix multiplication with FP32 outputs. Arguments are GPU element count, NPU
matrix count, paired sample count and an optional Chrome/Perfetto trace path.
Defaults are 16,777,216 elements, 1,024 matrices and 21 samples.

The example checks every output against CPU calculations, changes inputs across
replays, and alternates sequential and parallel execution. Timings exclude
compilation, initial upload and readback. Replay checks for new native allocations
or imports. Trace intervals use host clocks.

A 21-pair run on gfx1151/NPU5 on 2026-09-22 measured 0.754 ms sequential and
0.610 ms parallel (1.24×), with exact output agreement. The NPU-dominated
16,384-matrix case showed little speedup; small workloads can lose to scheduling
overhead.

`gemm_pipeline` runs an included 8×8 BF16/BFP16 matrix fixture between GPU
preprocessing and an epilogue, including concurrent requests.
`tests/native_execution.rs` checks nonuniform matrices and mixed C++/Loom workers.
Launch buffer lists must contain only used bindings: the pinned compiler's
unused-middle-binding relocation is rejected by the native image validator.
