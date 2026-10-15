# GPU and NPU execution

HRX uses libamdf for both devices. Enable the `npu` feature for XDNA execution;
Loom compilation for both targets is always available. Supported hardware is
`gfx1151` and Strix Halo NPU5 (`17f0`, revision `11`) on Linux x86_64.

## Native setup

```sh
cargo install hrx-rs --version 0.11.0 --locked --features npu
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
The bundled pipelines use `run` regions, explicit FIFO channels, and separate
DMA and compute strands. See [the copy kernel](../tests/kernels/copy.xdna.loom)
for a complete example using the current compiler syntax.
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

### Stable addresses and NUMA placement

`Fabric::reserve_virtual_memory` reserves host/device addresses independently of
their backing. Query `host_reservation_capabilities` or
`Device::reservation_capabilities(domain)` first, then use `ReservationAccess`
to select consumers and permissions. `VirtualMemory::access_info` reports the
stable address, mapping granularity and shared mapping-group identity.

Allocate sources with `Fabric::allocate_with_options`: `mapping_source` prepares
device remapping, `host_mapping_source` prepares CPU aliases, and
`required_backing_flags` carries the reservation capability's
`required_source_flags`. Host/SVA groups require both kinds of preparation for
every sharing device. `host_numa_node` requests physical placement on a specific
OS NUMA node and fails if it cannot be enforced.
`AllocationProfile::supports_host_numa` reports whether a profile accepts this
constraint. `Endpoint::gpu_architecture` classifies GCN/RDNA/CDNA independently
of queue services; queue capability queries expose remapping, command-array and
dispatch-timestamp support.

`VirtualMemory::update` replaces ranges; `detach` removes a complete mapping
group while retaining its addresses. Both are unsafe because callers must retire
users and order mapping changes. Sources remain leased, including replaced and
proposed sources after failure, until checked whole-group removal. Failed
destruction quarantines those owners. Ordinary `Buffer::read/write` returns
`Busy` while its source is leased by a reservation.

`Argument::Address(address)` binds a reservation address to a GPU pointer
parameter. It does not infer ownership, visibility or sanitizer bounds: retain
the reservation through completion and perform the required memory transitions.
The coordinated execution planner does not infer hazards for these raw addresses.

## Native observations

`Fabric::host_clock_info/host_clock` expose the shared monotonic profiling clock.
`Device::queue_clock_info(family)` reports nominal GPU timestamp units, and
`clock_sample(family)` returns a native device sample with host observations
before, paired with, and after the query. Equal rates do not establish equal
epochs or a bounded CPU/GPU correlation. Existing PM4 profiling now gets its rate
from this public native service.

`Device::aql_queue_with_timestamps(private_bytes, budget, true)` enables firmware
dispatch timestamps. After checked completion, `AqlCompletion::timing()` returns
the dispatch's owned start/end sample and clock units. Samples survive signal
reuse. A profiled prepared dispatch admits one invocation at a time and allocates
one sample per invocation; ordinary queues keep their existing behavior. These
intervals cover firmware packet processing, not exact shader instruction bounds.

For the NPU, `firmware_trace_info`, `configure_firmware_trace` and
`read_firmware_trace` expose the device-wide recorder. Reads return owned decoded
events, positions, lost-byte counts and discontinuity flags. Category constants
are in `fabric::native`. Configuration requires native admission and ownership
coordination across processes; dropping a device does not disable recording.
An empty read does not prove workload completion. `timer_clock_info` samples the
current RECORD_TIMER operating rate without assuming that it stays constant.

### Advanced native protocols

`Fabric::native_api()` exposes version-checked core, GPU, XDNA and observation
tables. `fabric::native` publishes the complete pinned ABI and its contracts;
`native_handle()` borrows instances, endpoints, devices, buffers or reservations
without transferring ownership. This provides access to the following protocols:

| Functionality | Native entry points |
| --- | --- |
| Queue-ordered remapping | `kernel_queue_update_memory` with VIRTUAL_MEMORY capability |
| Reservation host views and backing-qualified visibility | `memory_reservation_map`, `memory_query_pair_info` |
| Ordered NPU command arrays | XDNA `kernel_queue_submit`, bounded by queue command/packet capacity |
| Finite NPU timer/register results | `context_query_result_buffer_info`, `context_attach_result_buffer`, `context_collect_result_buffer`, `context_detach_result_buffer` |

These are unsafe caller-managed protocols, not automatic operations on an
`XdnaProgram` or execution graph. For NPU results, query the context's private
scope/profile, allocate and clear/publish a finite destination, attach it, submit
caller-authored RECORD_TIMER/READ_REGS instructions, retire every writer, collect,
acquire the host mapping and decode the known record layout before detaching.
The transport does not report a written extent or overflow count. Failed controls
can preserve native attachments; keep backing and owners alive until checked
retirement and removal. Never destroy Rust-owned handles through the raw API.

The native bundle remains scoped to gfx1151 and Strix Halo XDNA. Upstream x86,
SPIR-V/Vulkan and model-serving additions are separate systems, not newly enabled
Rust execution backends.

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
