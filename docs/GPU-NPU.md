# Coordinated GPU and NPU execution

`hrx::execution` is the coordinated API. `hrx::gpu` exposes the existing low-level
GPU API; its historical root-level names remain available. The old
`hrx::npu::Shared` API has been replaced. It could not safely retain memory across
BO clones or coordinate host access with pending device work.

Create a `Runtime`, load trusted GPU/NPU specializations, allocate `Buffer`s, and
build a `Graph`. `prepare()` creates native GPU graphs, NPU run objects and bounded
completion slots. `submit()` returns a `Completion` supporting `wait()`,
`wait_timeout()`, `is_complete()` and standard Rust `Future` without Tokio.

```rust,no_run
use hrx::execution::{MemoryPlacement, Runtime};
# fn main() -> hrx::Result<()> {
let runtime = Runtime::new()?;
let buffer = runtime.allocate(4096, MemoryPlacement::GpuLocal)?;
let mut graph = runtime.graph();
graph.fill(buffer.view(), 7)?;
let executable = graph.prepare()?;
executable.submit()?.wait()?;
# Ok(())
# }
```

For NPU work, load an xclbin through `runtime.npu(0)?.load_program(path)` and bind
its instruction bytes using `program.kernel(instructions, contract)`. Both are
unsafe trust boundaries. Compiling code or verifying a checksum does not prove
that its DMA descriptors obey the declared argument extents. Each
`BindingContract` declares bytes, alignment, access and layout; scalar bytes and
GPU launch dimensions are fixed when the specialization is loaded. Native buffer
addresses are checked against alignment requirements during graph construction.

`MemoryPlacement::Shared(program.clone())` allocates GPU-visible memory, exports
it as a dma-buf and imports it into XRT once. Nonzero HSA export offsets are
preserved with retained sub-buffer mappings. The same buffer binds directly to
GPU arithmetic kernels and NPU kernels. `NpuLocal(program.clone())` is host-mapped
NPU storage; `GpuLocal` avoids NPU setup for GPU-only work. Unsupported sharing
returns an error. Layout conversion and explicit copies remain visible operations.

Use `map_read()` and `map_write()` on host-mapped allocations. Their guards
retain access until dropped. Blocking mappings wait for pending device work;
`try_map_*` returns `Busy`. A conflicting live host guard returns `Busy` rather
than waiting, including when a graph tries to use its allocation. Do not keep a
mapping live while submitting work that needs the same bytes.

The scheduler infers dependencies from overlapping reads/writes, including
conflicts across prepared graphs and repeated submissions. Cross-engine cache
maintenance currently reserves the entire shared allocation. Use separate
allocations for independently pipelined chunks. Two workers allow one native GPU region and one native NPU region in flight.
Either worker can continue a ready chain across devices, avoiding an unnecessary
thread handoff while preserving independent progress on the other engine. Serial
chains wake one worker; independent regions wake its peer. Host mapping waiters
have a separate wakeup path. Blocking `Completion::wait()` can execute ready
regions from its own submission, preserving the same dependency and engine
limits. `Future::poll` and `wait_timeout()` leave device execution to workers.
Cross-submission reservations are currently
at whole-graph granularity; they favor correctness over maximum overlap between
partially dependent graphs.

A prepared graph has two completion slots by default. Increase
`RuntimeOptions::graph_slots` for a deeper queue. A retained completion observer
keeps its slot reserved even after completion; drop completed observers to recycle
slots. Submission capacity exhaustion is `Busy`, never an implicit allocation.
Dropping an observer detaches it. `wait_timeout()` does not cancel execution;
`cancel()` prevents unscheduled nodes and drains running work. NPU native waits
have a 60-second watchdog and attempt a synchronous abort on failure. If native
completion becomes uncertain, allocations are poisoned and quarantined until
process exit; the library does not free memory beneath DMA.

`Runtime::statistics()` reports allocations, imports, submission/completion counts,
explicit copy bytes, cache-maintenance extents and retained memory. These counters
cover the coordinated API, not calls through the low-level GPU or raw XRT APIs.

## Native setup

Cargo builds and rustdoc need no XRT installation, Python, ROCm SDK or C++ compiler.
Install the CLI with GPU and NPU support:

```bash
cargo install hrx-rs --version 0.3.0 --locked --features runner,npu
hrx prepare
hrx doctor
```

For a source checkout, use `cargo install --path . --features runner,npu`.
The matched native runtimes and corresponding sources are published in
[the native release](https://github.com/zacharydenton/hrx-rs/releases/tag/native-20260910-gpu-npu).

`hrx prepare` downloads and verifies both manifests: `bundle.json` contains GPU
interop ABI 1 and Loom, and `npu-bundle.json` contains the NPU shim, matching XRT
core libraries, the XDNA plugin and libuuid. APIs also provision their runtime
on first use. Precompiled NPU programs require no IRON, Python, vendor ONNX
Runtime, Ryzen AI SDK or `xdna-vision` installation. NPU kernel compilation is
an optional, separately configured toolchain as described below.

The host still supplies Linux x86_64, its normal C/C++ runtime libraries,
`amdgpu`/KFD, the `amdxdna` kernel driver, NPU firmware and device permissions.
These are system prerequisites; Cargo cannot supply kernel drivers or firmware.
The NPU binaries are built against Ubuntu 26.04 (glibc 2.43 and GCC 15 runtime).
The GPU/Loom libraries are also built against Ubuntu 26.04.

For an offline install, provide the two matching archives:

```bash
HRX_OFFLINE=1 hrx prepare /path/to/gpu.tar.gz /path/to/npu.tar.gz
HRX_OFFLINE=1 hrx doctor
```

Passing only the GPU archive still resolves the NPU runtime from its cache or
download URL. Set `HRX_OFFLINE=1` to forbid downloads. The GPU directory is printed
to stdout as soon as it is ready; a subsequent NPU failure reports its error on
stderr and returns a nonzero exit status, leaving the verified GPU cache usable.

`hrx prepare-npu` prepares only the NPU runtime; its optional `MANIFEST [ARCHIVE]`
arguments support custom components. `HRX_NPU_BUNDLE_MANIFEST` selects a mirror or
custom pinned runtime, and `HRX_OFFLINE=1` forbids network provisioning.
`HRX_NPU_RUNTIME_DIR` and `HRX_RUNTIME_DIR` select trusted development runtimes.
`hrx doctor` reports accessible devices and probes installed runtimes without
provisioning. It checks the NPU shim ABI and linked dependencies; loading a
particular program remains a separate hardware check.

Maintainers can rebuild with `scripts/build-npu-runtime.sh`; see
[native/NPU-RELEASE.md](../native/NPU-RELEASE.md). To develop only the shim with
installed XRT headers/libraries, use `scripts/build-npu-shim.sh`. GPU development
overlays remain available through `scripts/build-interop-overlay.py`.

## NPU compilation

Enable `npu-compile`. `npu::compiler::Compiler` runs an IRON generator or AIE MLIR
project in a private subprocess workspace. It never initializes a GPU or NPU.
Compiler input generators are trusted host programs, not sandboxed code.

Pin an installed toolchain in its configured environment:

```bash
python3 scripts/pin-npu-toolchain.py \
  --python /absolute/ironenv/bin/python \
  --aiecc /absolute/ironenv/bin/aiecc \
  --backend Peano --output toolchain.json
cargo run --features npu-compile --example compile_npu -- toolchain.json 262144
```

For Chess, source its installed environment first and select `--backend Chess`.
The pinning tool inventories its support files and records child-only environment
settings. Add `--identity-root` for dependencies outside the virtual environment.
No proprietary compiler is downloaded or redistributed. Project generators must
honor their selected backend (`HRX_AIE_BACKEND`) and declare all source/header
inputs. Incomplete dependency declarations require `cacheable: false`.

`Compiler::compile_all()` bounds concurrency and preserves input order. Cache
identities include source contents, dependencies, specialization arguments,
contract and pinned compiler/environment identities. Hits verify output digests;
changed tools are rejected. Failed processes retain `build.log`; timeouts kill
the compiler process group. Artifacts contain `x.xclbin`, `x.bin`, `manifest.json`,
source snapshots and diagnostics. Load an artifact only after asserting its
native-code contract. NPU artifact caching uses the shared kernel cache's `npu/`
subdirectory and participates in `hrx gc`. Uncacheable artifacts own temporary
workspaces that disappear when the last artifact handle is dropped.

## Reproduction and qualification

`native/npu/fixtures/passthrough.py` is pinned from MLIR-AIE commit `0d49a88`.
Build it for `-d npu2 -n 262144`, then run:

```bash
cargo run --release --features npu --example shared_roundtrip -- \
  PATH/x.xclbin PATH/x.bin 1048576
HRX_TEST_NPU_DIR=PATH cargo test --release --all-features \
  --test heterogeneous -- --ignored --test-threads=1
```

The hardware test runs GPU BF16 arithmetic, an NPU DMA program, then GPU BF16
arithmetic, checking every element with alternating inputs across repeated
executions. `shared_bench` measures real warm NPU dispatch latency. The ignored
library test `heterogeneous_latency_against_direct_backend` alternates coordinated
submissions with direct execution of the same prepared native regions, checks
output, and verifies that allocation/import counts and copy bytes do not grow.
These fixtures establish execution and coherence; they do not claim that every
model benefits from using both devices.

The full [validation record](GPU-NPU-VALIDATION.md) documents measured overhead,
allocation-free replay, the GEMM example and remaining release qualifications.
Run `python3 scripts/qualify-npu-performance.py` with the same runtime and fixture
environment for a five-process comparison against the 5% reference target.
Add `--strict` only when a hard threshold is wanted.

### Published runtime and alignment checks

The coordinated `Graph::gpu` path requires the allocation-address query in interop
ABI 1 even for `GpuLocal` buffers: checking view offsets alone does not prove base
pointer alignment. The published GPU bundle pinned by this crate includes this query. The
low-level `hrx::gpu` API and coordinated GPU fill/copy operations do not gain this
requirement. No NPU device is required for coordinated GPU-only graphs.

NPU host-only and imported buffers may be passed between resident program contexts
on the same device when their memory banks match the argument. The graph validates
those properties and retains both contexts; it does not require context identity.
