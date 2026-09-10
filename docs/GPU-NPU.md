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

Cargo builds and rustdoc do not require XRT or a C++ compiler. The NPU shim is
loaded at runtime. For a development build with XRT headers and libraries installed:

```bash
bash scripts/build-npu-shim.sh
export HRX_NPU_RUNTIME_DIR="$PWD/artifacts/npu-runtime"
```

Shared GPU buffers additionally require native HRX interop ABI 1. Rebuild the
pinned native sources with `patches/loom/0008-export-owned-gpu-dmabuf.patch`, or
use `scripts/build-interop-overlay.py BUILD OUTPUT` against an existing matching
HRX build for local validation. The overlay script leaves that build unchanged.
Point `HRX_RUNTIME_DIR` at the result plus its matching native dependencies.
The currently published default bundle predates this extension; existing GPU-only
low-level use continues to work with it.

`hrx prepare-npu MANIFEST [ARCHIVE]` installs a separately verified NPU component.
`HRX_NPU_BUNDLE_MANIFEST` selects its pinned manifest; `HRX_OFFLINE=1` forbids
network provisioning. Local `HRX_NPU_RUNTIME_DIR` overrides are trusted development
inputs. `hrx doctor` reports device accessibility and configured runtime selection
without installing drivers or compilers.

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
