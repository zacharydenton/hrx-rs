# Native GPU and NPU execution

Version 0.8 uses libamdf for both engines. Enable `npu` for the tracked XDNA
execution API; Loom compilation for both targets is always available. The
qualified hardware is gfx1151 and Strix Halo NPU5 (`17f0`, revision `11`) on
Linux x86_64. Other profiles fail explicitly at device admission.

## Native setup

```sh
cargo install hrx-rs --version 0.8.0 --locked --features npu
hrx prepare
hrx doctor
```

One verified archive supplies libamdf, the executable bridge, and Loom.
The host supplies drivers, firmware, permissions, glibc 2.43+, and compatible
C/C++ runtimes. No ROCm SDK, XRT, Python, IRON, or external NPU compiler is used.
For offline installation, use `HRX_OFFLINE=1 hrx prepare native.tar.gz` with the
archive matching `bundle.json`. `HRX_RUNTIME_DIR` selects a trusted local build;
`HRX_AMDF_LIBRARY`, `HRX_FABRIC_LIBRARY`, and `HRX_LOOM_LIBRARY` override individual
libraries. Overrides bypass distribution integrity checks.

## Compilation

`loom::Compiler::for_target(None, &Target::xdna())` creates an offline compiler
for the exact NPU profile. Compile a pipeline export with `Specialization` and
load its `.xdna` artifact through `execution::NpuDevice::load_artifact`.
Supply the column count and a trusted `KernelContract` describing buffer
lengths, alignment, and access modes. Native images establish complete device
state on each invocation, including when contexts time-share the array.

`CxxSource` supplies named C23/C++26 translation units and explicit virtual
headers. `Compiler::sources` links them with Loom modules. Supported vector
workers can be written in C++ while Loom supplies the XDNA pipeline. There is
no implicit host filesystem include search. Unsupported language/target
operations produce compiler diagnostics with source identity.

Request `ReportMode::Summary` or `Details` on a specialization. Reports retain
the native JSON schema, compiler identity, target, backend, and processor mode.
Missing resource fields remain unknown. `hrx report show` and `report diff`
inspect saved reports and reject comparisons with incompatible identities.
CU/WGP policies apply to AMDGPU only.

## Memory and scheduling

`execution::Runtime` owns GPU/NPU lanes and prepares reusable dependency graphs.
Use `MemoryPlacement::Shared(device)` for one backing accessible by both
engines, or `NpuLocal(device)` for NPU storage. GPU-owned allocations are adopted
through native dma-buf export/import. Aliases retain the original allocation,
share access guards, and perform the required visibility transitions.
Caller-owned GPU host-page registration is no longer exposed by `Stream`.

Host map guards prevent conflicting device use. Graph contracts infer hazards;
independent lanes can overlap within configured run capacity. Completion handles
support waits and futures. A timeout does not cancel accepted native work or
release its storage. Failed retirement or teardown quarantines the ownership
chain. Submission hot paths reuse prepared native commands and buffers.

Memory budgets cover declared buffers, model reservations, and executable image
charges. They do not represent process RSS or driver/private queue overhead.
Private workspace requirements must be included in a model reservation.

Compiled code is trusted native code: loading and defining contracts are unsafe.
Bounds checks validate image structure and binding ranges, not arbitrary program
behavior. Do not load untrusted GPU/NPU programs.

## Runnable validation

```sh
bash scripts/test-npu-hardware.sh
cargo run --release --features npu --example shared_roundtrip
cargo run --release --features npu --example shared_bench
cargo run --release --features npu --example gemm_pipeline
cargo run --release --features npu --example gpu_npu_parallel -- 16777216 1024 21 trace.json
```

[`gpu_npu_parallel`](../examples/gpu_npu_parallel.rs) runs two independent
branches in one prepared graph: the GPU transforms an integer vector while the
NPU multiplies a batch of 8x8 BF16 matrices into FP32 outputs. Disjoint buffers
let the scheduler run both device lanes concurrently; waiting on the graph joins
both branches. The NPU kernel is included, with no external model or SDK needed.

The arguments are GPU element count, NPU matrix count, paired sample count, and
an optional Chrome/Perfetto trace path. Defaults are 16,777,216 elements, 1,024
matrices, and 21 samples. The example checks every result against independent
CPU calculations, changes inputs across replays, and compares sequential and
parallel completed execution in alternating order. Compilation, initial upload,
and result readback are outside the timed regions. It checks that replay adds no
native allocations or imports. Trace intervals are host-observed synchronized
device regions, not hardware timestamps. Small workloads can be slower in
parallel because scheduling overhead dominates; no speedup is assumed.

On the gfx1151/NPU5 development host (2026-09-22), one default 21-pair run
measured 0.754 ms sequential versus 0.610 ms parallel (1.24x). Every result
matched exactly after changed inputs. Batches of 1, 17, 1,024, and 16,384 also
passed; the NPU-dominated largest-batch case showed essentially no speedup.

The default GEMM example runs an included 8x8 BF16/BFP16 native matrix fixture
between GPU preprocessing and a GPU epilogue, including concurrent requests.
It is a correctness and scheduling example, not a large model GEMM benchmark.
`tests/native_execution.rs` separately checks nonuniform matrices against a CPU
oracle and mixed C++/Loom NPU workers with changed inputs. Launch buffer lists
must contain only used bindings: the pinned compiler's unused-middle-binding
relocation is rejected by the native image validator.

## Migration from 0.7

Compiler API callers must replace boolean report arguments with `ReportMode`,
serialize `CompileReport` with serde_json for text output, and include
`processor_mode` when constructing `CompilerOptions` without struct defaults.
