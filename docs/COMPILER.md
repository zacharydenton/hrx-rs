# Loom compilation

`hrx::loom::Compiler` compiles Loom in process and caches artifacts.
`Compiler::for_stream` selects the stream's device profile;
`Compiler::for_target` selects an explicit profile. Matching compiler instances
are shared. See [native setup](../README.md#install-the-cli) and the
[compiler patch list](../patches/loom/README.md).

## Targets and batches

`hrx::Target` is the device/compiler profile, such as `gfx1151`. It accepts bare
architecture keys. The source-level `amdgpu.target<...>` also accepts generic
names such as `gfx11-generic`, but hand-written assembly needs a bare architecture
with a low-assembly contract.

`Compiler::compile_all` distributes a batch across `CompilerOptions::workers`
workspaces and returns results in request order. `Module::compile` blocks;
calling it serially does not use the batch concurrency.

`CxxSource` supplies named C23/C++26 translation units and virtual headers.
`Compiler::sources` links them with Loom modules. There is no implicit host
filesystem include search.

## Launch geometry

AMDGPU artifacts contain an executable manifest and a host launch program.
`artifact.launch_program(artifact.symbol())?` selects an export;
`evaluate(&workload_bits)` returns its grid, workgroup, subgroup, cluster and
storage requirements. Workload arguments follow the kernel definition's
signature and may differ from device scalar arguments.

`Kernel::launch_config`, `ModelSession::launch_config` and
`ModelDefinition::launch_config` evaluate and validate these requirements during
preparation. Each evaluator owns its scratch and retains the native library.

## Reports and traces

Request `ReportMode::Summary` or `Details` on a specialization. Reports include
compiler identity, target, backend and processor mode; unavailable resource
fields remain unknown. `CompileReport::expansions` exposes source-to-low-level
expansion choices. `hrx report show` and `hrx report diff` inspect saved reports;
comparisons require compatible identities. CU/WGP policies apply to AMDGPU only.

The compile CLI supports:

```text
--manifest-output=FILE
--launch-output=FILE
--trace-output=FILE --trace-format=jsonl
--trace-before=PATTERN --trace-after=PATTERN
--trace-max-bytes=N
```

Trace filters can be repeated. Tracing recompiles even on a cache hit.
Diagnostics include source context and named parameters; `Error::compiler_report`
retains the native report from a failed compilation.

## Sanitizers

`CompilerOptions::sanitizer` selects access, value, operation and race checks,
with default, trap or report-only behavior. CLI equivalents are
`--sanitizer=access,value,operation,race` and `--sanitizer-reporting=report-only`.
Instrumentation has separate compiler and artifact cache identities.

Load report-only artifacts through `Device::load_sanitized` on AQL, or
`Runtime::load_sanitized_gpu_artifact` with the AQL compute engine. Wait for
execution, then call `sanitizer_reports()` on the loaded kernel. Collection
returns `Busy` while work is pending, drains the channel, and reports overflow
in `dropped`. Reports retain source and predicate metadata. Ordinary loading
rejects artifacts that require a feedback channel.

Address checks cover complete bound allocations and executable storage.
Allocation starts must be 8-byte aligned. Gaps, partial tails and undeclared
allocations are poisoned; slices within an allocation do not create separate
address limits. Preparation rejects virtual-address spans that exceed
`SanitizerRuntimeOptions::maximum_shadow_bytes`.

Race checks cover workgroup-local memory. Reports identify both access sites,
widths, addresses and workitem coordinates. Each prepared dispatch owns shadow
storage sized for its grid and LDS; ordered clears reset it before replay.
Global-memory and cross-queue races are outside this check.

The shadow limit covers combined address/race storage. Feedback, shadow storage
and private instrumented code retain their memory-budget charges through native
use. Sanitizers detect specific failed checks; callers still own the native-code
safety contract.
