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
`CxxSource::fp_model` defaults to `CxxFpModel::Precise`; select `Fast` to permit
the complete Loom fast-math contract. This replaces the former
`approximate_functions` flag. For approximate functions without other fast-math
permissions, use the source facade's `loom::scalar::approximate` operations.

The updated embedded `<loomcxx/kernel.h>` exposes subgroup scans, workgroup
scans and workgroup reductions through ordinary `CxxSource` import. For example,
`loom::kernel::workgroup::scan::addi<loom::kernel::scan_mode::exclusive>(count)`
computes each workitem's prefix offset. Inclusive/exclusive and forward/reverse
modes are supported; every required participant must converge. These value
collectives do not order unrelated memory accesses. Native u32/f32 workgroup
operations are supported; u64 workgroup collectives remain target-dependent.
The compiler also gains exact dynamic integer division on AMDGPU, improved BF16
narrowing, and source-level native-math alternatives. These are available through
the existing importer and Loom source APIs without additional Rust switches.

For repeated imports sharing leading includes, retain a semantic preamble:

```rust,ignore
let mut source = CxxSource::new("kernel.cpp", contents);
source.headers.insert("common.h".into(), header_contents);
source.preamble = Some(compiler.cxx_preamble(source.clone())?);
// Clone source and vary the body after its leading includes for subsequent imports.
let module = compiler.import_cxx(source)?;
```

Preambles are immutable and can be shared across threads. They own copied
frontend state and keep the native library alive. They are process-local and
omitted from source serialization and cache identity; changing headers or
frontend configuration falls back to cold parsing. Import roots and math policy
may vary. `source_identifiers` maps physical virtual paths to logical diagnostic
identities without changing include lookup; mappings participate in source identity.

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

`Specialization::with_listing(true)` requests target assembly;
`Artifact::listing()` returns the text. Listings have their own specialization
identity and a verified cache companion. Missing or corrupt companions trigger
recompilation. A target without listing support returns an error when requested.

The compile CLI supports:

```text
--manifest-output=FILE
--launch-output=FILE
--listing-output=FILE
--fp-model=precise|fast
--source-identifier=PHYSICAL=LOGICAL
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
