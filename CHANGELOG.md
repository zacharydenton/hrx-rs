# Changelog

## 0.8.4 — reuse budgeted native commands

- Reuse dispatch and transfer commands for buffers owned by budgeted streams.
  Dropping a buffer evicts its commands before releasing backing; queued work
  retains its budget charge until completion. Foreign-stream bindings remain
  uncached.
- Bound dispatch reuse by 1,024 entries and 16 MiB of private argument, fence
  and scratch storage. This accommodates eager model pipelines without keeping
  arbitrarily large native scratch allocations resident.
- Retain the qualified 0.8.3 native bundle unchanged.

## 0.8.3 — restore cached GPU system memory

## 0.8.2 — release tracked backing with its budget

- Initialize fresh tracked GPU storage through its coherent native mapping.
  The shared allocation stream no longer retains the last allocation through
  an initialization completion fence after the tracked owner is dropped.
- Keep tracked buffers out of the uncharged stream allocation pool, so releasing
  tracked storage also releases its native backing.

## 0.8.1 — bounded allocation queues

- Reuse one lazy allocation stream per tracked runtime. Live tensors previously
  retained a separate native GPU queue each, causing large SCRFD inference plans
  to fail with `gpu.user_queue_create` status 28.
- Reuse that stream for code loading and device validation as well. Graph
  execution queues and cross-lane scheduling are unchanged.
- Qualify 512 simultaneously live allocations, SCRFD's complete InsightFace
  fixture, and the GPU/NPU integration suites on gfx1151 and NPU5.

## 0.8.0 — native GPU/NPU migration

## Prior upstream update (0.7 development)

- Select native release `native-20260922-hrx-update` for GPU execution and Loom
  compilation; retain the existing NPU bundle.

- Update the native HRX/Loom source pin to upstream `556c648e8` (2026-09-22),
  rebase the downstream patches, pin a newer ROCr exporting `hsa_amd_queue_create`,
  and remove patch 0006 after its five regression
  cases pass unmodified upstream. Adapt dma-buf export to the new buffer API.
- Move the local CU/WGP extension descriptor from 41 to 42 to avoid upstream's
  C++ importer descriptor. Existing locally patched compilers must be rebuilt
  for explicit modes. The new pinned native bundle includes this extension.

- Add `loom::CompilerOptions::processor_mode` for AMDGPU CU/WGP scheduling,
  with compiler and artifact caches separated by mode. Struct literals must add
  the field or use `..Default::default()`. Explicit modes require compiler patch
  0009, now included in the pinned compiler. Default-mode cache keys are preserved.

- Let the upload staging cache replace completed small allocations when larger
  uploads arrive. Retain the eight-entry and 64 MiB limits while avoiding
  repeated allocation after a workload changes its upload sizes.

## 0.7.1 — 2026-09-17

- Add audited graph-local private scratch reuse. Model IO and weights remain
  distinct; hazards order workspace reuse and separate slots remain independent.
- Add runtime-geometry packed RGB affine sampling for mixed-resolution batches,
  with checked extents, explicit sampling contracts and reusable graph bindings.
- Add direct host-slot publication callbacks, avoiding capacity-sized temporary
  buffers for variable-length packed inputs.
- Expose directly bound row-gather fragments for resident cross-model graphs.
- Remove the model-specific vision pipeline and Rust benchmark crates so this
  repository has no Cargo dependencies on downstream model crates. Model
  composition belongs in downstream applications using the generic runtime APIs.

## 0.7.0 — 2026-09-17

- Add validated model fragments that record into caller-owned execution graphs
  with directly bound intermediate tensors and shared immutable weights.
  Normalization, patchification, resize, affine sampling and similarity fitting
  expose the same composition API.
- Bind prepared inference slots to their actual model storage, including sliced
  and in-place IO, without duplicate model input/output allocations or copies.
  Validate runtime identity, consistent descriptors and independent slot IO.
- Allocate host upload/readback staging lazily, reuse it across submissions and
  allow retries after allocation-budget failures.
- Preserve host-visible storage in model fragments and map host-visible IO
  directly, including sliced tensors. Only device-local IO needs transfer
  staging; guarded host access retains allocation-wide synchronization.
- Add a standalone single-image DINOv3/SCRFD/ArcFace benchmark with stage
  timings, replay checks, runtime counters and machine-readable results.

Migration: `PreparedModel::prepare` takes a context, capacity and slot factory
returning `InferenceGraph { inputs, outputs, graph }`. Allocate independent
writable storage inside each factory call. No compatibility shim is provided.

This release provides graph-composition foundations, not completed cross-model
optimization or a general end-to-end performance guarantee.

## 0.6.0 — 2026-09-16

- Add shared model contexts, checked owned device tensors, bounded prepared
  inference slots, producer dependencies, reusable image/tensor operations,
  shape-plan caching, and budgeted idle-LRU model residency. Native GPU and NPU
  allocations can share the same ceiling and retain charges through queued work.
- Schedule upload, compute and download independently with cross-lane memory
  hazards; add bounded host-observed execution tracing and inference job pools.
- Add `execution::NativeSession` for synchronous stream-bound models with
  borrowed inputs/callbacks: bounded compute-lane admission, completion fences
  on errors/panics, and full-owner quarantine when completion is uncertain.

- Raise the supported Rust minimum to 1.91, matching the current Hugging Face/Xet
  dependency graph; the previous 1.88 claim no longer builds with the lockfile.

- Keep `npu` as the only optional Cargo feature. GPU execution, Loom compilation,
  downloads and the CLI are unconditional; NPU compilation and probes use `npu`.
  Remove the `loom`, `download`, `runner`, `npu-compile` and `npu-probe` flags
  without aliases. Native libraries remain lazily loaded and offline policy is
  unchanged.

## 0.5.1 — 2026-09-16

- Accept zero-length dimensions in ONNX initializers and tensor attributes.
  This restores INSwapper loading: its unused Resize ROI initializer has shape
  `[0]`. Negative dimensions and inconsistent tensor data remain errors, and
  rank-zero tensors retain scalar semantics.
- Add serialized ONNX regression coverage for empty float32/int64 tensors,
  tensor attributes, malformed dimensions/data, and scalars.

## 0.5.0 — 2026-09-16

- Add `artifacts::hf` for local-first Hugging Face resolution with offline,
  progress, revision, and checksum policy; add owned and memory-mapped
  `artifacts::safetensors` indexing; and add checked `artifacts::onnx` model,
  node, tensor, shape, axis, stride, and broadcast inspection.
- Move the resident inference API to `hrx::model` and remove the old
  `hrx::loom::model` path. `Specialization` fields are private and configured
  through builders and accessors; no compatibility re-exports are retained.
- Add `AccessGraph` for byte-range dependency inference in low-level GPU graphs,
  a bounded RAII `BufferPool`, reusable `ScratchPlanner`, offset transfer and
  zeroed-allocation helpers, and shared compiler selection by stream or target.
- Add reusable benchmark distributions and stage timing, including cumulative
  stage totals. ArcFace, SCRFD, DINOv3, H3, Krea2, and hrxdb now consume the shared
  artifact, memory, compiler, and timing facilities instead of carrying their
  own implementations.

Migration: import resident inference from `hrx::model`; construct and inspect
`Specialization` with `new`, `set_config`, `replace_config`, `set_symbol`,
`set_report`, `configuration`, `symbol`, and `report_requested`. ONNX and
SafeTensors parser types are intentionally not exposed.

## 0.4.1 — 2026-09-16

- Add `loom::model::ModelSession`, hoisting the resident buffer/kernel/graph
  engine shared by the vision-model consumers. Session-scoped region and kernel
  handles reject cross-session use; explicit binding access drives reusable
  graph dependencies at byte-interval granularity.
- Reuse coherent or device-local storage through one upload/readback path,
  batch multi-output downloads behind a reusable readback allocation, and
  provide common graph-versus-direct timing distributions and a public
  nearest-rank `benchmark::percentile` helper.
- Share the interval-frontier dependency implementation between coordinated
  execution and resident model graphs, and select the Loom compiler profile
  from the opened device instead of assuming the default architecture.

## 0.4.0 — 2026-09-10

- Preserve compiler limits, reporter state and concurrent batch retries alongside
  the keyed kernel cache. Keyed pending requests support warm lookup without source
  hashing while retaining batch compilation.
- Add coherent host-visible coordinated storage, consuming GPU buffer/kernel
  adoption, and scoped Stream handoffs with completion and quarantine handling.
- Infer graph dependencies from interval frontiers and compact access summaries,
  avoiding dense dependency lists when scratch allocations are reused.
- Preserve the installed Chess environment and wrapper precedence when pinning
  a compiler toolchain.
- Serialize direct kernel publication with pending batches to avoid duplicate loads.
- Test published GPU/NPU bundles separately from local native development builds.

Migration: exhaustive matches on `MemoryPlacement` must handle the new
`HostVisible` variant. This variant supports host/GPU transfers without an NPU.

## 0.3.0 — 2026-09-10

- Add `execution::Runtime`, explicit GPU/NPU/shared storage, checked kernel
  contracts, guarded host mappings, inferred hazards, and bounded prepared graphs.
  Completion supports blocking waits and executor-neutral futures.
- Replace the old NPU shared-memory wrapper with retained dma-buf imports and
  a separately provisioned ABI shim. Remove the sibling Cargo dependency.
- Add pinned IRON/AIE subprocess compilation, verified specialization caching,
  native diagnostics, hardware examples, Miri checks and NPU CI qualification.
- Pin a rebuilt GPU bundle with interop ABI 1 and a separate relocatable NPU
  runtime containing XRT and its XDNA plugin. With `npu` enabled, `hrx prepare`
  provisions both; kernel drivers and firmware remain host prerequisites.
  Bundles target Ubuntu 26.04 LTS (glibc 2.43). See `docs/GPU-NPU.md`.
- Add an independent SCRFD + DINOv3 throughput benchmark with pinned sample
  provenance and explicit measurement limitations.

## 0.2.0 — 2026-09-09

- Graph drop waits for its last replay's immutable completion point, without
  flushing unrelated stream work. A failed launch prevents further replay and
  retains the native graph when completion cannot be established.
- Correct the independent graph benchmark to use disjoint allocations and
  interleave its two schedules. Document the native cost of explicit joins.

- Compiled kernels use one machine-wide cache. `Module::compile` and
  `Compiler::compile_all` no longer take a cache path: artifacts are
  content-addressed, so a per-caller location could only duplicate identical
  bytes and hide them from `hrx gc`. `bundle::kernel_cache()` names it.
- `hrx gc` evicts by access time rather than by a timestamp inside one file
  layout, so entries written by an older release are dated like current ones
  instead of being deleted regardless of the age given on the command line.

- Drop `HRX_CACHE_DIR`, `KREA2_RUNTIME` and `IREE_HAL_AMDGPU_LIBHSA_PATH`. The
  cache follows the XDG Base Directory specification alone — `$XDG_CACHE_HOME/hrx`,
  else `$HOME/.cache/hrx`, ignoring a relative value as the specification requires.
  `KREA2_RUNTIME` was a dead alias from the removed compatibility module, and HSA
  is loaded from the same verified directory as libhrx, selected with
  `HRX_RUNTIME_DIR` rather than a second variable in another project's namespace.

- Buffers are bound to their device, not to the allocating stream. Any stream on
  that device may transfer, fill, copy or dispatch against one; order conflicting
  access with `record_event`/`wait_event`. This removes `Buffer::share_on`, which
  existed only to work around the old check, and the sticky flag that kept shared
  allocations out of scratch pools. A scratch pool still belongs to one stream.
- `Stream::dispatch`, `fill` and `copy` take `&self`. They read only handles
  behind the stream's `Arc` and mutate nothing; `&mut` was a pure exclusion
  marker that forced callers to hoist intermediates. Transfers, `submit`,
  `synchronize`, `scratch` and `recycle` still take `&mut`.
- `Kernel` is `Clone`, retaining the native executable, so caches need no `Arc`.
- `Diagnostic::severity` is a `Severity` enum (`Note`, `Warning`, `Error`) instead
  of a bare `u32`, so callers can filter backend remarks from real errors without
  hardcoding native values. Unknown native severities map to `Error`.
- Replace `SequenceBuilder` with `Graph`, a real dependency graph. `Stream::graph`
  and `Stream::launch` replace `sequence`/`launch_sequence`, `FixedSequence`
  becomes `GraphExec`, and `fill`/`copy`/`dispatch` take a leading `after: &[Node]`
  and return a `Node`. `join` records a dependency-only node for fan-in. Nothing
  is implicit: the old builder chained every operation to the previous one, which
  is not what the runtime requires. Dependency costs depend on the barriers
  and partitions they produce; an independent-node benchmark is not a per-edge price.
  A dependency can only name an already-recorded node, so a graph is acyclic by
  construction and instantiation stays on the runtime's linear fast path;
  duplicate entries in one list are collapsed rather than rejected. Resolving a
  dependency list of 16 or fewer nodes, including the dedupe, allocates nothing.
- Document that neither `ExportInfo` nor the compiler report carries per-slot
  scalar types, so mixed-width constants cannot be built by construction and each
  width must come from the declaring source.

- Add `Compiler::compile_all`, which runs a batch of specializations across
  `CompilerOptions::workers` workspaces and returns results in request order.
  `Module::compile` blocks, so that option previously did nothing unless the
  caller built its own thread pool; the compiler is the one place that knows how
  many workspaces it can afford.
- Rename the draining `read` to `read_blocking` and the queued `read_queued` to
  `read`, so the bare name means queued on both sides of a transfer. Previously
  `upload(..); read(..)` looked symmetric while silently draining the stream.
- Loom compile errors now lead with the first error, add a hint for a generic
  target used with hand-written asm, and report the count of cascading errors
  instead of printing them. The full list stays in `Error::Compile::diagnostics`.
- Add `hrx gc [DAYS]`, removing runtime bundles that `bundle.json` does not pin
  and kernel artifacts unused for longer than DAYS (default 30). Provisioning
  published but never evicted; cache hits now refresh an artifact's timestamp so
  the sweep tracks last use. Collection preserves installation and compilation
  staging directories and locks each artifact before checking its age and
  removing it. Nothing evicts implicitly.
- Compile the README as a doctest, so an API change that invalidates its example
  fails the build. Document that the `hrx::Target` profile and the source-level
  `amdgpu.target<...>` are chosen independently, and that only the latter accepts
  generic families.
- Replace the `loomrun` binary with an `hrx run` subcommand. It was a four-line
  shim over the same entry point, kept only for name compatibility with a C++
  tool that no consumer in this repository invokes any more. `runner::main()`
  becomes `runner::run(&argv)`; usage errors still exit 64.

## 0.1.0 — 2026-09-09

- Transfers, fills and copies now take `View` regions. `View::slice` checks a
  subregion relative to its parent; `offset` and `owner` expose its allocation origin.
  Stream copies require equal, nonempty views, matching sequence copies.
- Rename the synchronous `upload` to `upload_blocking`, and `upload_queued` to
  `upload`. Both write at the start of the destination view. Replace a buffer and
  offset with `buffer.binding().slice(offset, length)?`.
- `Constants::push` returns `Result<()>`; append each scalar in a separate call.
- `hrx pack` refuses an inventory whose `status` is not `"complete"` or that leaves
  any component's license unconfirmed, so an unfinished review cannot be packaged.
- Add docs.rs metadata and a publication checklist in THIRD-PARTY.md.
- Ship complete `THIRD-PARTY.json`, `NOTICE`, license texts, and corresponding
  sources. Replace the Arch library chain with a coherent AMD runtime build.
- Drop the unused `loom-compile` executable from the release bundle. HRX compiles
  in process, so no compiler executable is needed.
- Remove `Gpu`, the `compat` and `ffi` modules, their features, and the bytemuck
  dependency. `Stream` owns execution directly; dispatch requires explicit
  `Constants`. The model C-ABI verification script and the allocator benchmark
  superseded by the stream benchmark are removed with them.
- Add a release benchmark for stream uploads, dispatch, graph replay and scratch reuse.
- Kernel dispatch and graph recording reject workgroup dimensions that disagree
  with compiled export metadata.
- Device architecture now drives executable loading. `CompilerOptions::target`
  selects the Loom profile and artifact cache target; `Device` is Clone, not Copy.
  `TARGET_FAMILY` is now a CStr constant.
- Add immutable completion events and explicit unsafe sharing of buffers between
  streams on the same device. Callers must order conflicting accesses.
  Shared allocations, including their original handles, cannot enter scratch pools.
  This restriction persists after aliases are dropped.
- Batch queued uploads until staging pressure, reuse larger staging allocations,
  and release unmapped scratch even after failed synchronization.
- Bound cached modules with `CompilerOptions::module_cache_capacity` (default 64).
  Struct literals for compiler options should use `..Default::default()`.
- Avoid clearing unused dispatch bindings and remove the intermediate HSACO byte vector.
- Device architecture reports are normalized before the first feature separator
  (`:`). Explicit `Target` values require bare architecture keys.
- Remove public raw FFI modules and duplicate aliases. ExportInfo remains available
  at the crate root.
- Remove internal review documents from the repository and allowlist package files.
- Enable publication after verifying the public native release and its source archive.
- Anchor package file patterns to the repository root so local native build
  artifacts cannot enter the published crate.

### Earlier prerelease API changes

`Error` is a non-exhaustive enum (`Error::Message` replaces the tuple constructor).
`Submission::is_complete` needs a mutable token, and `FixedSequence` no longer has
a lifetime parameter. `loomrun` respects scalar flag widths exactly; use `--i64`
for 64-bit Loom indices. Minimum Rust version: 1.88.
