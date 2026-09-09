# Changelog

## 0.2.0 (unreleased)

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
  is not what the runtime requires and costs about 0.95 us per needless edge --
  64 tiny fill nodes replay in ~151 us chained and ~88 us independent on gfx1151.
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
  the sweep tracks last use. Nothing evicts implicitly.
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
