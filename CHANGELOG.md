# Changelog

## 0.2.0 (unreleased)

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
