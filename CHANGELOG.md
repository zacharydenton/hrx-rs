# Changelog

## Unreleased

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
- Disable publication pending native provenance, notices and anonymous availability.

### Earlier prerelease API changes

`Error` is a non-exhaustive enum (`Error::Message` replaces the tuple constructor).
`Submission::is_complete` needs a mutable token, and `FixedSequence` no longer has
a lifetime parameter. `loomrun` respects scalar flag widths exactly; use `--i64`
for 64-bit Loom indices. Minimum Rust version: 1.88.
