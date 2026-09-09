# Changelog

## Unreleased

- Kernel dispatch and graph recording reject workgroup dimensions that disagree
  with compiled export metadata. Compatibility initialization failures return errors.
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
- Restrict opaque-argument allocation scans to the current stream. Avoid clearing
  unused dispatch bindings and remove the intermediate HSACO byte vector.
- `compat::Device::open` retains its default-target scan. `open_for_target(target)`
  selects an architecture and `open_device(index)` selects a GPU by index.
- Device architecture reports are normalized before the first feature separator
  (`:`). Explicit `Target` values require bare architecture keys.
- Remove public raw FFI modules and duplicate aliases. ExportInfo remains available
  at the crate root. Replace `ffi_call!(...)` with `ffi::boundary(...)`.
- Remove internal review documents from the repository and allowlist package files.
- Disable publication pending native provenance, notices and anonymous availability.

### Earlier prerelease API changes

`Error` is a non-exhaustive enum
(`Error::Message` replaces the tuple constructor), `Args` is `Clone` without
`Copy`, `Submission::is_complete` needs a mutable token, and `FixedSequence` no
longer has a lifetime parameter. `Gpu::dispatch` is deprecated; migrate to
`Stream::dispatch` with explicit `Constants`. `loomrun` now respects scalar flag
widths exactly; use `--i64` for 64-bit Loom indices instead of relying on inferred
widening of `--i32` arguments. Minimum Rust version: 1.88.
