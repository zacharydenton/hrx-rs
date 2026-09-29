# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`fbbf3003121cce0322c771345505979809c84165`; `base-revision` is the machine-readable
pin. `native/release-inputs.json` records the source archive and each active
patch's SHA-256. The public upstream ABI remains the boundary, with the typed
processor-mode extension described below.

- `0001-vopd-source-cache-banks.patch` fixes AMDGPU VOPD source-cache bank
  constraints. FMAMK addends use hardware SRC2 even though they occupy encoded
  VSRC1. Allocation and emission check the actual source cache, retaining legal
  dual instructions. Generator and native assembly regressions cover the rule.
- `0005-materialize-encoding-config.patch` resolves configured i4/i8 encodings
  into concrete encoding definitions before native emission. It includes
  canonicalization and matrix-fragment regressions.
- `0009-amdgpu-profile-processor-mode.patch` retains the typed default/CU/WGP
  extension across profile specialization and serialization, uses the matching
  occupancy domain, and emits consistent native and assembly descriptors.
  Explicit modes require GFX11/GFX12. The rebase uses upstream's new kernel
  emission path and metadata-owned descriptor interface. Its unsupported-mode
  diagnostic is now AMDGPU_052, preserving upstream's new AMDGPU_051 diagnostic.
- `0011-loop-carried-accumulator-reuse.patch` retains whole-tuple back edges,
  records per-unit last uses within semantic segments, and prefers a concat's
  eventual edge destination when reserving registers. The rebase uses upstream's
  retained storage-component query and indexed loop-edge conflict search;
  it does not restore the removed recursive coalescing walk or fixed-storage
  scans. Storage extending beyond semantic segments remains conservative.
  The patch includes liveness, tuple-decomposition, and assembly regressions.
- `0012-retain-rdna35-lds-overlap.patch` retains the previously qualified gfx1151
  LDS overlap policy. Completion latency and source hazards remain active while
  independent loads can remain in flight. Width-specific classes required by
  generic address lowering are retained without their bandwidth reservation.
  Requalify this local performance policy when updating the compiler or target.

- `0013-retain-value-domain-arena.patch` fixes the 0.8.14 compiler crash in
  hrxdb merge selection. The local value domain owns its acquisition arena;
  symbolic queries cannot grow its retained value-ID list in temporary storage.
  Internal registration no longer accepts a caller-selected arena. A native
  regression destroys the query arena before reading and releasing the domain;
  `tests/compiler_sources.rs` also compiles the failing hrxdb kernel without a GPU.

Earlier SMEM, GFX11 VMEM-source reuse, dependent-inline-type and allocation-layout
patches are no longer in the active set. Only the six files above are applied.
Historical performance measurements do not establish performance of a new pin.
The matching runtime is published as `native-20260929-fbbf300312-fix1` and selected
by `bundle.json`. Use `HRX_RUNTIME_DIR` to select a local rebuild.

## Rebuild

Use `scripts/fetch-native-inputs.py` and `scripts/rebuild-hrx.sh`, documented in
[native/RELEASE.md](../../native/RELEASE.md). For a development checkout,
`LOOM_SOURCE=/path/to/clean/pinned/source bash scripts/apply-loom-patches.sh`
applies the active compiler set. `scripts/build-amdf.sh SOURCE BUILD` then applies
the native build, queue and cache-policy patches and builds all three libraries.
The compiler patches must be applied in filename order.

## Validation at this pin

The 2026-09-29 integration built all three native libraries with Clang 21 on
Ubuntu 26.04. Regenerating the Rust bindings produced no semantic changes.
All eight compiler/native patches applied to the pinned archive with zero fuzz;
the resulting patched files exactly matched the build source.

- `cargo test --all-targets`: 109 passed; hardware-dependent tests stayed ignored.
- `compiler_sources` with `--ignored --test-threads=1` against the new compiler:
  all seven passed, including C/C++ import and offline XDNA compilation.
- Descriptor timing, VOPD tables and occupancy Python suites: 35 passed.
- Native live-range, unit-liveness, target-constraint, occupancy and AMDGPU C API
  test executables: all five passed.
- The six patched `.loom-test` fixtures: all 85 cases passed.

Native regression tests used additional AMDGPU targets and `loom-check-test`
to include test descriptors. Release libraries use the original gfx1151/XDNA
configuration.

Release qualification also passed all 93 ignored hardware/compiler cases with
NPU enabled, including the shared queue pool, large graph batching, argument
arena and GPU/NPU execution checks. The fabric suite used an explicit
`HRX_AMDF_LIBRARY` path; its Busy assertion now runs before completion polling
can retire the new dispatch. A timing-only DAG assertion narrowly missed its
threshold on the first run and passed on rerun.

The paired compiler corpus passed numerical checks for all nine attention,
GEMM and convolution cases (30 alternating pairs per case) against the previous
published compiler. Timing confidence intervals were wide on this shared host;
no statistically clear regression was detected, and these results do not
establish general throughput or latency improvements. Release CPU checks,
clippy, documentation and the feature matrix also passed. Builds and test
processes used bounded memory with one Cargo job.

### 0.8.15 lifetime regression

The same Rust executable with hrxdb `e06d1aa6` succeeds on all four reported
shapes with the 0.8.13 compiler, segfaults with the 0.8.14 compiler, and succeeds
with patch 0013. The new Rust compile-only regression also segfaults against the
released 0.8.14 library and passes with the fixed library. The preceding 0.8.14
qualification did not include this merge-selection path.
