# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`7e9c7bbd5e93d20c1e1a64c60bb4644499404f9b`; `base-revision` is the machine-readable
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
  diagnostic is AMDGPU_052. The public extension uses structure ID 44;
  upstream now owns IDs 42 and 43 for artifact compilation and pass tracing.
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
- `0014-materialize-bounded-index-extrema.patch` materializes signed 32-bit
  `index.min`/`index.max` operands whose kernel ABI carriers remain 64-bit.
  Both inputs must independently fit signed 32 bits; a narrow result is not
  sufficient. Contract tests check that proof, and the bounded-extrema GPU
  regression checks tile boundaries and untouched output guards. This fixes
  H3 video-encoder convolutions without changing their source or fixtures.

- `0015-retain-partial-incoming-vmem-leases.patch` retains the uncovered units
  of an incoming wide VMEM result when a younger narrow VMEM load reuses only
  part of its registers. Later ALU writes must still wait for the original load.
  A fixed-register compiler regression checks the partial wait, and H3 decoder
  differentials verify the existing goldens without fixture changes.

- `0016-honor-polled-feedback-and-active-masks.patch` honors the native feedback
  ABI's null notification signal and intersects vector comparison masks with
  active EXEC before scalar tests. Without the latter, an undefined wave32
  high word can admit reservations past capacity and overwrite reports. Authored
  lowering cases cover wave32/wave64; GPU tests verify bounded drops and polling.

Earlier SMEM, GFX11 VMEM-source reuse, dependent-inline-type and allocation-layout
patches are no longer in the active set. Patch 0011 is also retired: upstream now
owns CFG unit-use indexing, storage leases and concat placement. The old
per-segment liveness overlay is not carried into those analyses. Only the eight
files above are applied. Historical measurements do not establish performance
of this compiler pin.

The matching native archive is staged as `native-20261008-7e9c7bbd5e`.
Rust bindings require this compiler's current C API, including
`loomc_compile_artifact` and its diagnostic layout. Older compiler bundles are
not supported. No private diagnostic-layout probe is added to Loom.

## Rebuild

Use `scripts/fetch-native-inputs.py` and `scripts/rebuild-hrx.sh`, documented in
[native/RELEASE.md](../../native/RELEASE.md). For a development checkout,
`LOOM_SOURCE=/path/to/clean/pinned/source bash scripts/apply-loom-patches.sh`
applies the active compiler set. `scripts/build-amdf.sh SOURCE BUILD` then applies
the native build, queue and cache-policy patches and builds all three libraries.
The compiler patches must be applied in filename order.

## Validation at this pin

See the native refresh entry in [CHANGELOG.md](../../CHANGELOG.md) for completed
checks. Local build and qualification logs are under
`artifacts/native-refresh-20261008/`.
