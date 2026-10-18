# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`61c152ac4b40dd758803a2cbcd237184a8dfb669`; `base-revision` is the machine-readable
pin. `native/release-inputs.json` records the source archive and each active
patch's SHA-256. The processor-mode patch extends the upstream ABI as described
below.

- `0005-materialize-encoding-config.patch` resolves configured i4/i8 encodings
  into concrete encoding definitions before native emission. It includes
  canonicalization and matrix-fragment regressions.
- `0009-amdgpu-profile-processor-mode.patch` retains the typed default/CU/WGP
  extension across profile specialization and serialization, uses the matching
  occupancy domain, and emits consistent native and assembly descriptors.
  Explicit modes require GFX11/GFX12. The rebase uses upstream's new kernel
  emission path and metadata-owned descriptor interface. Its unsupported-mode
  diagnostic is AMDGPU_053. The public extension uses structure ID 44;
  upstream now owns IDs 42 and 43 for artifact compilation and pass tracing.
- `0012-retain-rdna35-lds-overlap.patch` retains the gfx1151
  LDS overlap policy. Completion latency and source hazards remain active while
  independent loads can remain in flight. Width-specific classes required by
  generic address lowering are retained without their bandwidth reservation.
  Rerun performance tests when updating the compiler or target.
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
- `0017-bound-address-sanitizer-shadow-loads.patch` checks both application and
  shadow extents before each shadow lookup, including wide loads and wrapped
  addresses. Excluded lanes issue no memory load and receive full poison after
  restoring EXEC. Native lowering cases retain the bounds and lane-mask order;
  GPU regressions cover partial allocation tails, null pointers, wraparound,
  wide accesses, wave32/wave64, replay and combined address/race instrumentation.

Earlier SMEM, GFX11 VMEM-source reuse, dependent-inline-type and allocation-layout
patches are no longer in the active set. Patch 0011 is also retired: upstream now
owns CFG unit-use indexing, storage leases and concat placement. The old
per-segment liveness overlay is not carried into those analyses. VOPD source-cache routing and value-domain arena ownership are also
upstream now (commits `7a947dce16` and `490a9ac096`). Only the seven files
above are applied.

The matching native archive is built locally as `native-20261015-61c152ac4b`.
Rust bindings require this compiler's current C API, including
`loomc_compile_artifact`, link-index builder allocation/free, and the
precise/fast C++ import policy. Older compiler bundles are
not supported.

## Rebuild

Use `scripts/fetch-native-inputs.py` and `scripts/rebuild-hrx.sh`, documented in
[native/RELEASE.md](../../native/RELEASE.md). For a development checkout,
`LOOM_SOURCE=/path/to/clean/pinned/source bash scripts/apply-loom-patches.sh`
applies the active compiler set. `scripts/build-amdf.sh SOURCE BUILD` then applies
the native build, queue and cache-policy patches and builds all three libraries.
The compiler patches must be applied in filename order.

## Experimental Radeon VII bringup

`0018-gfx906-wave64.patch` adds an executable GCN5 descriptor family for gfx906,
with wave64, ordinary vector FP32 arithmetic, software BF16 conversion, and no
MFMA/AGPR capability. It uses the compatible subset of the vendor CDNA1 XML
with GCN5 exclusions and corrected fused mixed-precision instruction names.
The accompanying `native/amdf/gfx906.patch` enables the qualified native queue
path. Its legacy cache-control capability uses bit 3, avoiding upstream's newer
PRIV/TRAP_PRESENT bits 1 and 2. This does not upgrade HRX to that newer ABI.

Build the pinned source with `HRX_GPU_TARGETS='gfx906;gfx1151'`; retain gfx1151
in the target set when qualifying regressions on the original hardware.
Qualification on a Radeon VII has passed scratch-free native wave64 launches,
changed-input visibility checks, and authored Loom BF16 GEMMs with LDS staging
and ragged tails. The MOSS repository's `examples/qualify_gfx906.rs` carries the
kernel integration check. Scratch, comprehensive ISA coverage, device-local
transfer paths, and model performance remain unqualified. Do not interpret
these initial correctness checks as full GPU or inference support.
