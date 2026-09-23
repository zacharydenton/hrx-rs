# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`468508b9e27e749972382f78e1c67d6db7bec27e`; `base-revision` is the machine-readable
pin. The patches use the upstream C API without adding private ABI entrypoints.

`0001-vopd-source-cache-banks.patch` fixes AMDGPU VOPD register-bank constraints.
FMAMK addends use hardware SRC2, whose bank mask is 1, even though they occupy
the encoded VSRC1 field. Treating that field as SRC1 (mask 3) admits illegal
dual instructions and corrupts vision GELU outputs. Allocation and native
emission now check the actual source cache, including mixed FMAMK/FMAC pairs;
legal dual instructions remain enabled. The patch includes generator tests and
native assembly/placement regression fixtures and retains upstream license
headers. It originated in fork commit
[`675cc43bc`](https://github.com/zacharydenton/hrx-system/commit/675cc43bc).

`0003-smem-storage-reuse-drain.patch` fully drains SMEM before overwriting pending scalar-load
destination registers. A partial wait could mark a pointer load complete while it was
still outstanding, allowing its destination to be overwritten. The patch has a
deterministic assembly regression and an optional 34-line Loom GPU reproducer.
The reduced kernel did not fault reliably on hardware; use the assembly regression
for a deterministic check. The isolated upstream fix is
on fork branch `fix/loom-smem-storage-reuse`, commit
[`aa5f5c66a`](https://github.com/zacharydenton/hrx-system/commit/aa5f5c66a).

`0005-materialize-encoding-config.patch` resolves configured i4/i8 encodings
into concrete encoding definitions so generic kernel families specialize through
native emission.

Patch 0006 (dependent inline types) has been removed: upstream now substitutes
arguments during materialization. All five original regression cases pass with
the unmodified compiler at this revision.

`0007-gfx11-vmem-source-reuse.patch` removes memory-completion leases for GFX11
VMEM address/resource SGPR sources. Hardware interlocks their consumption;
reusing an address does not require waiting for the loaded value. Result-write
leases and the SMEM fix remain active. A minimal gfx1151 assembly test reuses
the address immediately and waits before consuming the destination. It fails
before the fix and passes afterward. The isolated fix is on fork branch
[`fix/loom-gfx11-vmem-source-reuse`](https://github.com/zacharydenton/hrx-system/tree/fix/loom-gfx11-vmem-source-reuse),
commit [`beaff74b2`](https://github.com/zacharydenton/hrx-system/commit/beaff74b2).

`0009-amdgpu-profile-processor-mode.patch` adds a typed AMDGPU profile
extension for default, CU, or WGP execution. The policy survives target
specialization and module serialization, selects the occupancy domain, and
sets the native and assembly kernel descriptors consistently. Explicit modes
are supported on GFX11/GFX12. Older libraries reject the nonempty extension
chain instead of silently ignoring it. The patch includes native profile,
serialization, and occupancy tests plus generator validation.

`0010-retain-qualified-allocation-layout.patch` retains the preceding masked
restore-block placement and structural-source-first allocation for linear
register files. Named physical register views retain consumer-first allocation.
The two newer policies interact to add loop copies and earlier waits, slowing
ArcFace by 32% and SCRFD by 11%. Keeping both qualified policies restores the
released timings; changing either alone is insufficient. Physical register
views, semantic live-segment alias checks, and all newer completion fixes remain
active. Three exact-reference convolution shapes extend the paired corpus.

The candidate applies six compiler patches. Optional patch 0007 is retained after
paired hardware qualification. Patches 0002 and 0004 are preserved under
`patches/retired` but are not applied; 0008 belongs to the removed legacy runtime.

## Rebuild

Use `scripts/fetch-native-inputs.py` and `scripts/rebuild-hrx.sh`, documented in
[native/RELEASE.md](../../native/RELEASE.md). For a development checkout,
`LOOM_SOURCE=/path/to/clean/pinned/source bash scripts/apply-loom-patches.sh`
applies the active compiler set. `scripts/build-amdf.sh SOURCE BUILD` then applies
the reviewed native build, queue and cache-policy patches and builds the three native libraries.
The manifest records every applied patch digest.
