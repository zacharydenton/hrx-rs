# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`556c648e8f301ad9656d325687cc93b417ea78ff`; `base-revision` is the machine-readable
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

`0002-amdgpu-fragment-repack.patch` selects a VGPR permutation path for
accumulator-to-RHS fp16 fragment repacking, with native regression coverage.
Upstream now also supports this operation through `ds_swizzle`; the patch
retains the existing downstream implementation.

`0003-smem-storage-reuse-drain.patch` fully drains SMEM before overwriting pending scalar-load
destination registers. A partial wait could mark a pointer load complete while it was
still outstanding, allowing its destination to be overwritten. The patch has a
deterministic assembly regression and an optional 34-line Loom GPU reproducer.
The reduced kernel did not fault reliably on hardware; use the assembly regression
for a deterministic check. The isolated upstream fix is
on fork branch `fix/loom-smem-storage-reuse`, commit
[`aa5f5c66a`](https://github.com/zacharydenton/hrx-system/commit/aa5f5c66a).

`0004-preserve-fragment-read-order.patch` keeps operand loads ahead of the
matrix fragments that consume them, preserving the source's latency hiding.
It includes current lowering expectations for scaled matrix operations.

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

This local compiler extension is included in `native-20260922-hrx-update`,
the runtime selected by `bundle.json`. Its descriptor type is 42 because
upstream assigned 41 to the C++ importer. Rebuild earlier locally patched
compilers before using explicit modes with these bindings. Default-mode cache
keys remain unchanged; compiler identity separates artifacts from each build.

## Rebuild the compiler

From this HRX checkout, with upstream's build prerequisites installed:

```sh
git clone https://github.com/ROCm/hrx-system.git ../hrx-system-patched
git -C ../hrx-system-patched checkout --detach "$(cat patches/loom/base-revision)"
export LOOM_SOURCE="$(cd ../hrx-system-patched && pwd)"
bash scripts/apply-loom-patches.sh
cd "$LOOM_SOURCE"
python3 dev.py --cmake-build-dir "$PWD/build" cmake setup
python3 dev.py --cmake-build-dir "$PWD/build" cmake configure \
  -DCMAKE_BUILD_TYPE=Release -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151 -DIREE_BUILD_TESTS=OFF \
  -DLIBHRX_BUILD_CTS=OFF -DHRX_INSTALL_TESTS=OFF
python3 dev.py --cmake-build-dir "$PWD/build" cmake build \
  loomc_shared loom-compile libhrx_src_libhrx_hrx
export HRX_LOOM_LIBRARY="$PWD/build/loom/binding/c/libloomc.so"
```

HRX compiles in process through `libloomc.so` and never runs a compiler
executable, so the release bundle does not ship one. The `loom-compile` target
above is built for local use only: it is useful for comparing artifacts during
compiler development. Compiler identity and artifact cache keys include the
shared library's content digest, so applying a compiler fix automatically
invalidates affected cache entries.

The published bundle records this base and every patch digest in `provenance.json`.

## Validation

See [VALIDATION.md](../../VALIDATION.md) for the tested bundle and crate results.
The patch files include the native regression fixtures described above.

## Upstream patch audit (2026-09-22)

The audit compares pristine upstream `556c648e8` with the rebased compiler.

| Patch | Result on pristine upstream | Decision |
| --- | --- | --- |
| 0001 VOPD banks | Three illegal source-cache pairings remain; a legal pairing is also missed | Retain correctness fix |
| 0002 fragment repack | Repacking works, but uses `ds_swizzle` instead of the patched `v_permlanex16` path | Retain existing optimization; current performance benefit is unqualified |
| 0003 SMEM reuse | Destination reuse emits `lgkmcnt(2)` before consuming an unordered load | Retain correctness fix |
| 0004 fragment reads | Operand loads are still sunk after scale loads | Retain existing scheduling policy; current performance benefit is unqualified |
| 0005 encoding config | Two canonicalization cases fail and i4/i8 configured kernels are rejected | Retain compilation fix |
| 0006 inline types | All three inlining and both template regressions pass | Remove |
| 0007 VMEM sources | SGPR address reuse still forces unnecessary completion waits | Retain existing optimization; refresh loop-plan expectations |
| 0008 dma-buf export | hrx-rs's interop entry points are absent | Retain; migrate to upstream `iree_hal_buffer_export` |
| 0009 CU/WGP mode | hrx-rs's profile execution extension is absent | Retain; use descriptor type 42 and upstream residency reporting |

Code-generation differences establish that an optimization is not redundant;
they do not establish a performance benefit on the updated compiler. No new
end-to-end performance claim is made for 0002, 0004, or 0007.

Patch SHA-256 values:

- `0001-vopd-source-cache-banks.patch`: `7b8784f70a92200c1e2fdc60562cafe9bcdf82a537568ee9f2dfba5de6ce609d`
- `0002-amdgpu-fragment-repack.patch`: `e7f43eb989ed6b43ec2fda21976338446f5a877df81ed6bbd1f79c4d5d946911`
- `0003-smem-storage-reuse-drain.patch`: `b6fdff10054d8b2bfe28fc69c003dedee2834bf33f49a4ed2ff3ffe820dba845`
- `0004-preserve-fragment-read-order.patch`: `2d5342ca88ccf8bfe8109497835c93471b5bb130f0dad6d664d936ebd9dc95df`
- `0005-materialize-encoding-config.patch`: `91d2be463c6046e2571eb6bba86902da6d1e9d5b3cf2b91e70682c5131f310e0`
- `0007-gfx11-vmem-source-reuse.patch`: `5da1cec6f6dffa8ca1d775198875d74a063aaffeff944fc56933b601eb1ddbc4`
- `0008-export-owned-gpu-dmabuf.patch`: `15d51680d96762c1d2f426c1b0f9eb2bd12ff15aa38e7b444932c5eaed986909`
- `0009-amdgpu-profile-processor-mode.patch`: `318db9d69eb9ccb0fd22d7f69fb1571e9ed628a2d2649769d71e585cc49255e2`
