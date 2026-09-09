# Loom compiler patches

These patches apply to public `ROCm/hrx-system` commit
`ecaaf7376f7dcaa599f6258b0d1c38ff7fbd0e3d`; `base-revision` is the machine-readable
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

`0002-amdgpu-fragment-repack.patch` adds the accumulator-to-RHS fp16 fragment
repack used by query-32 attention kernels, with native regression coverage.

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

`0006-bind-dependent-inline-types.patch` substitutes callable arguments into
dependent vector types before validating and inlining selected templates.

`0007-gfx11-vmem-source-reuse.patch` removes memory-completion leases for GFX11
VMEM address/resource SGPR sources. Hardware interlocks their consumption;
reusing an address does not require waiting for the loaded value. Result-write
leases and the SMEM fix remain active. A minimal gfx1151 assembly test reuses
the address immediately and waits before consuming the destination. It fails
before the fix and passes afterward. The isolated fix is on fork branch
[`fix/loom-gfx11-vmem-source-reuse`](https://github.com/zacharydenton/hrx-system/tree/fix/loom-gfx11-vmem-source-reuse),
commit [`beaff74b2`](https://github.com/zacharydenton/hrx-system/commit/beaff74b2).

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
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151
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

The bundle records this base and every patch digest in `provenance.json`.

## Validation

See [VALIDATION.md](../../VALIDATION.md) for the tested bundle and crate results.
The patch files include the native regression fixtures described above.

Patch SHA-256 values:

- `0001-vopd-source-cache-banks.patch`: `cee76d2ab6cd3d279070a93e578b28931a7e6debd949080cd15a5bd0dcb3a06b`
- `0002-amdgpu-fragment-repack.patch`: `cc415b38829104821c7964c1c22aac764e2cb8aabc288f41a7536db32ccd884f`
- `0003-smem-storage-reuse-drain.patch`: `62506281067aa86810e4b373e3b7e1b150999c150fe753f524789695f50d96f0`
- `0004-preserve-fragment-read-order.patch`: `7608c0d3d86ce64b72a08f5310fb6657a1473e1e20428b14958a5091db6e06b5`
- `0005-materialize-encoding-config.patch`: `a74b0ae2cd73be6bbd219b8f27d324944df088eeb582e411baa47ebabbd6ead2`
- `0006-bind-dependent-inline-types.patch`: `c196c0e284c2d8ba6946c00b9816a43208ab6ef55dc1601fab30efc3b4f8d56c`
- `0007-gfx11-vmem-source-reuse.patch`: `032a77cd4786b71e916f9e1d7ce69faed1d8e8b22c7f111bfa1ca47b2d814556`
