# Unified native release

The native archive contains `libamdf.so`, `libhrx_fabric.so`, and `libloomc.so`.
All three are built from the pinned HRX source on Ubuntu 26.04. GPU execution
uses KFD directly; NPU execution uses the amdxdna driver directly. Loom compiles
Loom and explicitly supplied C23/C++26 translation units in process.

`release-inputs.json` records every upstream download, the compiler patch set,
the native build, queue and cache-policy patches, and the Ubuntu image digest. The upstream tree
also pins its build dependencies. `provenance.json` records bridge source hashes,
compiler version, CMake cache hash, and library hashes. OS package versions are
included with the rebuild sources. This is source provenance, not a claim of
bit-for-bit build reproducibility.

The qualified profiles are Linux x86_64 gfx1151 and
`amd.xdna.strix_halo.17f0_11`. Host requirements are glibc 2.43 or newer,
compatible libstdc++/libgcc, amdgpu/KFD and amdxdna drivers, firmware, and device
permissions. The bundle ships no HSA/ROCr, XRT, IRON, or Python runtime.
HSA headers supply code-object definitions only. The C++ importer embeds the
MIT-licensed cplusplus parser, including the patches in the pinned upstream
source. Linux syscall headers retain their syscall exception and full notices.

## Rebuild and stage

Use Python 3.12+, patch, tar, and Docker or Podman. Starting in this repository:

```sh
python3 scripts/fetch-native-inputs.py --work artifacts/native-release
CONTAINER_ENGINE=podman HRX_BUILD_JOBS=2 HRX_BUILD_MEMORY=8g \
  bash scripts/rebuild-hrx.sh artifacts/native-release
python3 scripts/stage-native-release.py --work artifacts/native-release \
  --release-tag native-20261008-7e9c7bbd5e
cargo run --release --bin hrx -- pack artifacts/native-release/stage-amdf \
  artifacts/native-release/output \
  https://github.com/zacharydenton/hrx-rs/releases/download/native-20261008-7e9c7bbd5e/hrx-linux-x86_64-gfx1151.tar.gz \
  'HRX 7e9c7bbd5e; current Loom artifact API; Ubuntu 26.04' gfx1151
```

The container build defaults to two jobs and an 8 GiB memory limit, with no
additional swap allowance. `HRX_BUILD_JOBS` and `HRX_BUILD_MEMORY` override these
limits. The job count is passed into the container. Direct `build-amdf.sh` builds
also default to two jobs; apply an external memory limit when building on the host.

The build enables the task driver declarations required by upstream's shared
compiler ABI-layout libraries, while the distributed bridge continues to use
native libamdf GPU and XDNA execution.

The source archive contains upstream inputs, local patches, the native bridge,
license texts, and rebuild scripts. To rebuild from the archive, extract it,
move `upstream/*` into your work directory's `sources/`, then run the same
fetch/rebuild/stage commands. Fetch verifies cached inputs before reuse.
CMake may download its pinned build-only dependencies. No vendor SDK is needed.

Before publication, run CPU checks, the feature matrix, all ignored hardware
checks, and the paired compiler corpus against the exact staged libraries.
Inspect `readelf -d` and symbol versions. Pack twice and compare archive hashes;
install the local archive into an empty cache and rerun with `HRX_OFFLINE=1`.
Publish the source archive alongside the binary and manifest, then update
`bundle.json` and publish the matching crate. Do not mix different native builds.
