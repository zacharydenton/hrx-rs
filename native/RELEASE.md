# Native release inputs and rebuilds

`release-inputs.json` pins the download URL and SHA-256 of every input. It also
preserves AMD's TheRock manifest, including exact submodule revisions. The
published HRX v0.3.0 dependency manifest identifies TheRock build **26672984641**
(ROCm **7.14.0a20260530**, TheRock commit
`9dfbd936fa3750b21f4aabea2f861f97a435aec0`, rocm-systems commit
`cb6561243e0a80215f5566a0feeb19eb44702aa4`). Its published archive digest matches
our downloaded bytes. The accompanying `sysdeps_dev` artifact records libelf
0.192, libnuma 2.0.19, and libdrm 2.4.127 in their pkg-config files.

## Review scope

The runtime contains 13 shared-library files. Ten are copied, without changing
their bytes, from AMD's dependency archive. Three are a fresh build of HRX/Loom
from `ecaaf7376f7dcaa599f6258b0d1c38ff7fbd0e3d` with the eight patches under
`patches/loom`. `libhrx.so` and `libhrx.so.0` contain identical bytes.

The old Arch fmt, glog, gflags, and rocprofiler-register binaries are removed.
AMD's rocprofiler-register embeds fmt **11.1.4** and glog **0.7.1**; their exact
submodule pins and license texts remain in the inventory. HSA embeds the ROCT
thunk, including its Nginx-derived BSD-2-Clause rbtree. HRX/Loom include IREE and
its MIT-licensed CORE-MATH adaptation. Libbacktrace and runtime tracing are
disabled in the HRX build. The link commands contain no separately linked
third-party archives. Generated compiler tables use the MIT-licensed AMD GPU
ISA XML (2026-03-05) and SPIR-V grammar; the build also uses HSA and Vulkan
headers. Their pinned archives and license texts are included. The XML declares
MIT and its copyright in its Document element; its notice supplies the standard
MIT text with that attribution. The native compiler executable is not shipped.

License identifiers are taken from the pinned source distributions. Libelf's
LGPL-3.0-or-later option is selected; libnuma uses LGPL-2.1-only; Zstandard's
BSD-3-Clause option is selected. XZ 5.8.1 identifies liblzma as 0BSD. Libdrm keeps
its MIT notices in individual source headers; those notices are collected
verbatim for the core/AMDGPU sources and headers. The complete corresponding
sources, license texts, AMD build recipes, and symbol/SONAME patch scripts are
provided as a separate release asset beside the binary archive.

This records source and artifact provenance, not bit-for-bit reproduction of
AMD's CI environment. The fresh HRX/Loom libraries target Ubuntu 24.04 (glibc 2.39). System glibc,
libstdc++, libgcc, libatomic and the kernel driver are
provided by the host and are not redistributed in the bundle.

## Rebuild HRX and stage a release

Run from the hrx-rs repository. Use Python 3.12+, tar with zstd support, patch,
and Podman. The build runs Clang 18 in the pinned Ubuntu 24.04 image; it requires
no host ROCm SDK. Compiler and OS package versions are recorded in the generated
provenance and source archive.

```sh
python3 scripts/fetch-native-inputs.py --work artifacts/new-release
bash scripts/rebuild-hrx.sh artifacts/new-release
python3 scripts/stage-native-release.py --work artifacts/new-release \
  --release-tag native-20260910-gpu-npu
cargo run --release --features runner --bin hrx -- pack \
  artifacts/new-release/stage artifacts/new-release/packed \
  https://github.com/zacharydenton/hrx-rs/releases/download/native-20260910-gpu-npu/hrx-linux-x86_64-gfx1151.tar.gz \
  'HRX ecaaf7376f7d + eight patches; TheRock 26672984641' gfx1151
```

The scripts preserve downloaded source archives. The container builder uses
checked-in GPU device binaries; Loom kernel compilation is included. Use a fresh work directory for
a rebuild; staging refuses a nonempty destination. Both the binary archive and
`hrx-native-sources.tar.gz` must be uploaded to the release named in the command.
The inventory and NOTICE record the source archive URL and SHA-256.

## Corresponding source and library replacement

The source release contains pristine upstream archives under `upstream/`, all
eight HRX patches, this document, license texts, the input manifest, and release
scripts. The scripts, manifest, and patches preserve the repository layout. Copy
`upstream/` archives into the fetch script's work cache to reuse them; it verifies
existing files before use. The AMD binary artifacts must be fetched separately
when staging, using the URLs and hashes in `native/release-inputs.json`.

AMD's modifications to libelf and libnuma are in the bundled TheRock archive:

- `third-party/sysdeps/linux/elfutils/{CMakeLists.txt,patch_source.sh,patch_install.sh}`
- `third-party/sysdeps/linux/numactl/{CMakeLists.txt,patch_source.sh,patch_install.sh}`

Those CMake files contain the configure/build/install commands. The source
scripts prefix ELF symbol versions and rename the libraries; the install scripts
normalize library names. The complete TheRock sources supply the surrounding
build infrastructure and the compression-library recipes. To rebuild either
library independently, unpack its source archive, apply its `patch_source.sh`,
and use its normal Autotools build with the configure flags from the CMake file.
Libnuma needs `autoreconf -fi` after patching. Libelf needs the compression-library
headers and libraries (zlib, zstd, xz, bzip2); their sources and AMD recipes are
included. Preserve the `AMDROCM_SYSDEPS_1.0_` symbol versions, SONAMEs, and dynamic
dependency names when replacing these libraries in this runtime.

For the full original AMD build workflow, unpack the TheRock archive and follow
its README and `.github/workflows` at the pinned revision. Its source manifest
records the exact rocm-systems revision and every other submodule pin. The
upstream HRX dependency manifest is included. The input manifest pins the original
TheRock `base_lib`, `base_run`, `sysdeps_dev`, and `core-runtime_run` artifacts as
build evidence; these binary archives are not part of the source distribution.

To run with modified libraries, copy the staged runtime directory, replace the
shared libraries, and set `HRX_RUNTIME_DIR` to that directory. This intentionally
bypasses the verified-cache hashes. Modification and reverse engineering for
debugging modifications to the LGPL libraries are permitted. Full license terms
are supplied in `LICENSE-elfutils-*` and `LICENSE-numactl-*`.
