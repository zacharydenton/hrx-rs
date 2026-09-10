# Native distribution

The original Rust crate code is MIT licensed. The maintained NPU raw bindings
and shim derive from Apache-2.0 code; the NPU fixture uses Apache-2.0 WITH
LLVM-exception. See [the NPU notice](native/npu/NOTICE) and its adjacent license text. The native libraries retain their upstream
licenses, listed in [THIRD-PARTY.json](THIRD-PARTY.json). [NOTICE](NOTICE) and
[native/licenses](native/licenses) preserve the license texts and attributions;
both are also included in the native archive.

The replacement bundle uses a fresh HRX/Loom build and a coherent AMD runtime
set from the public HRX v0.3.0 dependency release. Its manifest identifies
TheRock build **26672984641** and exact source revisions. Every downloaded input
is pinned by SHA-256 in [native/release-inputs.json](native/release-inputs.json).
See [native/RELEASE.md](native/RELEASE.md) for evidence, build commands, changes,
and instructions for replacing the LGPL libraries.

| Components | Source / version | Selected licenses |
| --- | --- | --- |
| HRX, IREE, Loom | `ecaaf7376f7d` + seven compiler patches | Apache-2.0 WITH LLVM-exception; MIT for the CORE-MATH adaptation and generated compiler data; NCSA for HSA headers |
| HSA runtime and embedded ROCT thunk | rocm-systems `cb6561243e0a8`, HSA 1.21.0 | NCSA, MIT, BSD-2-Clause for the embedded rbtree |
| rocprofiler-register | 0.6.0 from the same AMD build | MIT; embedded fmt 11.1.4 (MIT) and glog 0.7.1 (BSD-3-Clause) |
| libelf | elfutils 0.192 | LGPL-3.0-or-later |
| libnuma | numactl 2.0.19 | LGPL-2.1-only |
| libdrm / libdrm_amdgpu | 2.4.127 | MIT |
| zlib / Zstandard / liblzma / bzip2 | 1.3.2 / 1.5.7 / 5.8.1 / 1.0.8 | Zlib / BSD-3-Clause / 0BSD / bzip2-1.0.6 |

The Arch fmt, glog, gflags, and rocprofiler-register shared libraries have been
removed. The archive contains 13 library files, including the duplicate
`libhrx.so` / `libhrx.so.0` names. `loom-compile` is not shipped.

The separate `hrx-native-sources.tar.gz` release asset contains the upstream
sources, AMD build and patch scripts, HRX patches, source manifests, and notices.
The inventory and NOTICE record its public URL and digest. Libelf and libnuma
remain dynamically linked and replaceable through `HRX_RUNTIME_DIR`; debugging
modifications to these LGPL libraries is permitted.

## Publication checklist

- [x] Replace the Arch libraries with the coherent AMD build.
- [x] Record dependency versions and exact source/archive identities.
- [x] Identify component licenses from upstream sources, including embedded code and generated-data inputs.
- [x] Stage complete license texts, attributions, inventory, and corresponding source.
- [x] Repack without `loom-compile` and pass installation and native tests.
- [x] Make the repository publicly accessible.
- [x] Upload the reviewed binary and source archives and verify anonymous installation.
- [x] Remove `publish = false` after release verification.

The [reviewed release](https://github.com/zacharydenton/hrx-rs/releases/tag/native-20260909-reviewed)
is public. On 2026-09-09, the default first-use download installed into an empty
cache without authentication and initialized gfx1151 offline. Both uploaded
archive digests match the local files; the source URL returns HTTP 200.

## Unpublished GPU/NPU extension

The working-tree extension adds interop ABI 1 and a separately built XRT shim.
It is not contained in the reviewed GPU bundle described above. Its source
patch is pinned in `native/release-inputs.json`; the NPU staging script creates
a separate hashed component manifest. System XRT remains dynamically linked
and is not included in that archive. Chess is never redistributed.
