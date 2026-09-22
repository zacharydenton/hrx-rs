# Native distribution

The original Rust crate code is MIT licensed. The maintained NPU raw bindings
and shim derive from Apache-2.0 code; the NPU fixture uses Apache-2.0 WITH
LLVM-exception. See [the NPU notice](native/npu/NOTICE) and its adjacent license text. The native libraries retain their upstream
licenses, listed in [THIRD-PARTY.json](THIRD-PARTY.json). [NOTICE](NOTICE) and
[native/licenses](native/licenses) preserve the license texts and attributions;
both are also included in the native archive.

The GPU bundle combines a fresh HRX/Loom build, nine unchanged library files
from HRX v0.3.0's AMD dependency archive (TheRock **26672984641**), and an updated
ROCr library from TheRock **35670146294**. The newer ROCr supplies
`hsa_amd_queue_create`, required by this HRX revision. Every input and the ROCr
library are pinned by SHA-256 in [native/release-inputs.json](native/release-inputs.json).
See [native/RELEASE.md](native/RELEASE.md) for source provenance, build commands,
and instructions for replacing the LGPL libraries.

| Components | Source / version | Selected licenses |
| --- | --- | --- |
| HRX, IREE, Loom | `556c648e8` + eight patches (compiler and interop) | Apache-2.0 WITH LLVM-exception; MIT for the CORE-MATH adaptation and generated compiler data; NCSA for HSA headers |
| HSA runtime and embedded ROCT thunk | rocm-systems `0816fc809a4f`, HSA 1.21.0 | NCSA, MIT, BSD-2-Clause for the embedded rbtree |
| rocprofiler-register | 0.6.0 from TheRock 26672984641 | MIT; embedded fmt 11.1.4 (MIT) and glog 0.7.1 (BSD-3-Clause) |
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

The [native release](https://github.com/zacharydenton/hrx-rs/releases/tag/native-20260922-hrx-update)
contains the GPU binary bundle and matching source archive. It includes shared
buffer interop ABI 1 and CU/WGP compiler profiles. The NPU runtime remains a
separate pinned component with its own inventory, licenses, and source archive;
see [native/NPU-RELEASE.md](native/NPU-RELEASE.md). Chess is never redistributed.
