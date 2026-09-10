# NPU runtime release

This is the user-space runtime for precompiled XDNA2 programs, separate from the
GPU/Loom bundle and from optional NPU compilation tools. It includes five shared
libraries: the HRX ABI 1 shim, XRT coreutil, XRT core, the open-source XDNA plugin,
and libuuid. XRT discovers its plugin beside its loaded library; no process-global
`XILINX_XRT` or `LD_LIBRARY_PATH` changes are required.

`npu-release-inputs.json` pins the Ubuntu 26.04 build image and every Git source
revision, including nested modules. The matching XRT revision is the one recorded
by the XDNA repository. The only source changes disable OS/VTD packaging and
AIEBU documentation generation (the pinned source already includes its ISA
headers). Every shipped DSO has a relative RUNPATH; the shim uses `$ORIGIN/lib`,
and XRT uses `$ORIGIN`. The builder records all Ubuntu package versions.

The host supplies normal glibc/libstdc++/libgcc libraries, the `amdxdna` kernel
driver, NPU firmware and device access. Builds target Ubuntu 26.04's glibc 2.43
and GCC 15 C++ runtime. The per-library symbol requirements are recorded in
`NPU-THIRD-PARTY.json`. The GPU bundle has its own host-library requirements.
No XRT installation, Python, ONNX Runtime or Ryzen AI SDK is required to execute
precompiled programs through Rust. Compiling new NPU kernels remains an optional
external toolchain.

## Build and stage

These are maintainer commands, never Cargo build steps. They require Git,
Python 3.12+, Podman and network access. Use a fresh work and output directory.

```sh
scripts/build-npu-runtime.sh artifacts/npu-build
python3 scripts/stage-npu-component.py \
  artifacts/npu-build/runtime artifacts/npu-release \
  --work artifacts/npu-build \
  --url https://github.com/zacharydenton/hrx-rs/releases/download/native-20260910-gpu-npu/hrx-npu-linux-x86_64.tar.gz
```

The output contains `npu-bundle.json`, the binary archive and
`hrx-npu-sources.tar.gz`. Copy the manifest into the crate only after validating
the artifacts. Stage the GPU bundle using `RELEASE.md`. Both binary archives and
both source archives belong in the named release; staging does not publish.

The source archive contains pristine Git archives for every source revision,
source identities and hashes, patches, shim source, build scripts, license texts,
and the package inventory. `source_archive` in its input manifest maps each
repository path to its archive; extract each into that path under `xdna-driver`
and apply the listed patches to their specified repository roots. The container
build script then builds from `/work/xdna-driver`. Alternatively, the build
wrapper fetches those same commits directly from the recorded public URLs.
This records provenance, not bit-identical reproduction of distro packages.

## Licenses and replacement

XRT, its XDNA plugin and the HRX shim use Apache-2.0. The fixture's separate LLVM
exception notice is preserved. Embedded AIEBU, AIE-RT, ELFIO, GSL, cxxopts and
nlohmann JSON use MIT; Zstandard and xxHash use BSD licenses; Boost headers use
BSL-1.0. Ubuntu libuuid uses BSD-3-Clause. Original per-file notices, upstream
license texts, Ubuntu package copyright records and referenced license texts
are included in the binary archive. No VTD diagnostic archives, firmware or
kernel module binaries are redistributed.

To use a locally modified build, select its directory with
`HRX_NPU_RUNTIME_DIR`. Directories containing `component.json` are still verified;
use a development directory without that cache manifest for modified libraries.
