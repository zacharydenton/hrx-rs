# Runtime setup

```sh
cargo install hrx-rs --version 0.10.1 --locked --features npu
hrx prepare
hrx doctor
```

GPU execution requires the `amdgpu`/KFD driver, access to `/dev/kfd` and the
render device, compatible C/C++ runtimes, and `libatomic`. NPU execution also
requires the `amdxdna` driver and firmware. A ROCm SDK is not required.

`hrx prepare` downloads the archive pinned in [bundle.json](../bundle.json).
APIs also provision it on first use. For offline installation, run
`HRX_OFFLINE=1 hrx prepare native.tar.gz` with the matching archive.

| Setting | Purpose |
| --- | --- |
| `HRX_RUNTIME_DIR` | Use a trusted local native-library directory; bypasses bundle verification |
| `HRX_BUNDLE_MANIFEST` | Use a local bundle manifest for a mirror or custom build |
| `HRX_OFFLINE` | Disable network provisioning when set |
| `HRX_AMDF_LIBRARY` | Override `libamdf.so` |
| `HRX_FABRIC_LIBRARY` | Override `libhrx_fabric.so` |
| `HRX_LOOM_LIBRARY` | Override `libloomc.so` |

Caches use `$XDG_CACHE_HOME/hrx`, or `$HOME/.cache/hrx` if the variable is unset
or relative. The runtime lock uses `$XDG_RUNTIME_DIR`. Compiled kernels share a
content-addressed cache across applications; each cache hit verifies the artifact
hash. `hrx gc [DAYS]` removes unpinned bundles and kernels unused for DAYS
(default 30).

