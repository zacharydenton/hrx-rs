# Native components and source provenance

The unified native distribution contains three libraries built from
HRX/IREE revision `7e9c7bbd5e93d20c1e1a64c60bb4644499404f9b`:

| Library | Purpose | Principal license |
| --- | --- | --- |
| libamdf.so | Native GPU/NPU devices, memory and queues | Apache-2.0 WITH LLVM-exception |
| libloomc.so | Loom compiler, C23/C++26 import, GPU/XDNA emission | Apache-2.0 WITH LLVM-exception; MIT C++ parser |
| libhrx_fabric.so | Executable loading and native command construction | MIT bridge; Apache-2.0 WITH LLVM-exception helpers |

The compiler also includes the MIT CORE-MATH adaptation and AMD ISA tables.
HSA headers retain their NCSA license; Khronos headers retain MIT notices.
The Linux/amdxdna syscall headers retain their original notices and Linux
syscall exception. Host C/C++ libraries and kernel drivers are not distributed.

[THIRD-PARTY.json](THIRD-PARTY.json) inventories library hashes, dependencies,
license evidence, and the matching source archive. [NOTICE](NOTICE) preserves
attribution. [native/RELEASE.md](native/RELEASE.md) explains rebuilding and the
Ubuntu 26.04 baseline. [native/release-inputs.json](native/release-inputs.json)
pins upstream inputs and local patches.

Benchmark sources in `native/qualification` come from krea2-hrx (MIT) and
h3-hrx (Apache-2.0). Their exact source revisions, hashes, and license texts are
included in that directory. XDNA fixtures derived from HRX retain their
Apache-2.0 WITH LLVM-exception headers. Original Rust and native bridge code is MIT.

`src/artifacts/onnx.proto` is the ONNX project's IR schema (Apache-2.0; its
SPDX header is retained, and the license text is `src/artifacts/LICENSE-ONNX.txt`).
`src/artifacts/onnx_proto.rs` is generated from it by rust-protobuf 3.7.2 through
`scripts/generate-onnx-bindings.sh`.

The compiler regression fixture `tests/kernels/hrxdb_select_family.loom` comes
from hrxdb revision `e06d1aa67ae49e52b8ac472e52fba358b54c5ce3` (MIT). Its
license is retained in `tests/kernels/LICENSE-hrxdb.txt`.

Resident exchange, completed-clock and file-exchange fixtures in `tests/kernels`,
and the native route composer in `tests/support/resident_routes.rs`, derive from
libamdf CTS at the pinned HRX revision (Apache-2.0 WITH LLVM-exception). The
io_uring layouts in `src/fabric/storage_ffi.rs` derive from the pinned Linux UAPI;
full headers and their syscall exception are in `native/licenses`. Regenerate
with `scripts/generate-storage-bindings.py` using verified release inputs.
