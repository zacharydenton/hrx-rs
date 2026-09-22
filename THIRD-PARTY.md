# Native components and source provenance

The unified 0.8 native distribution contains three libraries built from
HRX/IREE revision `556c648e8f301ad9656d325687cc93b417ea78ff`:

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
