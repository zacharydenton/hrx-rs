# hrx-rs

Rust APIs for AMD GPU and NPU execution, with an in-process
[Loom](https://github.com/ROCm/hrx-system) compiler.

Cargo builds need no native SDK. On first use, HRX downloads a pinned,
hash-verified bundle containing the runtime and compiler. The host supplies
Linux drivers and system libraries.

Supported hardware is AMD Strix Halo (`gfx1151`) and its NPU
(`amd.xdna.strix_halo.17f0_11`) on Linux x86_64. The native bundle is built on
Ubuntu 26.04 and requires glibc 2.43 or newer.

## Quick start

Requires Rust 1.91 or later. The package is `hrx-rs`; the Rust crate and CLI are
named `hrx`.

```toml
[dependencies]
hrx = { package = "hrx-rs", version = "0.10.1" }
```

```rust,no_run
fn main() -> hrx::Result<()> {
    let mut stream = hrx::Stream::open()?;
    let buffer = stream.allocate(4096)?;
    stream.upload(buffer.binding(), &[7; 4096])?;
    let readback = stream.read(buffer.binding())?;
    assert_eq!(readback.wait(&mut stream)?, vec![7; 4096]);
    Ok(())
}
```

`Device` selects a GPU, `Stream` orders work, and `Buffer` owns an allocation.
Use `loom::Compiler` to compile kernels and `Stream::graph` to prepare reusable
GPU work. `execution::Runtime` coordinates GPU/NPU graphs, shared buffers and
completion futures. Add the `npu` Cargo feature for NPU execution.

Loading native code, dispatching kernels and declaring memory-access contracts
are unsafe operations. Callers must validate code, arguments and synchronization.

## Install the CLI

```sh
cargo install hrx-rs --version 0.10.1 --locked --features npu
hrx prepare
hrx doctor
```

GPU execution requires the `amdgpu`/KFD driver, access to `/dev/kfd` and the
render device, compatible C/C++ runtimes, and `libatomic`. NPU execution also
requires the `amdxdna` driver and firmware. A ROCm SDK is not required.

`hrx prepare` downloads the archive pinned in [bundle.json](bundle.json).
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

## Guides

- [GPU execution and inference](docs/EXECUTION.md): streams, graphs, memory,
  model composition and allocation budgets.
- [Loom compilation](docs/COMPILER.md): targets, launch geometry, reports,
  tracing and sanitizers.
- [GPU/NPU execution](docs/GPU-NPU.md): shared memory, scheduling and examples.
- [Native storage](docs/STORAGE.md): buffered/direct file I/O and read leases.
- [Queue pool benchmarks](benchmarks/queue-pool/README.md): queue sizing and
  reproduction commands.
- [Native builds](native/RELEASE.md) and [compiler patches](patches/loom/README.md).
- [API reference](https://docs.rs/hrx-rs) and [changelog](CHANGELOG.md).

To run independent GPU and NPU work in one graph:

```sh
cargo run --release --features npu --example gpu_npu_parallel -- 16777216 1024 21 trace.json
```

The example checks a GPU vector transform and NPU matrix multiplication against
CPU results, compares sequential and concurrent execution, and writes a
Chrome/Perfetto trace. See the [GPU/NPU guide](docs/GPU-NPU.md#examples) for details.

## Projects using HRX

| Project | What it does |
| --- | --- |
| [hrxdb](https://github.com/zacharydenton/hrxdb) | Embedded GPU vector database with exact cosine search, batched top-k, and custom scoring over resident collections. |
| [h3-hrx](https://github.com/zacharydenton/h3-hrx) | MiniMax H3 video generation with sound, from text, a first frame, or image and audio references. |
| [krea2-hrx](https://github.com/zacharydenton/krea2-hrx) | Krea 2 Turbo and Raw text-to-image generation with a complete text encoder, diffusion transformer, and VAE pipeline. |
| [dinov3-hrx](https://github.com/zacharydenton/dinov3-hrx) | DINOv3 image embeddings and patch features, with resident weights and reusable execution graphs. |
| [arcface-hrx](https://github.com/zacharydenton/arcface-hrx) | ArcFace face embeddings with five-point alignment and cosine similarity. |
| [scrfd-hrx](https://github.com/zacharydenton/scrfd-hrx) | SCRFD face detection with bounding boxes, confidence scores, and five facial landmarks. |

| h3-hrx · video with sound | krea2-hrx · image generation |
| --- | --- |
| [![An enormous alien creature glides above a fjord and a small boat](https://raw.githubusercontent.com/zacharydenton/h3-hrx/master/docs/media/benchmarks/20260914/h3-i8.jpg)](https://github.com/zacharydenton/h3-hrx/blob/master/docs/media/benchmarks/20260914/h3-i8.mp4) | [![A figure on a basalt sea cliff beneath a ringed planet](https://raw.githubusercontent.com/zacharydenton/krea2-hrx/master/docs/images/planetrise.png)](https://github.com/zacharydenton/krea2-hrx#gallery) |
| [Watch the 768p alien video](https://github.com/zacharydenton/h3-hrx/blob/master/docs/media/benchmarks/20260914/h3-i8.mp4) | [Explore the image gallery and prompts](https://github.com/zacharydenton/krea2-hrx#gallery) |

See H3's [generation benchmark](https://github.com/zacharydenton/h3-hrx/blob/master/docs/benchmarks/20260914/README.md)
for timings, memory use and a ComfyUI comparison.

## Development

```sh
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo doc --all-features --no-deps --open
```

Run `scripts/check-feature-matrix.sh` for feature checks. Hardware test commands
are in the device and storage guides.

Original Rust code is [MIT licensed](LICENSE). Native component licenses and
source provenance are listed in [THIRD-PARTY.md](THIRD-PARTY.md).
