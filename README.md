# hrx-rs

**GPU and NPU compute from Rust—no PyTorch or ROCm SDK installation.**

Write GPU kernels in [Loom](https://github.com/ROCm/hrx-system), compile them
inside your application, and run them through [HRX](https://github.com/ROCm/hrx-system).
Add the crate with Cargo; the runtime and compiler download automatically on
first use, with pinned versions and verified hashes. GPU execution and Loom
compilation need no Python environment, HIP headers, or `hipcc`.

The **GPU runtime and Loom compiler together are a 7.6 MB download**. The optional
**NPU runtime adds 4.3 MB**, including XRT. These are the compressed archives in
the [current native release](https://github.com/zacharydenton/hrx-rs/releases/tag/native-20260910-gpu-npu),
separate from Cargo dependencies and any model weights your application uses.

- **Build with Cargo:** no GPU SDK, C++ compiler, or native-library download at build time.
- **Compile and reuse:** Loom kernels compile in process and share a verified disk cache across applications.
- **Control execution:** owned buffers, ordered streams, events, and reusable graphs keep data and work on the device.
- **Use the NPU too:** run precompiled XDNA2 programs without Python or a Ryzen AI SDK installation, and coordinate GPU/NPU work through one API.

Tested on **AMD Strix Halo (`gfx1151`)**, on Linux x86_64. The host supplies the
kernel drivers and compatible system libraries; see [requirements and setup](#cli-and-native-setup).

The package is `hrx-rs`; Rust imports and the primary CLI use `hrx`.
Native licenses, source provenance, and rebuild instructions are documented in
[THIRD-PARTY.md](THIRD-PARTY.md).

## Project showcase

These projects use HRX and Loom on AMD Strix Halo for local generation,
computer vision, and vector search.

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
| [![A glacier floats above a fjord, with waterfalls falling beside a small boat](https://raw.githubusercontent.com/zacharydenton/h3-hrx/master/docs/media/benchmarks/20260913-768p/h3.jpg)](https://github.com/zacharydenton/h3-hrx/blob/master/docs/media/benchmarks/20260913-768p/h3.mp4) | [![A figure on a basalt sea cliff beneath a ringed planet](https://raw.githubusercontent.com/zacharydenton/krea2-hrx/master/docs/images/planetrise.png)](https://github.com/zacharydenton/krea2-hrx#gallery) |
| [Watch the 768p glacier video](https://github.com/zacharydenton/h3-hrx/blob/master/docs/media/benchmarks/20260913-768p/h3.mp4) | [Explore the image gallery and prompts](https://github.com/zacharydenton/krea2-hrx#gallery) |

h3 generates the five-second 768p clip in **37 min 33 s**, versus **roughly
4 h 20 min with ComfyUI** at 20 evaluations on Strix Halo—about **7× faster
overall**. See the [timing calculation and memory comparison](https://github.com/zacharydenton/h3-hrx/blob/master/docs/benchmarks/20260913-768p/README.md#end-to-end-timing).

The vision libraries also fit together: SCRFD supplies face landmarks to
ArcFace for alignment and embeddings, while DINOv3 produces image vectors
that an application can index with hrxdb.

## GPU + NPU pipelines

The coordinated API is `hrx::execution`: owned shared buffers, checked kernel
contracts, inferred dependencies, reusable GPU/NPU graphs, and completion handles
that support blocking waits and Rust `Future`. `hrx::gpu` exposes the existing
low-level GPU API. Enable `npu` for XDNA2 execution and `npu-compile` for integrated
IRON/AIE compilation; neither feature needs native tools during Cargo builds.

See [the GPU/NPU guide](docs/GPU-NPU.md) for the trust boundary, host mapping guards,
shared native runtime setup, compiler pinning, and runnable hardware validation.
The published GPU and NPU bundles include the matching shared-memory runtime.

The [SCRFD + DINOv3 throughput benchmark](scripts/vision-bench/README.md) compares
GPU-only and mixed GPU/NPU image processing with standalone model backends.
Its [measured results](scripts/vision-bench/RESULTS.md) cover application throughput
using those runtimes; they do not measure the Rust scheduler.

## Use from Rust

Requires Rust 1.88 or later.

```toml
[dependencies]
hrx = { package = "hrx-rs", version = "0.4", features = ["npu"] }
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
A buffer is bound to its device, not to the stream that allocated it: any stream
on that device may use it, and `record_event`/`wait_event` order conflicting
access. Unordered cross-stream use yields whichever bytes the device held.
Dispatches, fills and copies take `&self`; only staging-backed transfers and
synchronization need `&mut`. `Kernel` is `Clone`, retaining the executable.
Transfers, fills, copies and dispatch bindings use `View`. Borrow a whole buffer
with `buffer.binding()` and a subregion with `view.slice(offset, length)?`.
`upload` queues a transfer through owned staging; `upload_blocking` waits for it.
Events coordinate streams. `Stream::graph` records work as a dependency graph:
each operation names its predecessors, and `&[]` starts an independent branch.
Pass branch endings directly to their consumer. `join` collects dependencies
in an empty node, which adds a native partition and queue barrier. The runtime
can schedule independent nodes on up to eight workstreams; the graph's shape
and GPU resource use determine whether execution overlaps. Loading kernels,
dispatching them, and sharing buffers across streams are unsafe: callers must
validate code, arguments, memory access, and synchronization.

`hrx::loom::Compiler` compiles Loom in process and caches artifacts. Select the
compiler target from `Device::target()` when compiling for a GPU.
`Compiler::compile_all` runs a batch across `CompilerOptions::workers`
workspaces and returns results in request order; `Module::compile` is blocking,
so a single-threaded caller never reaches that bound on its own. Compiler setup
and source pins are in [patches/loom](patches/loom/README.md).

Two different things are called the target, and they are chosen independently.
The **profile** target is `hrx::Target` — the architecture a device reports and
the one the compiler emits for. It is a bare architecture key such as `gfx1151`;
`Target::new` rejects generic names. The **source-level** target is what Loom
source writes as `amdgpu.target<...>`, which does accept generic names such as
`gfx11-generic` and compiles fine under a bare profile. A generic source target
has no low-asm contract, so hand-written asm needs a bare architecture there
too.

## CLI and native setup

GPU execution requires Linux x86_64, the AMD `amdgpu`/KFD kernel driver,
the system C/C++ runtimes and `libatomic`, and access to `/dev/kfd` and the render
device. You do not need a system ROCm SDK or PyTorch installation: HRX supplies
its own pinned user-space runtime, including HSA, and the Loom compiler.

```sh
cargo install hrx-rs --version 0.4.0 --locked --features runner,npu
hrx prepare
hrx doctor
```

`hrx prepare` downloads and verifies the GPU and NPU archives pinned in
[bundle.json](bundle.json) and [npu-bundle.json](npu-bundle.json).
GPU, compiler and NPU APIs also provision their runtime on first use.
For a matching local archive, run `hrx prepare /path/to/bundle.tar.gz`.
Once prepared, `HRX_OFFLINE=1` disables network provisioning.

This provisions both user-space runtimes, including XRT; no separate
XRT, Python or Ryzen AI SDK installation is needed for precompiled NPU programs.
Compiling new NPU programs uses an [optional external IRON/AIE toolchain](docs/GPU-NPU.md#npu-compilation).
Linux GPU/NPU drivers, firmware and device permissions remain host prerequisites.
The bundled runtimes target Ubuntu 26.04 LTS: hosts need glibc 2.43 or newer
and compatible C/C++ runtime libraries. Stock Ubuntu 24.04 is not supported by
these bundles. The current LTS is the selected distribution baseline.
See [the installation guide](docs/GPU-NPU.md#native-setup).

With NPU support enabled, supplying only a local GPU archive still lets `prepare`
download the NPU runtime. For an offline installation, use
`HRX_OFFLINE=1 hrx prepare /path/to/gpu.tar.gz /path/to/npu.tar.gz`.
`prepare` prints the verified GPU directory to stdout before preparing the NPU;
an NPU provisioning failure still produces a nonzero exit status.

| Setting | Purpose |
| --- | --- |
| `HRX_RUNTIME_DIR` | Use a trusted local directory of native libraries, bypassing bundle verification |
| `HRX_BUNDLE_MANIFEST` | Use a local JSON manifest for a mirror or custom bundle |
| `HRX_OFFLINE` | Disable network provisioning when set |
| `HRX_LOOM_LIBRARY` | Override the path to `libloomc.so` |

Caches live at `$XDG_CACHE_HOME/hrx`, or `$HOME/.cache/hrx` when that is unset,
per the XDG Base Directory specification; a relative `XDG_CACHE_HOME` is ignored
as the specification requires. The runtime lock lives in `$XDG_RUNTIME_DIR`.

Compiled kernels go to one cache under that root, shared by every consumer on the
machine. Artifacts are content-addressed — the key covers compiler identity,
source, export, target and canonical configuration, and every hit re-verifies the
bytes against their recorded digest — so there is nothing a per-model location
could distinguish, and two models that compile the same kernel compile it once.
`hrx gc [DAYS]` evicts runtime bundles `bundle.json` does not pin and kernels not
read for DAYS (default 30), by access time, so entries written by any release are
dated the same way.

Default features are `download` and `loom`. Enable `runner` for the `hrx` CLI,
whose `run` subcommand launches a compiled kernel and dumps its buffers.
`Stream` is the execution API; dispatch takes explicit `Constants`.

## Development

```sh
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo doc --all-features --no-deps --open
```

See [VALIDATION.md](VALIDATION.md) for GPU tests, stream benchmarks and the feature matrix,
[CHANGELOG.md](CHANGELOG.md) for API changes, and [THIRD-PARTY.md](THIRD-PARTY.md)
for native distribution status. Original Rust code is [MIT licensed](LICENSE); NPU-derived code retains its
[upstream licenses](native/npu/NOTICE).
