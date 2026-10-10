# hrx-rs

Rust APIs for AMD GPU and NPU execution, with an in-process
[Loom](https://github.com/ROCm/hrx-system) compiler.
Targets Strix Halo (`gfx1151`) and its XDNA NPU on Linux x86-64.

- GPU buffers, streams, events and reusable execution graphs.
- Loom compilation, shared artifact caches, resource reports and sanitizers.
- GPU/NPU scheduling with shared memory and completion futures.
- Model composition, allocation budgets and residency management.
- Buffered and direct file I/O with GPU-authored storage queues.

## Quick start

Requires Rust 1.91+. The Cargo package is `hrx-rs`; import it as `hrx`:

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

Use `loom::Compiler` for kernels and `Stream::graph` for reusable GPU work.
`execution::Runtime` coordinates GPU/NPU graphs. Enable the `npu` Cargo feature
for NPU execution. Native code loading, kernel dispatch and memory-access
contracts require caller validation at the unsafe API boundaries.

## Install the CLI

```sh
cargo install hrx-rs --version 0.10.1 --locked --features npu
hrx prepare
hrx doctor
```

HRX downloads a pinned runtime/compiler bundle; no ROCm SDK is needed to build.
The host needs amdgpu/KFD access, glibc 2.43+ and compatible C/C++ libraries.
NPU execution also needs the amdxdna driver and firmware.
[Setup, offline bundles and overrides](docs/SETUP.md).

Run independent GPU and NPU operations in one graph:

```sh
cargo run --release --features npu --example gpu_npu_parallel -- 16777216 1024 21 trace.json
```

The example checks results against CPU references, compares sequential and
concurrent execution, and writes a Chrome/Perfetto trace.

## Projects using HRX

| Project | Model or workload |
| --- | --- |
| [h3-hrx](https://github.com/zacharydenton/h3-hrx) | MiniMax H3 video and audio generation |
| [krea2-hrx](https://github.com/zacharydenton/krea2-hrx) | Krea 2 Turbo and Raw image generation |
| [clef-hrx](https://github.com/zacharydenton/clef-hrx) | Structured decisions from text and media |
| [dinov3-hrx](https://github.com/zacharydenton/dinov3-hrx) | Image embeddings and patch features |
| [scrfd-hrx](https://github.com/zacharydenton/scrfd-hrx) | Face detection and landmarks |
| [arcface-hrx](https://github.com/zacharydenton/arcface-hrx) | Face alignment and embeddings |
| [hrxdb](https://github.com/zacharydenton/hrxdb) | Vector search and custom scoring over resident corpora |

[H3 videos and timings](https://github.com/zacharydenton/h3-hrx/blob/master/docs/showcase.md) ·
[Krea image gallery](https://github.com/zacharydenton/krea2-hrx#gallery)

## Documentation

- [GPU execution and model composition](docs/EXECUTION.md)
- [Loom compilation and diagnostics](docs/COMPILER.md)
- [GPU/NPU scheduling](docs/GPU-NPU.md)
- [Native storage](docs/STORAGE.md)
- [Queue pool benchmarks](benchmarks/queue-pool/README.md)
- [Native builds](native/RELEASE.md) and [compiler patches](patches/loom/README.md)
- [API reference](https://docs.rs/hrx-rs) and [changelog](CHANGELOG.md)

## Development

```sh
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo doc --all-features --no-deps --open
```

`scripts/check-feature-matrix.sh` checks feature combinations.
Hardware test commands are in the execution and storage guides.

## License

Original Rust code is [MIT licensed](LICENSE).
[Native component licenses and source provenance](THIRD-PARTY.md).
