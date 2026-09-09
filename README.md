# hrx-rs

Rust GPU execution and in-process Loom compilation, built on
[HRX](https://github.com/ROCm/hrx-system). Includes owned buffers, ordered streams,
events and graph replay.

Requires Rust 1.88 or later. GPU execution supports Linux x86_64 with an AMD kernel
driver, the system C/C++ runtimes and `libatomic`, and access to `/dev/kfd` and
the render device; gfx1151 is the tested architecture. Native libraries load at
runtime, so building needs no native toolchain and downloads no native code.

The package is `hrx-rs`; Rust imports and the primary CLI use `hrx`.
Native licenses, source provenance, and rebuild instructions are documented in
[THIRD-PARTY.md](THIRD-PARTY.md).

## Use from Rust

```toml
[dependencies]
hrx = { package = "hrx-rs", version = "0.2" }
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
each operation states what it comes after, `&[]` records work that may start
immediately, and `join` collects branches. The runtime topologically sorts the
result and runs independent nodes on up to eight workstreams, so declaring only
real edges is what lets it overlap. Loading kernels,
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

```sh
cargo install hrx-rs --version 0.1.0 --features runner
hrx prepare
hrx info
```

`hrx prepare` downloads and verifies the native archive pinned in
[bundle.json](bundle.json). GPU and compiler APIs also provision it on first use.
For a matching local archive, run `hrx prepare /path/to/bundle.tar.gz`.
Once prepared, `HRX_OFFLINE=1` disables network provisioning.

| Setting | Purpose |
| --- | --- |
| `HRX_RUNTIME_DIR` | Use a trusted local directory of native libraries, bypassing bundle verification |
| `HRX_BUNDLE_MANIFEST` | Use a local JSON manifest for a mirror or custom bundle |
| `HRX_OFFLINE` | Disable network provisioning when set |
| `HRX_LOOM_LIBRARY` | Override the path to `libloomc.so` |

Caches live at `$XDG_CACHE_HOME/hrx`, or `$HOME/.cache/hrx` when that is unset,
per the XDG Base Directory specification; a relative `XDG_CACHE_HOME` is ignored
as the specification requires. `hrx gc` reclaims them. The runtime lock lives in
`$XDG_RUNTIME_DIR`.

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
for native distribution status. The Rust code is [MIT licensed](LICENSE).
