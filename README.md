# hrx

Shared Rust GPU execution and Loom compilation for gfx1151, with no native link
dependency, LLVM build, Python, NumPy, Torch, or BEAM dependency. Linux x86_64 is
the first supported platform. The host still needs a functioning AMD kernel
driver and access to `/dev/kfd` and the render device.

This checkout implements the shared runtime used by the sibling `minimax-h3-loom`
and `krea2-loom` workspaces. It has not been published to crates.io. The bundled
manifest names a proposed GitHub release URL; **that release has not been uploaded**.
The archive under `artifacts/` is a tested local candidate, pinned by SHA-256.
Its existing developer build provenance has not been reproduced from clean
sources. Public download requires publishing a reviewed native bundle and
updating `bundle.json` first. No changes to `hrx-system` are needed by this crate.

## Use from a model

```toml
[dependencies]
hrx = { version = "0.1.0", path = "../hrx.rs" }
```

```rust,no_run
# fn main() -> hrx::Result<()> {
let mut stream = hrx::Stream::open()?;
let buffer = stream.allocate(4096)?;
stream.upload_queued(&buffer, 0, &[7; 4096])?;
let readback = stream.read_queued(buffer.binding())?;
assert_eq!(readback.wait(&mut stream)?, vec![7; 4096]);
# Ok(()) }
```

`Device` selects a device. `Stream` requires exclusive access for ordered work.
`Buffer` owns an allocation; `View` checks its extent and borrows its owner.
Buffers belong to their allocating stream. Transfers, dispatch bindings and graph
operations reject buffers from another stream, including on the same device.
Loading native code and dispatching a kernel are unsafe: the model must prove
that the grid, scalar values and accessed spans match its kernel. The model's
prepared operation types should contain that proof and expose safe operations.
`Constants` packs explicit u32/i32/f32/u64/i64/f64 widths in declaration order.

Allocation uses `hrx_allocator_allocate_buffer`, which does not flush pending
commands. Host uploads/downloads have both synchronous and owned queued forms.
`upload` and `read` drain all pending work and reclaim completed upload staging.
A stream retains mapped upload staging even if a submission token is forgotten.
Completed staging is reused in a pool capped at eight buffers and 64 MiB. Queued
uploads poll completion and apply backpressure at eight pending buffers or 64 MiB;
a single larger upload is allowed. Readbacks stay unmapped until completion, so
the native command buffer's storage retention makes an abandoned readback safe. A
submission flushes before polling, since native query alone ignores pending
commands. Scratch reuse is bounded and belongs to one stream. Fixed sequences
record explicit ordered dependencies and use native graph instantiate/replay;
addresses, constants and shapes are fixed. A sequence builder borrows its stream,
buffers and kernels through `finish`; the resulting `FixedSequence` owns native
resources and keeps the original stream alive. Native capture/update are not exposed.

`Gpu` preserves H3's existing Send, non-Sync API. `compat` preserves Krea's
address-based API while adding scoped streams and retention of direct-argument
allocations. It is a migration surface for trusted model kernels, not the
recommended API for new code. Raw blobs lack pointer metadata, so their escape
hatch conservatively retains that stream's registered allocations through synchronization.
Compatibility kernel loading and launch are unsafe: callers must validate the
code, dimensions, argument layout, addresses and aliasing for every invocation.
Use `Args::clear()` to reset raw mode and pointer metadata before reuse.

The review fixes change several source APIs: `Error` is a non-exhaustive enum
(`Error::Message` replaces the tuple constructor), `Args` is `Clone` without
`Copy`, `Submission::is_complete` needs a mutable token, and `FixedSequence` no
longer has a lifetime parameter. `Gpu::dispatch` is deprecated; migrate to
`Stream::dispatch` with explicit `Constants`. `loomrun` now respects scalar flag
widths exactly; use `--i64` for 64-bit Loom indices instead of relying on inferred
widening of `--i32` arguments. The minimum Rust version remains 1.88.

## Provisioning and installation

Default features are `download`, `loom`, and `ffi`. `runner` adds `hrx` and
`loomrun` binaries. `compat` adds the Krea migration API. CPU-only calls do not
initialize or download native code, and builds never download it.

```sh
cargo install --path . --features runner
# Local/offline preparation using the tested archive from this checkout:
hrx prepare artifacts/hrx-linux-x86_64-gfx1151.tar.gz
HRX_OFFLINE=1 hrx info
# Once the native release is published, `hrx prepare` downloads it on first use.
```

A model CLI installed with Cargo provisions its dependency when it first needs
the GPU/compiler. `cargo install` installs executables; consumers of a C library
build the model's cdylib and distribute it with its generated headers.

| Setting | Meaning |
| --- | --- |
| `HRX_RUNTIME_DIR` | Explicit trusted directory with libhrx.so, HSA and dependencies |
| `HRX_BUNDLE_MANIFEST` | Explicit JSON manifest for a pinned deployment/release |
| `HRX_CACHE_DIR` | Shared cache root (default XDG_CACHE_HOME/hrx or ~/.cache/hrx) |
| `HRX_OFFLINE` | Any value disables network provisioning |
| `LOOM_COMPILE` | Explicit compiler override; model API override takes precedence |
| `IREE_HAL_AMDGPU_LIBHSA_PATH` | Explicit compatible HSA provider override |
| `KREA2_RUNTIME` | Compatibility alias for HRX_RUNTIME_DIR |

The manifest pins the archive and every file. Downloads use HTTPS; `file://`
supports offline mirrors and deployment tests. Installation rejects archive
links, traversal, duplicates and undeclared files, verifies all contents, and
publishes a directory atomically under process locks. Prepared caches work
offline. A corrupt cache is rejected offline and rebuilt from the verified
archive online. Explicit directories without manifests are trusted overrides.

HSA is preloaded by absolute path before HRX. The library never changes the
process environment. An OS lock serializes global initialization across separate
Rust copies in multiple model cdylibs. Its PID-scoped lock lives in the private
`XDG_RUNTIME_DIR`, or a verified private per-user directory under `/tmp`, and is
removed on normal process exit. Failed initialization can be retried. All copies must select the same canonical
HRX path; incompatible choices fail with a diagnostic. Native libraries and the
global device registry stay resident until process exit; session teardown only
releases session resources. Rust objects are never exchanged across model DSOs.

## Compiler cache

`loom::Compiler` pins the compiler's content hash. `loom::Request` keys source,
symbol, backend, target, canonical configuration, cache schema, and compiler
identity with SHA-256. Per-key file locks coordinate compilation; successful
outputs are atomically published with artifact digests. A replaced compiler is
refused during a session. Compiler logs are captured and errors include a bounded
tail. Models retain prepared kernels and buffers outside the inference loop.

H3 and Krea retain their model-specific weight layouts, shape constraints,
checkpoint loading, samplers and prepared operation types. Their Loom source and
tokenizer assets now reside inside the consuming package, with compatibility
symlinks at the repository root. Vision Rust ports are separate work; the generic
`loomrun` can launch their existing HSACOs now.

## Portable model ABIs

Each model builds an rlib API and a cdylib with explicit `extern "C"` entrypoints
and `repr(C)` request types. cbindgen runs as a build dependency and generation
failures fail the build. `ffi::boundary` / `ffi_call!` supply call bodies without
hiding declarations from cbindgen. Shared helpers validate pointer alignment and
length arithmetic, copy bounded UTF-8 error messages, contain unwind panics,
reject poisoned handles, and provide a cancellation token. Actual pointer
validity and input/output aliasing remain the caller's C contract. Build with
unwinding if panic containment is required.

H3 preserves ABI 8 and adds `_ex` calls with caller-owned errors. Its TLS error
API remains available. Krea preserves both ABI 3 interfaces and caller-owned
errors. A minimal C integration test loads both model libraries plus the Krea
kernel test library concurrently. The H3 `examples/rustler` adapter demonstrates
an application-owned worker calling the Rust API directly. Rustler is optional;
no Elixir dependency appears in HRX or the model libraries.

## Validation

```sh
cargo test --all-features
cargo test --all-features --test gpu -- --ignored --test-threads=1
cargo clippy --all-features --all-targets -- -D warnings
cargo check --no-default-features
```

The GPU suite covers queued staging, forgotten completion tokens, checked views,
concurrent streams, dropped allocations, explicit mixed-width constants, Loom
compilation, and repeated native graph execution. Provisioning tests cover
corruption, offline behavior, path validation, and simultaneous processes.
See `VALIDATION.md` for consumer checks and `tools/bundle.py` for release packaging.
