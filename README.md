# hrx

Rust GPU execution and in-process Loom compilation for Linux x86_64. Native
libraries load at runtime; building the crate requires Rust 1.88 or later and no
native toolchain. GPU execution requires an AMD kernel driver and access to
`/dev/kfd` and the render device. The tested architecture is gfx1151.

The crate is unpublished. As of 2026-09-09, the pinned native release returns
404 to anonymous clients. Use a matching local archive or configure a mirror.
Dependency provenance and license notices also need completion before release;
see [THIRD-PARTY.md](THIRD-PARTY.md).

## Installation

```sh
cargo install --path . --features runner
hrx prepare /path/to/hrx-linux-x86_64-gfx1151.tar.gz
HRX_OFFLINE=1 hrx info
```

The archive must match [bundle.json](bundle.json). The older `34591d78…` bundle
lacks `libloomc.so` and cannot be used. Archives are not included in the crate.

Default features are `download`, `loom`, and `ffi`. `runner` adds the `hrx` and
`loomrun` binaries; `compat` adds the legacy address-based API. Builds and
CPU-only utilities do not provision native libraries. GPU or compiler use
resolves them on first use.

| Setting | Meaning |
| --- | --- |
| `HRX_RUNTIME_DIR` | Trusted directory containing libhrx.so, HSA and dependencies |
| `HRX_BUNDLE_MANIFEST` | Local path to a JSON bundle manifest, including its archive URL |
| `HRX_CACHE_DIR` | Cache root; defaults to `$XDG_CACHE_HOME/hrx` or `~/.cache/hrx` |
| `HRX_OFFLINE` | Any value disables network provisioning |
| `HRX_LOOM_LIBRARY` | Compiler library path; an explicit API argument takes precedence |
| `IREE_HAL_AMDGPU_LIBHSA_PATH` | Compatible HSA provider override |
| `KREA2_RUNTIME` | Alias for `HRX_RUNTIME_DIR` |

To use a mirror, save its manifest locally and set `HRX_BUNDLE_MANIFEST` to that
file. Archive URLs support HTTPS and `file://`. Installation verifies the archive
and each file, rejects links and unexpected entries, and atomically publishes
the cache directory under a process lock. Offline use requires a verified cache.
Explicit runtime directories without a manifest bypass integrity verification.

All model libraries in one process must select the same canonical HRX library
path. HSA loads by absolute path before HRX; the loader leaves the process
environment unchanged. Native libraries remain loaded until process exit.
Dropping a session releases its streams, allocations and compiler state.

## Buffers and streams

```toml
[dependencies]
hrx = { path = "../hrx.rs" }
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

`Device::open(index)` selects a GPU; `Device::stream()` creates an ordered stream.
`Buffer` owns an allocation and `View` borrows a checked span. `Stream` is Send
but not Sync, and mutating operations require exclusive access.

`upload` and `read` synchronize before transferring. `upload_queued` copies input
into owned staging and queues a device copy. Uploads batch until staging reaches
eight buffers or 64 MiB; one oversized upload is allowed. Completed staging is
reused using the smallest available buffer that fits. `read_queued` returns an
owned readback whose `wait` method synchronizes and returns the bytes.

`submit` flushes pending commands and returns a completion token. Dropping or
forgetting the token leaves queued storage owned by the stream. `scratch` and
`recycle` reuse allocations on the same stream, with a default 256 MiB cache
limit adjustable through `set_scratch_limit`.
Once an allocation has been shared, `recycle` rejects both its original handle
and all aliases, even after the other handles are dropped.

A fixed sequence records ordered fills, copies and dispatches for graph replay.
The builder borrows its buffers and kernels until `finish`; the resulting
`FixedSequence` retains native resources. Replay uses the original stream and
fixed addresses, constants and dimensions. Build a new sequence to change them.

## Cross-stream dependencies

Events order work between streams on the same device without a host wait:

```rust,no_run
# fn main() -> hrx::Result<()> {
let device = hrx::Device::open(0)?;
let mut upload = device.stream()?;
let mut compute = device.stream()?;
let weights = upload.allocate(4096)?;
let output = compute.allocate(4096)?;

// Safety: the event below orders the upload before the shared buffer is read.
// This example makes no further accesses to weights on the upload stream.
let shared = unsafe { weights.share_on(&compute)? };
upload.upload_queued(&weights, 0, &[7; 4096])?;
let ready = upload.record_event()?;
compute.wait_event(&ready)?;
compute.copy(&output, 0, &shared, 0, 4096)?;
compute.synchronize()?;
# Ok(()) }
```

`share_on` is unsafe because all conflicting accesses through shared handles
must be ordered, including host transfers and graph replay. Before overwriting
weights for another iteration, record an event after their last compute use and
make the upload stream wait for it. Unshared buffers are accepted only by their
owning stream. Events are immutable completion points; dropping one after
queuing a wait preserves that dependency.

## Kernels and compilation

Loading code and dispatching kernels are unsafe. Use trusted code, match its
argument layout, and ensure every accessed address stays within a live allocation.
Binding lengths are checked on the host but do not restrict device addressing.
Binding dispatch validates binding count, constant byte length and compiled workgroup
dimensions when present in export metadata. `Constants` packs explicit scalar
widths in declaration order.

Choose the compiler target from the device:

```rust,no_run
# fn main() -> hrx::Result<()> {
let device = hrx::Device::open(0)?;
let compiler = hrx::loom::Compiler::with_options(None, hrx::loom::CompilerOptions {
    target: device.target().clone(),
    ..Default::default()
})?;
let module = compiler.module(&std::fs::read_to_string("kernel.loom")?);
let mut spec = hrx::loom::Specialization::new("my_kernel");
spec.config.insert("model.width".into(), "256".into());
let artifact = module.compile(&spec, &hrx::bundle::cache_root()?.join("kernels"))?;
let stream = device.stream()?;
// Safety: this application trusts the source and selected compiler.
let kernel = unsafe { stream.load_artifact(&artifact)? };
# let _ = kernel;
# Ok(()) }
```

Executable loading uses the device architecture and rejects mismatched artifact
targets. The default compiler target is gfx1151. Other architectures require
native libraries built with that Loom target enabled and are not yet tested here.
Device names such as `gfx90a:sramecc+:xnack-` use their base architecture,
`gfx90a`. Explicit `Target` values accept bare architecture keys only.
The pinned runtime selects the wave size from the executable; its dispatch
subgroup field is unused.

`Compiler` loads `libloomc.so` in process. It can compile without a GPU when given
a target directly. Modules share parsed source indexes across specializations.
`CompilerOptions` limits concurrent workspaces (up to four by default) and cached
modules (64 by default; zero disables module caching). Evicted modules remain
valid while callers hold them. `Compiler::trim` releases cached module references
and idle workspace scratch.
Module eviction is arbitrary; the cache does not track recency.

Artifact cache keys include source, export, target, configuration, compiler hash
and report mode. Cache reads verify bytes and metadata; writes use per-key locks
and atomic publication. Artifacts own their bytes and diagnostics independently
of the compiler. Replacing a loaded compiler at the same path is rejected; use a
new path or restart the process. Build instructions and source pins are in
[patches/loom](patches/loom/README.md).

GPU timestamp profiling is not available. `loomrun` reports host elapsed time,
including submission and synchronization. Native event timing also uses host
timestamps and is not exposed as GPU timing.

## Compatibility and C APIs

New code should use `Stream`. `Gpu` retains the older stream API; its
`dispatch` method is deprecated in favor of explicit `Constants`.

With `compat`, `Device::enter` selects a stream for the current thread.
`Device::open` scans for the first GPU matching the default compiler target;
`open_for_target` selects another architecture and `open_device` selects an index.
`Args` tracks typed pointer arguments so dispatch can retain their allocations.
`Args::raw` lacks that metadata and scans all live allocations on the selected
stream per launch. Use `Args::clear` before reusing the argument builder.
Compatibility loading and dispatch have the same trust and addressing obligations
as the main API. API migrations are listed in [CHANGELOG.md](CHANGELOG.md).

The `ffi` feature provides panic containment, checked spans, bounded error
messages, handle locking and cancellation for application-defined C entrypoints.
`ffi::boundary` wraps a call body; the application defines its exported functions,
request structs and headers. Callers remain responsible for valid pointers and
aliasing. Panic containment requires an unwinding build.

## Development

```sh
cargo test --all-features
cargo test --all-features -- --ignored --test-threads=1
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps
bash scripts/check-feature-matrix.sh
```

[VALIDATION.md](VALIDATION.md) records the tested bundle, results and CI setup.
`hrx pack RUNTIME OUTPUT URL REVISION [TARGET]` creates a release archive from a
staged directory. Required provenance and notice files are described in
[THIRD-PARTY.md](THIRD-PARTY.md).
