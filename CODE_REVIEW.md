# Code review: `hrx.rs`

Reviewed at commit `2a9048f` (2026-09-08). Scope: the full crate — `src/`, `tests/`,
`examples/`, `Cargo.toml`. Verification run during the review:

```sh
cargo clippy --all-features --all-targets          # clean
cargo check --no-default-features                  # clean
cargo clippy --all-features --all-targets -- -W clippy::pedantic -W clippy::nursery
```

Findings 1 and 2 were confirmed by compiling the reproducers shown against the crate.

The overall shape is strong. The ownership model — `Arc<Inner>` keeping the device alive
underneath buffers and kernels, `View<'a>` borrowing its buffer, `PhantomData<Cell<()>>`
to express `Send + !Sync` — is genuinely well designed, and the safety comments explain
*why* rather than restating the code. The provisioning and compiler-cache paths are
careful about locking, atomicity, and path traversal. What follows is mostly about places
where the design has drifted since those comments were written, plus idiom and
performance work.

---

## Soundness / correctness

### 1. `Stream::sequence<'a>` has a lifetime unconstrained by any input

`src/runtime.rs:919`

```rust
pub fn sequence<'a>(&self) -> Result<SequenceBuilder<'a>> {
```

`'a` appears nowhere in the arguments, so the caller picks it. Confirmed to compile
against the crate:

```rust
fn escape() -> hrx::Result<hrx::FixedSequence<'static>> {
    let stream = hrx::Stream::open()?;
    let builder: hrx::SequenceBuilder<'static> = stream.sequence()?;
    Ok(builder.finish()?)   // stream dropped here; the sequence survives
}
```

`Buffer` and `Kernel` both hold an `Arc<Inner>` precisely to stop this, but
`SequenceBuilder` (`runtime.rs:915`) and `FixedSequence` (`runtime.rs:1057`) store a bare
`sys::Device` and nothing keeps the stream alive.

Variance saves the *buffer* half — with `'a = 'static` you cannot produce
`View<'static>`, so an escaped sequence is necessarily empty — but the stream/device half
is unguarded.

**Fix (do both):** tie the lifetime to the receiver
(`fn sequence(&self) -> Result<SequenceBuilder<'_>>`) and give both types the
`Arc<Inner>` that every other resource carries.

### 2. The `unsafe impl Sync for Buffer` justification is stale

`src/runtime.rs:479-485`

> every transfer and fill goes through `&Gpu`, which is deliberately not `Sync`, so two
> threads cannot reach the same allocation concurrently through this API.

That was true when `Gpu` was the only entry point. `Device` is `Clone` and
`Device::stream()` can be called repeatedly, each call producing a `Send` `Stream` with
its own native stream over the same device. This compiles today:

```rust
let dev = hrx::Device::open(0)?;
let (mut a, mut b) = (dev.stream()?, dev.stream()?);
let buf = a.allocate(4096)?;
std::thread::scope(|s| {
    s.spawn(|| { let _ = a.upload(&buf, &[1u8; 4096]); });           // hrx_synchronous_h2d
    s.spawn(|| { let mut o = [0u8; 4096]; let _ = b.read(buf.binding(), &mut o); });
});
```

Two threads calling `hrx_synchronous_h2d` / `hrx_synchronous_d2h` on one `hrx_buffer_t`.

This is not Rust-level UB — there is no `&mut` aliasing — but it matters concretely
because `hrx_buffer_get_device_ptr` appears to be able to populate a mapping lazily. If
`hrx_buffer_s` has *any* non-atomic interior state, this is a real data race, and the
comment asserting otherwise is now load-bearing and wrong. It is also a silent
correctness footgun for users: ordering between the two streams is undefined.

**Fix:** either re-derive the claim against the current API surface and rewrite the
comment, or add a provenance check to `Stream`'s buffer-taking methods the way `recycle`
(`runtime.rs:819`) and `Readback::wait` (`runtime.rs:1118`) already do.

### 3. Staging and readback use contradictory retention models

`src/runtime.rs:660`, `src/runtime.rs:1071`

`upload_queued` pushes staging into `Stream::staging` because:

> Storage is retained by the stream, not by a completion token whose Drop could be
> skipped. Forgetting a token can never free in-flight host memory.

But `Readback`'s doc says:

> Dropping it before completion is safe: the native command buffer retains its unmapped
> storage.

Dropping a `Readback` runs `hrx_buffer_release` on a buffer that is the *destination* of
an in-flight copy. If native retention holds for the readback destination, it holds for
the staging source too and the `staging` Vec is redundant complexity (plus an unbounded
memory hold — see finding 11). If it does not, the `Readback` drop path is unsound.

**Fix:** one of the two is wrong. Pick a model, document it once, and apply it to both.

### 4. `Stream::drop` waits for staging but drops scratch blind

`src/runtime.rs:900`

```rust
impl Drop for Stream {
    fn drop(&mut self) {
        if !self.staging.is_empty() && self.gpu.sync().is_err() {
            std::mem::forget(std::mem::take(&mut self.staging));
        }
    }
}
```

`self.scratch` is dropped by the normal field drop with no synchronize, releasing buffers
that may still be referenced by recorded-but-unflushed commands. Same class of question
as finding 3 — apply the same answer to both fields.

### 5. `Stream::upload` / `Stream::read` silently drain the whole stream

`src/runtime.rs:677-682`

They delegate to `Gpu::h2d` / `Gpu::d2h_ref`, which call `self.sync()` first. This is
documented on `Gpu` ("The synchronous transfers bypass the stream's pending commands, so
the stream is drained first") but not on `Stream`, and not in the README's `Stream`
section. So an innocuous-looking `stream.upload(...)` in the middle of a queued pipeline
is a full pipeline stall.

Worse: that internal sync makes all pending staging idle but does **not** clear
`self.staging` — only `synchronize()` does — so the retained memory outlives its need.

**Fix:** at minimum document it on `Stream`. Better, route them through the queued path,
or clear `staging` after the internal sync.

### 6. Initialization failure is cached permanently

`src/runtime.rs:79` and `src/compat/device.rs:87`

```rust
static READY: std::sync::OnceLock<Result<()>> = std::sync::OnceLock::new();
READY.get_or_init(|| { ... }).clone()
```

A transient failure — driver not up yet, `/dev/kfd` not yet permissioned, bundle mid-download
by another process — is memoized for the life of the process with no way to retry. Note
that `sys::load()` (`sys.rs:188`) *does* retry on failure, so the two layers disagree
about whether initialization is retryable. `compat::DEFAULT` has the same shape.

**Fix:** a `Mutex<Option<...>>` that only memoizes success.

### 7. `Gpu::dispatch` infers scalar width by division and can silently mispack

`src/runtime.rs:422-439`

```rust
let width = size / scalars.len();
if width != 4 && width != 8 { /* reject */ }
for (i, v) in scalars.iter().enumerate() {
    constants[i * width..i * width + 4].copy_from_slice(&v.to_le_bytes());
}
```

Two scalars against an 8-byte constant block gives `width = 4` — but a kernel taking a
single `u64` is also "8 bytes", and there is no way to tell them apart from a `&[u32]`.
Mixed widths (one `u32` + one `u64` = 12 bytes) are either rejected or mispacked
depending on how many `u32`s the caller splits them into. The bounds arithmetic is
correct; the *layout inference* is not decidable from the inputs.

`Constants` (`runtime.rs:864`) is the correct API and already exists.

**Fix:** mark `Gpu::dispatch` `#[deprecated]` pointing at `Stream::dispatch` rather than
leaving a silently-wrong path in the legacy surface.

### 8. `SequenceBuilder::dispatch` treats bindings and constants inconsistently

`src/runtime.rs:992-1021`

`kernel: &'a Kernel` and `bindings: &[View<'a>]` are lifetime-tied to the builder, but
`constants: &Constants` is borrowed only for the call while `attrs.constants` stores a raw
pointer into it. Either `hrx_graph_add_kernel_node` copies its attributes (in which case
the `'a` on bindings is unnecessary) or it retains them (in which case the constants
pointer dangles on return).

That `attrs` is itself a stack local passed by pointer argues for "copies" — but then the
code should say so, because as written the two halves imply opposite contracts.

---

## Security

### 9. Predictable lock path in a world-writable directory

`src/sys.rs:358-365`

```rust
let path = std::path::PathBuf::from(format!(
    "/tmp/hrx-{}-{}.lock",
    unsafe { libc::getuid() },
    std::process::id()
));
```

`O_NOFOLLOW` and `mode 0o600` (`bundle.rs:60-67`) are good instincts, but
`create(true).truncate(false)` on a fully predictable path in a world-writable directory
means a local attacker can pre-create the file with permissive modes and control its
contents.

The blast radius is bounded: `load()` only ever *rejects* based on the contents
(`sys.rs:217`), never loads a library path from it. So this is a local denial of service,
not code execution.

Separately: these files are per-PID and never unlinked, so they accumulate in `/tmp`
indefinitely. Unlike the cache locks — where the comment "Never unlink a lock file while
another process might be waiting on its inode" correctly applies — this one is
PID-scoped and *can* safely be cleaned up at exit.

**Fix:** use `$XDG_RUNTIME_DIR` (already `0700` and per-user) with `/tmp` as fallback,
and unlink the PID-scoped file on exit.

---

## Performance

### 10. `Readback::wait` zeroes the whole buffer before overwriting it

`src/runtime.rs:1122`

```rust
let mut bytes = vec![0; self.buffer.bytes];
...
std::ptr::copy_nonoverlapping(pointer.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
```

Two full passes over memory for every readback. Use `Vec::with_capacity` +
`spare_capacity_mut()` + `set_len`, or `MaybeUninit`. `runner.rs:306` has the same
pattern before `d2h`.

### 11. `upload_queued` allocates a fresh staging buffer per call and never reuses one

`src/runtime.rs:698`

Each call allocates a host-visible device buffer of exactly `bytes.len()`, and staging is
freed only at `synchronize()`. A model streaming a large checkpoint in chunks accumulates
one pinned allocation per chunk *and* pays an allocator round-trip per chunk.

There is already a scratch pool for ordinary buffers (`runtime.rs:802`). Extending the
same idea to staging — a small ring of reusable staging buffers, reclaimed on submit —
removes both costs.

### 12. The Loom compiler binary is hashed three times per compilation

`src/loom.rs:278`, `src/loom.rs:320`, `src/loom.rs:358`

`Compiler::resolve` digests it, then `compile` digests it again before and after the
subprocess. Two of those are a deliberate tamper check and should stay. But `resolve()`
being a full read of the compiler makes constructing a `Compiler` surprisingly expensive
— worth a note in the docs at least.

### 13. Smaller performance items

- **Eager error formatting on the success path.** `runtime.rs:345`, `:355`, `:360`:
  `check(status, &format!("loading {}", path.display()))` allocates and formats even when
  the status is `NULL`. Take `impl FnOnce() -> String` or `impl Display` instead of
  `&str`. Cold path, but it is the kind of signature that spreads.
- **`Stream::scratch` linear-scans the pool** (`runtime.rs:802`); a
  `BTreeMap<usize, Vec<Buffer>>` keyed by size makes it `O(log n)`. `set_scratch_limit`
  (`runtime.rs:828`) also evicts via `pop()`, discarding the most-recently-recycled
  buffer — the one most likely to be reused next.
- **`Device::open`** (`runtime.rs:637`) opens a full `Gpu` — creating a native stream —
  purely to validate the index, then drops it; `stream()` then creates another.
- **`bundle::file_digest`** (`bundle.rs:29`) uses a 64 KiB stack array; clippy's
  `large_stack_arrays` flags it at pedantic level. Heap-allocate it, or `io::copy` into
  the hasher.
- **`api()`** (`sys.rs:366`) performs an atomic load plus `expect` on every FFI call.
  Negligible per dispatch, but `Gpu` could cache `&'static Api` at open time.

---

## API design and idiom

### 14. `pub struct Error(pub String)` is the biggest API-quality gap

`src/lib.rs:18`

```rust
pub struct Error(pub String);
impl From<std::io::Error> for Error { fn from(e: std::io::Error) -> Self { Self(e.to_string()) } }
impl From<String> for Error { ... }
impl From<&str> for Error { ... }
```

Every error allocates and discards structure:

- `From<io::Error>` throws away `ErrorKind`, so a caller cannot distinguish "bundle not
  found" from "permission denied" from "disk full" without string matching.
- There is no `source()`, so error chains are flat.
- The public tuple field pins the representation forever.
- `From<String>` / `From<&str>` make `?` convert *anything* into an error, which is how
  context gets lost silently.

For a crate that two model workspaces depend on, move to a `#[non_exhaustive]` enum with
`thiserror`, keeping `Display` output identical so nothing downstream breaks.

### 15. No public type in `runtime.rs` implements `Debug`

Zero `derive(Debug)` in the file. `Buffer`, `View`, `Stream`, `Kernel`, `Constants`,
`Device`, `Readback`, `Submission`, `FixedSequence` and `SequenceBuilder` are all opaque
in a `dbg!`, and in any downstream `#[derive(Debug)]` struct that contains one.

This is Rust API guideline C-DEBUG. `compat::Kernel` (`compat/kernel.rs:347`) does it
right with a hand-written impl that avoids exposing the raw handle — the same approach
works for the rest.

### 16. Missing `#[must_use]`

- `Submission` (`runtime.rs:842`) — dropping it silently discards the fence.
- `Scope` (`compat/device.rs:91`) — dropping it immediately un-does `enter()`.
- `Readback` (`runtime.rs:1073`).
- `Constants::push` (`runtime.rs:880`) — returns `Result<&mut Self>`.
- `View::len` / `View::is_empty` (`runtime.rs:509`).

### 17. `sys.rs` is ~535 lines of mechanical repetition out of 725

Three parallel lists that must be kept in sync by hand:

1. `Api` struct fields (`sys.rs:108-186`)
2. 40 × `library.get(b"...\0").map_err(|e| Error(e.to_string()))?` (`sys.rs:235-343`)
3. 40 × identical `pub unsafe fn` forwarding wrappers (`sys.rs:371-655`)

This is exactly the failure mode the module doc says it is avoiding bindgen for. One
`macro_rules!` taking `name(args) -> ret` and emitting all three collapses this to roughly
80 lines and makes adding a symbol a one-line change. The stated goal — "the whole surface
this project uses is readable in one file" — is better served by the short version.

### 18. `compat::Args` is a 536-byte `Copy` type

`src/compat/args.rs:202` — `[u8; 256]` + `[usize; 32]` + three fields, measured at 536
bytes. `Copy` on something that large invites accidental memcpys; `Clone` alone would
force them to be visible at the call site. `Constants` at 264 bytes is borderline too.

Also: `Args::raw` sets `opaque = true` permanently with no way to clear it
(`args.rs:266`). A reused `Args` that once held a raw blob keeps forcing
`compat::Device::dispatch` down the conservative "retain every live allocation" path
(`compat/device.rs:312`).

### 19. Smaller API and idiom items

- `Device` (`runtime.rs:633`) wraps a single `i32` and derives `Clone` — make it `Copy`.
- `View::len(&self)` / `is_empty(&self)` take a reference to a `Copy` type; take `self`.
- The `scalars!` macro (`runtime.rs:898`) is a single ~200-column line — reflow it.
- `runner::main` returns `ExitCode` but `bin/hrx.rs:479` uses `std::process::exit(1)`,
  which skips destructors. Return `ExitCode` from both.
- `pub mod sys` exposes 40 `pub unsafe fn` that panic if the runtime is not loaded.
  Consider `#[doc(hidden)]` or a `raw` feature gate so they are not part of the semver
  surface by accident.
- `bundle::cache_root()` (`bundle.rs:39`) calls `create_dir_all` — a getter with a
  filesystem side effect. Split the two, and consider `0o700` on the cache root.
- `Manifest::install` (`bundle.rs:165`) and `Compiler::compile` (`loom.rs:369`)
  `fs::rename` a live `TempDir`, leaving its `Drop` to fail silently on a path that no
  longer exists. Use `TempDir::keep()` to make the handoff explicit.
- `Compiler::compile` writes `kernel.sha256` *after* `sync_all`-ing the artifact
  (`loom.rs:361-365`), so the digest file is not durable; and the staging directory is not
  fsynced before the rename.
- `examples/allocator_bench.rs:572` hardcodes `queued[500]` for the median, correct only
  because the loop constants happen to line up. It also `assert!`s on `is_ok(...)`,
  leaking the status object on failure rather than calling `hrx_status_ignore`.
- `runner.rs:424` and `:464` use `_ =>` where the enum has exactly two variants; clippy's
  `match_wildcard_for_single_variants` flags both. Naming the variant means a future
  third variant becomes a compile error.
- `Cargo.toml` has no `[lints]` table. Clippy is already clean at the default level —
  `[lints.rust] missing_docs = "warn"` plus a few `clippy::pedantic` selections would lock
  that in. There are real doc gaps (`Stream::open`, `Error`, most of `sys`). Also missing
  `repository`, `readme`, `keywords`, `categories` for a crate intended for publication.

---

## Suggested order

1. **Findings 1, 2, 3, 4** — the ownership and lifetime claims that the code no longer
   backs up. These are the ones where the comments actively mislead a future reader.
2. **Finding 14** (`Error`) — the cost of changing it grows with every consumer.
3. **Findings 5, 6, 7** — behaviour surprises with concrete failure modes.
4. **Findings 10, 11** — the two performance items on real data paths.
5. **Findings 15–19** — mechanical; finding 17 alone removes roughly 450 lines.
