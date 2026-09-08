# Review fixes

Addresses `CODE_REVIEW.md` and `RUST_CODE_REVIEW.md` from 2026-09-08.
The original reviews remain unchanged. Source API migration notes are in README.md.

| Finding in CODE_REVIEW.md | Resolution |
| --- | --- |
| 1, sequence lifetime | Recording borrows its stream and resources. Both sequence types retain `Arc<Inner>`. Instantiation retains native HAL resources, so `FixedSequence` is now owned without a lifetime parameter. |
| 2, cross-stream access | Every buffer/view operation checks its original stream, including legacy Gpu calls, dispatch bindings and sequence replay. Compatibility addresses have the same ownership check. |
| 3–4, retention and drop | Documented the native distinction: command buffers retain HAL storage, while a mapped HRX wrapper unmaps on final release. Upload wrappers survive completion; readbacks remain unmapped until wait. Stream drop drains staging and scratch, leaking them if completion fails. |
| 5, synchronous transfers | Upload/read document the full drain, synchronize once, and reclaim completed staging. |
| 6, initialization retry | A shared mutex cache stores successful initialization only, for both runtime and compatibility default device. |
| 7, scalar packing | Deprecated Gpu::dispatch. The CLI uses Stream/Constants with explicit --i32, --i64 and --f32 widths. |
| 8, graph constants | Documented native graph.c's descriptor/constant copies and borrowed HAL resources until instantiation. Tested constants going out of scope before finish. |
| 9, lock files | Prefer private XDG_RUNTIME_DIR; validate ownership, modes and symlinks, with a private per-user /tmp fallback. Remove the PID lock on normal exit. |
| 10–11, readback/staging cost | Readback initializes spare Vec capacity directly. Upload staging reuses completed buffers, bounds pending/cached storage, applies backpressure, and reclaims on successful polling or waits. |
| 12, compiler hashing | Documented full-binary hashing at resolution and reuse of Compiler; kept both tamper checks. |
| 13, smaller costs | Lazy formatting, size-indexed scratch pooling and largest-class eviction, device validation without creating a throwaway stream, and heap-backed hash scratch storage. Kept the single global Api lookup rather than adding a second cached handle to every resource; no measured benefit justified that optional change. |
| 14, errors | Non-exhaustive thiserror enum preserves IO kinds, JSON/library/download sources, native status codes and contextual chains. |
| 15–16, Debug/must_use | Runtime public structs implement Debug; completion/readback/scope guards and the suggested methods carry must_use annotations. |
| 17, native declarations | One macro declaration list generates the function table, symbol resolution and forwarding wrappers. |
| 18, Args | Removed Copy and added clear() to reset bytes, raw mode and pointer metadata. |
| 19, API and housekeeping | Device is Copy, View accessors take self, scalar macro is formatted, CLI returns ExitCode, sys is doc-hidden, cache path lookup is side-effect-free, new cache directories are private, TempDir handoff is explicit, artifact/digest/directories are fsynced, benchmark statuses are consumed and medians use actual sample counts, matches name variants, and publication metadata/documentation lints are enabled. |

All five findings in `RUST_CODE_REVIEW.md` are covered: compatibility loading and
launch are unsafe with invocation contracts and compile-fail tests; foreign-stream
access is rejected; completed staging is reclaimed; manifests are validated at
parse/verify/prepare/install and archive entry names are independently checked;
allocation registry entries are removed on final allocation drop instead of
scanning unrelated sessions during every synchronization.

The additional modernization suggestions are covered by structured errors,
constant C string literals and testing the actual checked_span helper. Kept flock
to preserve the declared Rust 1.88 minimum; File::lock would require raising it.

Native retention was checked against the local libhrx sources in
`hrx-system/libhrx/src/libhrx/{buffer,transfer,stream,graph,graph_exec}.c`, and the
GPU regression suite exercises the pinned runtime bundle. This does not establish
clean-build provenance for that bundle; the existing release caveat still applies.

Validation after the fixes:

- `HRX_OFFLINE=1 cargo test --all-features -- --include-ignored --test-threads=1`: 34 tests plus 3 compile-fail doctests passed, including 8 hardware tests and the CLI scalar/readback/lock-cleanup checks.
- `cargo clippy --all-features --all-targets -- -D warnings`: passed.
- `cargo check --no-default-features`: passed.
- `cargo +stable check --all-features`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

Rust 1.88 itself and the sibling model workspaces were not retested. Their earlier
validation record predates these source API changes; consumers must apply the
migration notes before repeating that validation.
