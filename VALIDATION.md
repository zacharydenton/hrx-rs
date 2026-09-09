# Validation record

## Compiler update, 2026-09-09

Bundle `34591d78d625f9c696657820d04615f3f55a134010a9354c0e455a4c2e60caa0`
contains Loom built from `hrx-system` commit
`9e4fff00d244a8b5300d03569addd019ec8e262f`. Compiler SHA-256:
`74a0c9dc5f387e89b85a3cd9d2000644dc0e20a0657d9fe79dfcd627ff5ecdb6`.
Every runtime library retains its previous digest.

- `cargo test --all-features`: all non-ignored tests and doctests pass.
- The archive installs through the normal digest-verifying provisioner.
- All 15 H3 GPU kernel tests pass using that installed bundle, with bitwise
  comparisons against the pre-consolidation sources and independent CPU oracles.
- 550 available Loom fixture suites pass. Four other source-low fixture suites
  have the same failures when rebuilding the unchanged parent commit; tests for
  unbuilt optional executables were excluded.
- Exact i4/i8 encoding configuration and dependent vector template arguments
  compile through native emission; the emitted WMMA forms are checked.

The native release is `native-9e4fff00d244` in the private GitHub repository.
Authenticated archive preparation is documented in the README. Runtime library
build provenance still has the limitations recorded below.

## Original runtime archive

Validated locally on Linux x86_64 / gfx1151 on 2026-09-08. Native archive SHA-256:
`750f265ce4fd6a194fbac12a795c96cb19cc9ed3696fd5123c5edd5589a4cd05`.
This is a byte-pinned candidate made from the existing staged runtime, not a
reproducible build attestation. No native source or build was changed.

- Shared crate: unit/contract/provisioning tests, including simultaneous
  installers in separate processes; explicit GPU suite; no-default-feature
  build; Clippy with warnings denied.
- H3: workspace tests and all-feature Clippy; release cdylib; GroupNorm (three
  amplitudes) and four attention layouts against the existing numerical oracle.
- Krea: workspace tests and Clippy; release cdylibs; BF16 GEMM shapes/transposes,
  activations, norm/SiLU, convolution, upsampling, embedding, batched/causal
  softmax, Euler and guidance against existing numerical oracles.
- Krea real checkpoint, 4115 tokens, one block: same-input update cosine
  `0.999998`; repeated composition exact; update cosine versus BF16 `0.99997`.
  This is not a full-model or full-generation quality comparison.
- C: compile one client against both generated headers, dlopen independent H3,
  Krea and testkit DSOs, check legacy versions/new errors, initialize different
  crate copies concurrently, upload/read back and repeatedly drop/reopen sessions.
- Elixir: build the Rustler 0.38 adapter, open H3's Rust API on a model worker,
  receive a typed response from BEAM. No Python is used in this path.
- ELF dependencies of both release model libraries: libc, libm, libgcc_s and the
  platform loader. No libhrx, ROCm, Python, Torch, libtorch or OpenSSL DT_NEEDED.
- Both model CLIs installed successfully with `cargo install --path cli` into
  isolated temporary prefixes. H3 defaults to embedded sources and a per-user
  cache rather than writing beside the installed executable.
- Cargo package contents contain the embedded model source/tokenizer assets.
  The shared crate packages successfully without native binaries or build tools.

The archive and manifest can be exercised without a hosted release:

```sh
cargo run --features runner -- prepare artifacts/hrx-linux-x86_64-gfx1151.tar.gz
HRX_OFFLINE=1 cargo test --all-features --test gpu -- --ignored --test-threads=1
```

Reproduce the cross-library test from the Krea repository:

```sh
cc -Wall -Wextra -Werror -O2 -I../minimax-h3-loom/include -Ibuild/include \
  ../hrx.rs/tests/c/models.c -ldl -pthread -o /tmp/hrx-models-smoke
HRX_OFFLINE=1 /tmp/hrx-models-smoke \
  "$PWD/../minimax-h3-loom/target/release/libh3.so" \
  "$PWD/target/release/libkrea2.so" "$PWD/target/release/libnative_ops_test.so"
```

`cargo run --release --example allocator_bench` measured 1000 interleaved idle
1 MiB allocations after 100 warmup iterations: native stream allocation median
7.069 µs; shared Rust device allocator median 0.550 µs. This demonstrates removing
the stream flush/allocation wait; it is not an inference speedup measurement.

Public distribution still requires a public repository or mirror, crate
publication, and verification of the runtime libraries' build provenance and
redistribution notices. The compiler update above publishes a pinned archive
for repository collaborators.
