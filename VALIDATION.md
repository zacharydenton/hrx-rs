# Validation

## Local results, 2026-09-09

Tested on Linux x86_64 with a gfx1151 GPU. Archive installation was checked with
an empty cache. Native tests use the verified bundle cache with `HRX_OFFLINE=1`
and without runtime, compiler or HSA library overrides.

| Check | Result |
| --- | --- |
| CPU tests | 31 passed |
| Rust API doctests | 3 passed |
| Ignored runtime and compiler tests | 15 passed |
| Feature matrix | All 32 subsets build, including on Rust 1.88 |
| Clippy | Passes with warnings denied on stable and Rust 1.88 |
| Rustdoc | Passes with warnings denied |
| Package contents | Internal review documents excluded |

The native tests cover queued transfers, staging batching and reuse, event
ordering between streams, rejection of shared scratch allocations, argument
packing, workgroup validation,
resource retention, graph replay, compiler caching and diagnostics.

The tested archive SHA-256 is:

```text
c151a978eff1c7def5c54b9acfd595cc1e8ca21b793b4613864a18281d846bb1
```

A cached directory matched every file digest in `bundle.json`. Repacking it
reproduced this archive hash and replaced the stale `34591d78…` archive in
`artifacts/`. The anonymous release URL still returns 404. Dependency provenance
and notices remain incomplete; see [THIRD-PARTY.md](THIRD-PARTY.md).

To repeat the bundle check with an empty cache:

```sh
export HRX_CACHE_DIR="$(mktemp -d)"
unset HRX_RUNTIME_DIR KREA2_RUNTIME HRX_LOOM_LIBRARY HRX_BUNDLE_MANIFEST
unset IREE_HAL_AMDGPU_LIBHSA_PATH
cargo run --all-features --bin hrx -- prepare /path/to/hrx-linux-x86_64-gfx1151.tar.gz
HRX_OFFLINE=1 cargo run --all-features --bin hrx -- info
HRX_OFFLINE=1 cargo test --all-features -- --ignored --test-threads=1
```

These results cover this crate. Consumer model quality and performance were not
retested. Only gfx1151 was exercised; GPU timestamp profiling is not implemented.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests using stable
Rust and Rust 1.88. It checks formatting, builds all targets, runs clippy, CPU tests
and rustdoc, checks every feature subset, and lists package contents.

`.github/workflows/gpu.yml` runs on pushes or manual dispatch. It needs a
registered runner with labels `self-hosted`, `linux`, `x64`, and `gfx1151`, access
to `/dev/kfd` and the render node, rustup, curl, and HTTPS access. Runner
registration is still outstanding.

The GPU job provisions an empty cache and runs every ignored test, including
compiler and library tests. It fails if provisioning fails. The optional
repository variable `HRX_BUNDLE_MANIFEST` is an HTTPS URL to a mirror manifest;
the workflow downloads it and sets the process environment variable to its local
path. Without a mirror, the job uses the currently unavailable pinned release.
