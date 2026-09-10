# Follow-up to the GPU/NPU code review

The review fixes and native runtime provisioning are implemented and tested locally.
Publishing the staged native assets remains a release step.
Cross-context NPU binding is intentional and has an explicit device/group contract.

| Finding | Disposition |
| --- | --- |
| 1. Host→GPU invalidates host writes | Fixed: sync direction considers the last writer. Host writes flush before either GPU or NPU consumption. |
| 2. GPU kernels require unpublished ABI 1 | Fixed in the pinned ABI 1 GPU bundle, built and verified locally. Its staged release assets await publication. Base alignment checks remain enforced. |
| 3. Cross-context NPU bindings | Retained: host-only/imported BOs are device-global; device and memory-group compatibility are the required checks. Owning and dispatch contexts stay alive. |
| 4. Scheduler lock spans cache maintenance | Fixed: reserve the host lease under the scheduler lock, release the lock, then sync. Failure rolls back the lease and notifies waiters. |
| 5. Interop symbols outlive temporary library handle | Fixed: one cached typed API retains its library for the lifetime of the function pointers; resolution failures are cached too. |
| 6. Doctor aborts on interop probe error | Fixed: print the probe error and continue the remaining diagnostics. |
| 7. Percentile sample indices | Standardized these Rust benchmarks on nearest-rank quantiles, with a shared helper and tests. |

## Native distribution gap

The new `bundle.json` pins a full ABI 1 build, preserving allocation-address
validation for GPU-local kernel bindings. `npu-bundle.json` pins the shim, matching
XRT libraries, the XDNA plugin and libuuid. `hrx prepare` provisions both when
compiled with `npu`; APIs also provision their runtime on first use. No separate
XRT/Python/SDK installation is required for precompiled programs. Linux drivers,
firmware, device permissions and standard host libraries remain prerequisites.

Both binary archives and both source archives are staged locally. Automatic
public downloads require publishing those exact assets. The existing crates.io
0.2.0 release does not include these changes.

## Context compatibility

`raw::Context::run_start_shared` explicitly permits a BO from another context when
it is on the same physical device and in a compatible bank. The coordinated layer
checks the device ordinal and actual kernel argument memory bank, owns only host-only or
imported BOs, and retains the contexts through prepared dispatch. Requiring context
identity would prohibit this supported sharing path.

XRT group IDs contain a memory bank in bits 0–15 and a context slot in bits 16–23.
The graph compares the bank, matching XRT binding validation, while allocation
continues to use the full group ID. Comparing the whole ID incorrectly rejected
compatible buffers from another context on the new runtime.

The hardware test drops the cached program before loading a second one, while
the BOs keep their original context alive. This creates two independent contexts
from the passthrough image
and executes the second context's kernel with buffers allocated through the first.
Both NPU-local and GPU/NPU-shared placements pass. This tests distinct contexts
using the same xclbin; it does not qualify every pair of different xclbins.

## Validation

- `cargo test --all-features`: passed, including the new direction, lease rollback
  and percentile tests; hardware tests are ignored in this CPU invocation.
- Clippy with all features and with no default features: passed with warnings denied.
- Miri on execution tests with no default features: 13 passed; the deliberate
  quarantine/leak test remains ignored under Miri.
- Real GPU arithmetic → NPU DMA → GPU arithmetic: passed.
- Cross-context NPU-local and shared-buffer execution: passed.
- `doctor` on both the published runtime and ABI 1 overlay: completed all diagnostics.
- Installation from the verified Cargo package: passed (`runner,npu`). Both
  manifests and the percentile helper are included; no images/model files ship.
- Ubuntu 24.04 container with no system XRT: preparation, offline reuse, doctor,
  Loom compilation and both heterogeneous hardware tests passed. Loader traces
  show all three XRT libraries loading from the NPU cache component.
- Five fresh-process scheduler comparisons with the final runtime: median ratio
  1.022; individual ratios 1.010, 1.045, 1.022, 1.068 and 0.979. One process exceeds
  the advisory 1.05 target. Absolute timings vary substantially, so this is not a
  guarantee of a fixed overhead. The reference target remains nonbinding.


Percentiles are a convention rather than a unique indexing rule. Nearest rank
uses `ceil(n × p / 100) - 1`: p50/p95 select indices 49/94 for 100 samples, and
24/47 for 50. The prior 50-sample p95 index was already correct under that rule.

## Follow-up observations

- The percentile helper now lives under `src/`; examples include it from there.
- The host mapping fault-injection seam is identified, and the raw same-context
  convenience API points to its explicitly unsafe sharing alternative.
- GPU→NPU cache synchronization remains conservative; no unverified narrowing.
- The vision harness now uses raw, buffered newline framing with a total timeout
  and role/log diagnostics for truncated or invalid responses. Provenance hashes
  all checkpoint shards, their index and config. Unused face memmaps are removed.
- The original Lena results are retained with explicit queue-policy and partial
  CPU-accounting caveats; preprocessing documents partially padded edge patches.
  Nine CPU benchmark tests pass, including pipe framing and shard provenance.

The final GPU and NPU binaries target Ubuntu 24.04. The GPU build recipe now
uses Clang 18 in a pinned container, avoiding dependencies on the development
host’s newer glibc. Doctor also probes Loom, so compiler-library load failures
appear in diagnostics. Rust 1.88, all 64 feature combinations, Clippy and rustdoc
checks passed; the pre-existing broken Kernel documentation link is corrected.
