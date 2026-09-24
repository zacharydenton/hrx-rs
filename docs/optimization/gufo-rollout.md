# Gufo techniques in HRX

This branch implements the first measurable transfers from Gufo on gfx1151.
Candidates advance independently. A kernel win is not a model win, and a local
native bridge build is not a published runtime release.

The initial source changes are shipped. The
[expanded ecosystem plan](ecosystem-rollout.md) schedules the remaining HRX
consumers, native/compiler qualification, reusable kernel work and HRX Demos.

## Promotion contract

Keep saved baseline/candidate executables, source revision **and working-tree
identity**, compiler and all native-library hashes. Fix model revision, input,
seed, graph mode, precision, cache state, allocation policy, and timed scope.
Application comparisons use the same native bundle. Runtime comparisons use
the same application source. Record every attempt, including failed runs.

Use three alternating fresh-process pairs for complete model requests and five
for short runtime workloads. A latency candidate needs a median paired ratio
at most 0.95 and improvement in at least two thirds of pairs. Model controls
must remain within 3%; runtime controls within 5%. A memory candidate needs
lower measured peak allocation with at most 3% latency regression. These are
minimum gates; existing model numerical contracts still apply.

Use `scripts/capture-build.py --binary PATH --native DIR --output IDENTITY.json
-- BUILD_COMMAND...` from the application checkout to capture a build identity.
It hashes tracked and non-ignored source files before and after compilation and
rejects source/runtime changes during the build. Keep its output outside the
source tree or under an ignored artifacts directory.

`performance_evidence.py` implements these rules. Missing pairs, duplicate
records, invalid timings, failed numerical checks, changed build identities,
and contended devices cannot qualify. Existing fixed hashes remain useful
regressions; independent model parity determines whether numerical changes are
acceptable. Never replace a missing-model result with a quality pass.

`benchmark-consumers.py --qualification-manifest MANIFEST --primary NAME
--exclusive-device --baseline-runtime DIR --candidate-runtime DIR` adds this
policy to saved consumer campaigns. The manifest keys must exactly match all
selected workloads, including controls. Each workload provides:

- `comparison_kind`: `application` or `runtime`;
- `workload`, `model_identity`, `input_identity`, `cache_state`,
  `allocation_policy`, `timed_scope`, `hardware_identity`, `oracle_identity`;
- `baseline` and `candidate`: `source_revision`, `source_tree_sha256`,
  `binary_sha256`, `native_hashes`, `compiler_sha256`, `rustc_identity`,
  `build_environment`, `quality_report`,
  `quality_report_sha256`.

Quality reports are resolved relative to the manifest and must contain
`passed: true`, the measured `binary_sha256`, and matching `oracle_identity`.
They are outputs from the model's existing parity workflow, not substitutes
for running it. `--exclusive-device` is an operator reservation attestation;
the runner additionally checks idle state before each request. It does not
stop unrelated jobs. This collector cannot independently exclude every
external user or graphics process.

## Mapped weights

`FileView::prepare_bytes` prepares only an explicitly selected, validated
mapped range. Linux uses `MADV_POPULATE_READ` in bounded 16 MiB chunks with at
most 16 workers; unsupported kernels fall back to `MADV_WILLNEED`. Owned
storage is already resident. Real OS failures propagate and every worker
joins before return. No page preparation runs during checkpoint schema reads.

`cargo run --release --example prepare_weights -- FILE TENSOR none|advice|populate [--upload]`
measures preparation plus a complete first host read or completed device upload.
The upload destination is allocated before timing; all bytes are checked after
the completion fence, outside timing. `scripts/benchmark-prepare-weights.py`
collects five alternating process pairs in warm and DONTNEED-advised conditions.
This is a loader screening experiment, not full-model evidence. Compare warm and
cold/advised states separately and include preparation time. Only integrate
into Qwen/H3 selected-tensor materialization after upload and complete-load
measurements justify it. Never fault all tensors just because the checkpoint
was opened.

## Native graph timestamps

`Graph::finish_profiled(labels)` and `Stream::launch_profiled` expose an
explicit instrumented graph. Labels correspond to non-join payload nodes.
`Queue::prepare_profiled_batch` also supports a single prepared eager dispatch.
The optional bridge emits GPU-clock markers and queries the counter frequency
through the endpoint's DRM identity. Older bridges reject profiling while
ordinary inference remains usable.

Preparation allocates timestamp storage once. Replay waits for retirement
before reading it. Device intervals retain labels, process/device domain, process-local queue and recorder identity,
recorder-local execution number, ticks, frequency, interval union, span, and
gaps. Completion barriers serialize the diagnostic intervals; **these are not
hardware utilization measurements**. Use uninstrumented requests to qualify
performance. Counter wrap/reversed/unwritten samples fail explicitly.

The native build owns `profile.c`; rebuilding `libhrx_fabric.so` is necessary
for the optional exports. Do not republish or replace cached native bundles
with a locally relinked bridge. Validate with `test-profile-bridge.py LIBRARY`
and the ignored GPU test `profiled_graph_replay_reports_device_ticks_and_preserves_results`.

## Model and compiler boundaries

Qwen owns its padded normalization producer and explicit logical/physical
width contract. The existing direct-MLP experiment consumes those rows without
another packing launch. Padding can contain NaNs: no reduction may consume
it. The focused tests cover compact fallback, bias, tails, and changed-input
replay. The GEMM harness measures reused and rotating weight allocations on a
shared output buffer; its host dispatch timing remains diagnostic.

H3 already contains four/eight-wave F16 attention and reduced-query-residency
variants. Its focused independent F64 oracle now includes those candidates;
`examples/attention_f16.rs` compares resident inputs with identical output
bindings and alternating graph replays. No INT8/INT4 default follows from an
F16 result. Promotion requires actual encoder/decoder shapes, preparation
cost, and H3's existing parity gates before complete-request validation.

Compiler/library extraction follows evidence. Existing Loom workgroup-tree,
partial-subgroup and masked-memory conformance tests already own generic
lowering contracts. Keep model normalization/attention arithmetic local until
two consumers establish the same reusable interface. The standard library's
CONTRIBUTING.md requires a benchmarkable execution leaf and target qualification;
copying a model kernel into a new motif without those is not an implementation.

## Release sequence

1. Finish focused numerical, replay, ABI and CPU checks.
2. Preserve artifact identities and run unprofiled paired requests and controls.
3. Promote only qualified model routes; keep rejected measurements in the ledger.
4. Rebuild the pinned native bundle with the optional bridge, run existing
   consumer/runtime compatibility checks, then update release manifests.
5. Update consumer pins together only after that runtime release is qualified.

No release publication, dependency-pin bump, or compiler-policy change is
implied by the local prototype. Local results and pending gates are recorded
in the workspace rollout ledger alongside the immutable run directories.

## Local qualification on 2026-09-24

See [the measured ledger](gufo-results-20260924.json). Qwen saved 1.001 GiB of
tracked allocation with matching images and essentially unchanged median
latency; its padded DiT path passed both independent component fixtures.
H3 attention candidates were slower, and page preparation regressed warm
uploads despite helping the advised-cache case. Those policies remain
unpromoted. Full reference-image controls, baseline golden reconciliation and
native release qualification remain separate gates.

The unchanged Qwen encoder was rerun on the independent real-image
`aurora-train` fixture: relative RMS 0.245834 and cosine 0.969862 fail its
0.03/0.999 envelope. Its output hash matches the existing diagnostic. This
pre-existing numerical failure blocks full-model promotion; the padded DiT
component itself passes both independent fixtures. Keep that encoder repair
separate from this rollout rather than weakening the oracle or accepting more
performance samples as a substitute for correctness.
