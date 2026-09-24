# HRX ecosystem optimization rollout

Plan expanded on 2026-09-24 after shipping the first Gufo transfer to
`hrx-rs` (`1d96693`), `qwen-image-hrx` (`51daab0`) and `h3-hrx` (`9975e8b`).
Those commits are on their remote default branches. This document schedules
the next work; it does not claim the candidates below are implemented or faster.

The [promotion contract](gufo-rollout.md) and
[first measured results](gufo-results-20260924.json) remain authoritative.
Qwen's existing encoder parity failure still blocks its default promotion.
The rejected H3 schedules and warm-upload regression are evidence against
copying a technique indiscriminately into other models.

## Order and ownership

Each implementation has one owning repository and a separate commit. Run GPU
qualification serially on the reserved gfx1151 machine. CPU development may
continue independently. First collect a profile, then select one mechanism;
the listed candidates are hypotheses, not a requirement to implement all of them.

| Order | Package | First deliverable | Dependency and completion gate |
| --- | --- | --- | --- |
| 0 | Evidence adapters | Extend existing campaign inputs for batch/shape controls and resident pipelines | Saved builds, independent quality receipts and complete paired records before any promotion |
| 1 | HRXDB | Profile scan versus selection at batch 3, with batch 1 control; test the dominant cost | Exact stored-representation ranking contract and completed-search gain |
| 2 | DINOv3 | Profile ViT-S+ descriptors at batch 4, with batch 1 control; inventory live intermediate bytes | Independent encoder/model checks and full descriptor latency/memory gate |
| 3 | ArcFace + SCRFD | Separate CNN, alignment/resize and selection costs, then optimize one measured stage | Each model's existing reference and fixture gates before integration |
| 4 | Faceswap | Measure the resident frame pipeline using qualified face components | Reconcile current local work and pinned component revisions; preserve frame order and pixels |
| 5 | Krea2 | Attribute launch-to-ready and first/warm requests; screen selected mapped ranges or scratch lifetimes | Existing unquantized parity gate, including native VAE output |
| 6 | HRX native/compiler | Qualify optional timestamps in a reproducible bundle; investigate compiler changes from captured hot kernels | ABI/replay checks, consumer controls, then a separate runtime release |
| 7 | Loom kernel library | Qualify a reusable representation/operation from demonstrated consumer needs | Complete motif/kernel/test/benchmark closure; no model-specific kernel dumping |
| 8 | HRX Demos | Establish a gfx1151 baseline for existing Ideogram 4 stage/session benchmarks | Its existing BF16/FP8 fixtures and HAL contracts; target-specific results |

Packages 1-5 can use the currently published native bundle. They do not need
to wait for package 6: use existing host/stage measurements first, and use a
separately identified diagnostic bridge only for attribution. Do not combine
a compiler update and an application optimization in one comparison.

## 0. Reuse the measurement infrastructure

`scripts/benchmark-consumers.py` already covers HRXDB, DINOv3, ArcFace, SCRFD,
Krea generation and H3 audio. Extend that collector's workload descriptions
and timing adapters instead of creating a separate framework per consumer.
Add Faceswap's existing resident benchmark as an adapter. Keep native C/HAL
and Loom benchmark formats intact; attach their reports to the shared evidence
contract rather than replacing their runners.

The current fixed workloads do not provide all batch/shape controls below.
Implement explicit workload IDs, argument vectors, timed scope and output
validation for each new case before calling it qualified. Every quality receipt
must identify the measured binary and oracle; timing-script agreement alone
is not a numerical reference. Include a failed/partial run in the report even
when later attempts succeed.

For each work package:

1. Snapshot clean source revision, tree, dependency lock, model/fixture hashes,
   native bundle and compiler. Use `scripts/capture-build.py` for saved Rust
   builds. Save both executables before changing the candidate.
2. Record one primary workload and one relevant control in advance. Distinguish
   startup, first request, warm host completion and device dispatch intervals.
   Retain raw samples, peak tracked allocation, process memory, transfer bytes,
   launch count and compile resource reports where available.
3. Run the smallest existing independent numerical check, then replay with new
   inputs and awkward tails. Broaden only to the affected public contracts.
4. Run three alternating fresh-process pairs for full model requests, or five
   for short runtime/kernel screens. Warm each shape consistently; retain an
   unchanged-binary control if noise obscures the result. Use production builds.
5. Apply the shared gates: primary median latency ratio at most 0.95 with at
   least two thirds of pairs improving; model controls at most 1.03 and runtime
   controls at most 1.05. Memory candidates must reduce measured peak bytes
   with at most 3% median latency regression. Stricter model quality gates win.
6. Land successful paths as defaults after complete qualification. Remove new
   rejected experiment routes, retaining their measurements. Do not add tuning
   switches merely to preserve a losing variant.

Profile runs never supply headline speed measurements. Reused-weight and
rotating-weight screens should use the same output allocation; rotation must
exceed the measured cache working set. Neither rotation nor DONTNEED advice
proves cold storage. Keep warm and advised-cache loader results separate and
include preparation in launch-to-ready/first-request time.

## 1. HRXDB: partial batches and exact selection

Inspected base: `bc3cb0a`. The repository already owns resident corpora,
shared-read batches, tiled score storage and running GPU top-k. Relevant code:
`src/batch.rs`, `src/selection.rs`, `src/bin/bench.rs`, and `kernels/`.
Selection already reduces local candidate lists; do not propose replacing a
nonexistent full-corpus sort. The `k > 32` path sorts local lists and merges them.

First measure batch 3 over 100,000 x 384 rows, k=10, alongside batch 1.
Attribute scan, local selection, running merge and readback separately.
Batch workspace currently rounds widths up to at least eight: investigate
avoiding inactive-row work or reducing its scratch only if the profile supports
it. If selection dominates, screen an exact partial-selection improvement for
the existing local lists, retaining FP32 comparisons and deterministic ID ties.

Use `hrxdb-bench --rows 100000 --dimensions 384 --batch 3 --k 10 --samples 30`
on each saved executable. After a local win, add 3/4/8/33/60/64 query boundaries,
k=1/10/32/33/1024, row/tile/shard tails and a corpus larger than cache.
Reserve the multi-million-row campaigns for the candidate that survives screening.

Correctness entry points include `all_schedules_match_cpu`,
`batch_matches_individual_queries`, `small_empty_invalid_and_ties`,
`exclusions_compose_with_large_k_and_reset_between_queries`, and
`topk_merges_large_matrix_tiles_and_global_ids`. Preserve exclusions, empty rows,
device query strides and stream ordering. Measure completed search separately
from ingestion. A changed tie, dropped ID or unstable replay rejects the candidate.

## 2. DINOv3: remaining layout costs and live scratch

Inspected base: `6af6531`. Start in `src/encoder.rs`, `src/pooling.rs`,
`src/weights.rs` and `examples/bench_descriptors.rs`. Graph reuse, private scratch
reuse and fused SwiGLU already exist. First inventory materialized buffers and
layout conversions around those operations; remove a buffer/pass only after
establishing its last reader and the consumer's logical/physical layout.

Use ViT-S+ batch 4 at 224x224 as primary and batch 1 as control. The existing
descriptor runner accepts `rgb 4 100`; retain the same selected variant in both
arms. If a normalization-to-consumer layout change is useful, preserve its
reduction tree, every F16 boundary, prefix/register tokens and poisoned padding.
For scratch reuse, test changing batches and caller-owned graph composition.

Start with `downstream_encoder_specs_match_reference_and_compose_in_caller_graphs`
and the affected `src/kernel_tests.rs` check. Then use
`full_reference_and_changing_batch_vits` and descriptor replay tests.
The existing all-token and CLS cosine gate is greater than 0.9999.
Before a shared-family default, qualify each affected family, including F32
residual outliers and 128-channel heads. Begin with a small affected case;
ViT-7B is a final qualification requirement only if the change reaches it.

## 3. ArcFace and SCRFD: preserve the existing resident path

Inspected bases: ArcFace `d676fd6`, SCRFD `98ee1bb`. ArcFace already has coherent
terminal I/O, graph replay and activation liveness; SCRFD already combines
resize, inference and decode and writes crops into reusable batch storage.
Use `src/cnn.rs` and `src/plan.rs` in each repository, plus ArcFace
`src/alignment.rs` and SCRFD `src/detection.rs` / `src/postprocess.rs`.

Measure batch 1 and a ragged batch 6 before selecting a change. Screen bounded
convolution window reuse or pack/epilogue fusion only in the dominant CNN layer,
preserving channel/tap accumulation and intermediate rounding. If SCRFD decode
dominates, optimize that stage while preserving thresholding and stable ordering.
Inspect VGPR/LDS/spills; fewer launches or higher occupancy alone is not a win.

Use each repository's `native_reference_and_replay` and `insightface_fixture`.
ArcFace also has `convolution_tiles_preserve_embeddings_across_batch_boundaries`
and independent alignment crop tests. SCRFD has
`gpu_decode_matches_stable_cpu_oracle_and_errors`,
`parallel_decode_preserves_block_level_and_tail_order` and
`detect_batch_uses_shared_decoder_with_cached_options`.
Keep fixture tolerances, landmark coordinates, detector ordering and embedding
normalization unchanged. Extend SCRFD's existing `bench_detect_batch` and
`bench_resident` examples; keep ArcFace's CLI benchmark scope explicit.

## 4. Faceswap: qualify the complete resident chain

Inspected base: `8b16009`. Read `src/pipeline.rs`, `src/analysis.rs`,
`src/runtime/memory.rs`, `src/runtime/winograd.rs` and `src/video/stream.rs`.
The resident path already avoids intermediate host transfers and reuses graphs.
Start from `examples/bench_resident_pipeline.rs` and `bench_frame_batch.rs`.
Do not reintroduce transfers to make individual stages easier to time.

Compare one 1080p frame with a fixed face fixture at batch 1 and batch 4.
Measure crop, analysis, INSwapper, optional GPEN and composition; select one
remaining scratch lifetime, window reuse or batching opportunity. Report frame
completion latency alongside throughput and peak bytes. Keep codec I/O separate
from GPU pipeline timing, then validate one short complete video.

Use `resident_pipeline_six_configurations`, covering boost 128/256/512 and
enhancement on/off, followed by the existing short `tests/video_pipeline.rs`
qualification when batching/order changes. Include a partial final batch,
flush, cancellation and bounded buffering. Preserve pixels against the existing
serial/reference pipeline. Faceswap pins ArcFace/SCRFD git revisions: update
those only to separately qualified commits and retest the integration.

## 5. Krea2: active weight ranges and residual scratch

Inspected base: `4dbe5b6`. `src/session/mod.rs` already describes a ten-launch
block with fused preparation, QKV/gate, SwiGLU epilogues and resident residuals.
Do not repeat that fusion work. Inspect `src/session/weights.rs`,
`src/checkpoint/plan.rs`, `src/ops/tensor.rs` and `src/pipeline/profile.rs`.

Separate checkpoint mapping, conversion/upload, compilation, first request and
warm replay. Screen bounded preparation only for ranges actually consumed by
the active component; compare with no preparation under warm and advised-cache
conditions. An owned/converting load may not benefit. The first rollout's
warm-upload regression makes default prefaulting an unproven hypothesis.
Alternatively, use a liveness profile to remove a measured scratch overlap.

Start with `examples/bench_runtime.rs` (256x256, two steps) for integration
screening, then the affected production shape and a second shape as control.
Use focused arithmetic/quantized tests, `qualify_pipeline` saved-baseline
comparison, and `scripts/parity.sh`. The official unquantized reference gate
allows at most 0.1 dB regression in latent error and image PSNR from the accepted
baseline. Saved conditioning does not validate the text encoder; changes there
need its own independent fixture. Never accept new goldens to pass an optimization.

## 6. HRX native and compiler qualification

Inspected `hrx-system` HEAD: `556c648e8`, with substantial local edits.
Keep compiler/runtime work in an isolated checkout. The optional profiling
implementation lives in HRX Rust's `native/amdf/profile.c` and Rust graph API;
reproduce it through the supported full native build before publishing a bundle.
Test an older bridge's graceful profiling rejection and ordinary execution,
new marker/clock exports, timestamp retirement/reset behavior and changed-input
graph replay. Preserve the existing native ABI for uninstrumented consumers.

Only then run runtime comparisons on unchanged application sources for HRXDB,
DINOv3, ArcFace, SCRFD, Krea2, Qwen and H3 using saved binaries. Add Faceswap
integration before changing its pin. Native qualification, version bump,
bundle publication and coordinated consumer pins are a separate release task;
pushing the initial source commits has not published that runtime capability.

Compiler candidates must come from captured consumer kernels. Start with one
register-pressure, scheduling, masked-tail or repack problem and its independent
oracle. Retain generic reduction/memory contracts in Loom and model arithmetic
in the model. Compare compile reports and unprofiled requests, including an
unchanged-kernel control. No global wave-mode or scheduling-policy change based
on one favorable microbenchmark.

## 7. HRX Loom Kernels: reusable contracts after evidence

Inspected base: `e1af7cd`. Existing format motifs include Q4_K, Q5_K, Q6_K and
Q8_1 x4; a Q8_1 x4 quantization kernel is already benchmarkable. Begin by
qualifying that existing leaf on gfx1151 with decode, ragged and larger shapes.
An exact packed-integer transform or vector-store improvement is a candidate
only if it preserves all signedness, scales, byte/lane order and tail contracts.

Follow the repository's `CONTRIBUTING.md`: canonical source, lint, public cases,
benchmark plans, target/execution profiles and compile-report comparison.
Use the existing benchmark entry point:

```sh
python dev.py benchmark --config=amdgpu --device=amdgpu://0 \
  --output-dir=PATH --target=//kernel/ggml/quantize:q8_1_x4_f32 -- \
  --measure=dispatch_complete --batch-size=64 --profile-final-batch=true
```

Extract new model-derived motifs only after two consumers establish the same
interface and numerical contract. Production motifs have no launch ABI;
kernels own dispatch. The current repository forbids source-bearing `model/`
and `target/` packages without their admission rules and forbids experimental
payloads on the release line. Do not use extraction to preserve rejected H3
schedules or duplicate model-specific normalization.

## 8. HRX Demos: separate target baseline

Inspected base: `b893ecd`. Its Ideogram 4 case study measures gfx1100, so those
numbers are not a gfx1151 baseline. First build and run its existing stage and
session benchmark/fixture paths on gfx1151 using a supported target profile.
If that target is unavailable, record the support gap before optimizing.

Inspect `pipeline/parameter_materialization.c`, `parameter_residency.c`,
`parameter_slab.c`, `stages/*_benchmark.cc` and
`tooling/dispatch_profile_compare.py`. Screen active-range preparation,
materialization reuse or a measured transient-slab lifetime while preserving
compact FP8 weights, scale semantics and BF16 fixture behavior. Use HAL timing
for attribution and completed C API calls for claims. Keep coarse reusable
command buffers, explicit barriers/semaphores and the existing memory-plan tests.
Any pinned HRX submodule update stays separate from application changes.

## Checkouts and progress records

At inventory time ArcFace had a modified `src/cnn.rs`, Faceswap had changes in
`README.md` and `src/analysis.rs`, and `hrx-system` had ongoing compiler/runtime
work. Do not commit, discard or benchmark those edits as this plan's baseline.
Create isolated worktrees and reconcile relevant changes with their owners
before integrating. Recheck state and revisions at the start of every package.

The `llama.cpp-hrx` / `llama.cpp-hrx2` checkouts and DINO NPU/XDNA experiments
are a deferred lane. They have local work and different integration/toolchain
contracts; establish the active backend and baseline before assigning kernel
work. Duplicate worktrees and compiler experiments are not independent products.

Each package ends with a repository-local retained/rejected result and an entry
linked from this plan: exact revision/build identities, workload/control, raw
pairs, quality result, decision and remaining blockers. Keep large artifacts
outside Git with stable hashes. Update status as evidence arrives:

`planned -> baseline captured -> screened -> independently qualified -> promoted`

A failed numerical gate goes to `blocked`; a slower candidate goes to `rejected`.
Neither status becomes a pass because additional timings were collected.
