# Resident vision composition

`hrx-vision` is an unpublished integration crate: model dependencies live here,
not in the HRX runtime. It currently uses sibling model checkouts and local HRX.
Applications consuming it must also patch `hrx-rs` to this checkout until the
new runtime API is released; Cargo does not inherit a dependency's patches.

`ResidentVision::new` prepares DINOv3 + SCRFD + ArcFace for one image shape.
`ResidentVision::faces` prepares just SCRFD + ArcFace. Load all models in one
`ModelContext`, with DINO/SCRFD capacity at least one and ArcFace capacity 32.

The execution path is:

1. Publish decoded RGB once into a reusable coherent input allocation. The
   first graph binds that image directly to GPU resize/letterbox, DINO
   normalization/inference/pooling (if enabled), and SCRFD inference/decode.
2. Wait for compact candidate status and score/box summaries. CPU NMS/ranking
   publishes only selected indices; it does not read landmarks or image pixels.
3. The second graph gathers selected rows and directly binds their landmarks
   and the original image into ArcFace fitting, sampling and inference. Every
   selected face is processed in model chunks of at most 32.
4. Validate compact geometry status. `analyze` returns completed resident
   descriptors, selected rows and embeddings. Use them in another GPU graph,
   or call `readback` to obtain final host records.

This is a two-phase hybrid pipeline, not GPU-only NMS, asynchronous host control,
or automatic kernel fusion. Decoding remains on CPU. Recorded graph composition
removes model-boundary copies; it does not merge all kernels into one kernel.

Warm execution uses two submissions (one without faces), no explicit device
copies, no device allocations and no native graph preparation. Host publication,
mapped control/final reads, and cache maintenance are still memory traffic.
One input slot applies backpressure to retained outputs. Face-count plans use a
four-entry idle-only cache; preparation can fail under memory-budget pressure.
Several image shapes should likewise use a bounded caller-owned cache.

The [vision benchmark](../../benchmarks/vision/README.md) uses this path by
default and retains `--pipeline host` for same-model comparisons. Its ignored
GPU test checks exact host-path parity, changing pixels, no-face inputs,
retained-output backpressure and warm resource invariants.

## Mixed-resolution batches

`BatchedVision` composes batches of up to 32 photos using shared model weights.
All models must support batch 32. `AnalysisImage` accepts original RGB plus the
caller's prepared 640-square detector and 224-square descriptor inputs. This
keeps antialiasing/letterbox policy explicit and preserves an application's
existing descriptor space; it does not claim GPU downscaling for this entry point.

Residency is a placement choice, not a universal speedup. Publishing a large
CPU-decoded original just to sample a few small faces can cost more than CPU
crop preparation. Applications should qualify source-size and face-density
workloads, retain a CPU-prepared path where it wins, and benchmark decode and
publication as well as model time. Already-resident sources have a different
cost model and should use the resident tensor entry points directly.

The first graph runs batched DINO and SCRFD. After compact CPU NMS, only images
with selected faces are published into a packed coherent source buffer. Runtime
offset/width/height records let one sampler handle different resolutions without
resolution-specific compilation. Fitting, crops and ArcFace remain on-device.
ArcFace chunks share graph-local activation scratch. A complete call takes two
submissions, or one when no faces survive. Every selected face is returned.

Source storage is lazy, power-of-two bucketed and bounded by the byte budget
passed to the constructor. Source axes are limited to 32767 pixels. Two batch
shapes and eight face/source-size specializations per shape are retained, with
idle-only eviction. Face capacity is rounded to powers of two up to 32, then
multiples of eight; padded rows never escape in the returned result. Retained
descriptors/outputs apply backpressure rather than allowing storage overwrite.
The source-byte ceiling is per prepared face plan, not a global process budget;
use the shared context's memory budget to bound all resident plans and weights.

`BatchOutput` exposes resident descriptors, selected rows, embeddings,
per-image face offsets and original RGB views for photos with faces. The image
views retain their source lease and can feed a subsequent GPU crop, swap or
enhancement graph without republishing pixels. `readback` is the explicit final
host-record boundary.

Hardware parity/resource test, with an optional alternating timing comparison:

```sh
HRX_OFFLINE=1 VISION_BATCH=32 VISION_SAMPLES=20 \
  cargo test --release --manifest-path pipelines/vision/Cargo.toml \
  --test batch -- --include-ignored --nocapture
```

The timed comparison excludes decode/downscaling and uses identical prepared
model pixels. Its host baseline runs CPU affine crops and batched neural models;
it is not an application-wide throughput result.
