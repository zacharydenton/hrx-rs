# Single-image vision latency

Runs a JPEG or PNG through DINOv3 ViT-S+/16 descriptors, SCRFD detection, and
ArcFace embeddings for every detected face. All three models share one HRX
context. This is a serial, batch-one image-latency benchmark, not a bulk-throughput
or cross-model concurrency benchmark.

The development harness uses sibling checkouts of `dinov3-hrx`, `scrfd-hrx`, and
`arcface-hrx`, plus this checkout of HRX. It is a separate, unpublished crate so
model dependencies do not enter the runtime crate's dependency graph.

From the HRX repository:

```sh
cargo run --release --manifest-path benchmarks/vision/Cargo.toml -- \
  /path/to/face.jpg --samples 30 --warmup 5

HRX_OFFLINE=1 cargo run --release --locked \
  --manifest-path benchmarks/vision/Cargo.toml -- \
  /path/to/face.jpg --offline --samples 30 --warmup 5 --json > /tmp/vision.json
```

Weights resolve at each client's pinned revision; missing weights are fetched
unless `--offline` is set. `HRX_OFFLINE=1` separately forbids native artifact
downloads. Model resolution, loading/compilation, first execution, and warmup
are excluded from warm timing and reported separately. Keep other GPU work idle.

CPU migration can add substantial decoding-time noise. For a controlled Linux
comparison, run both built executables under the same `taskset -c CORE` affinity,
using an allowed core. This also pins their worker threads; do not mix pinned and
unpinned results. JSON records the main thread's allowed CPU list when available.

The total timer includes decoding cached compressed bytes on every iteration,
preprocessing, model execution, host/device transfers, face selection, alignment,
and waits until all descriptors and embeddings are available on the CPU. File
I/O and output validation/reporting are outside it. Stage boundaries are:

- `decode`: JPEG/PNG to RGB, without automatic EXIF rotation.
- `dino_preprocess`: centered 224-square bilinear letterbox and patch-center mask.
- `dino`: RGB normalization, transformer, masked pooling and descriptor readback.
- `scrfd`: original RGB to 640-square detector input, CNN, decode/NMS and landmarks.
- `arcface`: alignment from those landmarks, CNN and embedding readback; faces are
  processed in chunks of up to 32 with none dropped.
- `total`: the complete serial path above, measured directly rather than by
  adding stage medians.

No-face images fail by default, so a successful run exercises all three models.
`--allow-no-faces` explicitly permits skipping ArcFace and labels that in both
human and JSON output. All warmup and timed outputs must exactly match the first
run; this checks replay consistency, not accuracy against an independent model.

JSON includes median/p95, every timing sample, first-run and setup times, image
SHA-256, checkpoint paths/revisions, runtime allocation/submission/copy counters,
boxes/landmarks, DINO descriptors and face embeddings. Zero copied bytes means
no explicit device-copy operations, not zero CPU memory traffic. Retain the same
image, checkpoint pins, preprocessing and source revisions when comparing builds.

```sh
cargo test --manifest-path benchmarks/vision/Cargo.toml
cargo clippy --manifest-path benchmarks/vision/Cargo.toml --all-targets -- -D warnings
```
