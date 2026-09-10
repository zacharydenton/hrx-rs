# Strix Halo: standalone Lena throughput run

On 2026-09-10, the shared GPU/NPU DINO queue measured **1.025×** the faster
GPU-only mode's median throughput (**+2.5%**) at GPU batch eight.
These are exploratory measurements on one repeated image with existing model
runtimes, not a measurement of `hrx` Rust execution or general vision accuracy.
The GPU-only and mixed round ranges overlap, so this does not establish a reliable
speedup. Putting all DINO work on this NPU artifact was much slower.

| Pipeline | Median images/s | Range across four rounds | Median accounted CPU cores¹ |
| --- | ---: | ---: | ---: |
| GPU only, serial models | 88.79 | 85.92–91.43 | 3.29 |
| GPU only, parallel models | 92.48 | 92.26–92.66 | 4.33 |
| GPU faces + NPU DINO only | 8.39 | 8.36–8.40 | 0.10 |
| GPU faces + DINO shared between GPU/NPU | 94.82 | 90.64–97.80 | 4.28 |

¹ Parent CPU plus worker request-service CPU divided by elapsed time. This excludes
worker activity outside request service and is not whole-system CPU usage.

Each mode processed 256 images per round, with four rotated mode orders:
**4,096 completed images**, each with one detected face, landmarks, CLS embedding
and masked-mean embedding. Every result passed its backend's consistency checks.
The balanced mode assigned 21–23 of each round's 256 DINO
embeddings to the NPU. It actually executes both accelerators; frames are neither
skipped nor answered from a result cache.

## Input and accuracy

The sole input is OpenCV 4.10.0's 512×512
[Lena sample](https://github.com/opencv/opencv/blob/4.10.0/samples/data/lena.jpg),
with source URL and SHA-256 in [sample.json](sample.json). Download it using
`python3 scripts/vision-bench/fetch_sample.py`. The JPEG is local and Git-ignored.
No deleted reference photos are needed or restored.

A fresh eager float32 CPU run of the matching HF checkpoint supplies the reference.
Both CLS and masked-mean cosine must reach 0.997 against it, and against the other
accelerator. Measured cosine values, in CLS / masked-mean order:

- GPU versus CPU float32: 0.999920 / 0.999923.
- NPU versus CPU float32: 0.999463 / 0.999517.
- GPU versus NPU: 0.999455 / 0.999554.

These validate numerical consistency for Lena. Repeating one image cannot validate
diverse-input accuracy, detector quality, or production throughput. CPU scheduling
tests use distinct image IDs and values to check slot reuse and out-of-order results.

## Setup and timing

- AMD Ryzen AI Max+ 395, Radeon 8060S, XDNA2; Linux 7.2.3-arch1-3, GPU power policy `auto`.
- SCRFD-10GF from `scrfd-loom`, 640×640, both GPU configurations and mixed configurations.
- Matching DINOv3 ViT-L/16 checkpoint, 224×224: PyTorch 2.13.0
  BF16 SDPA with `torch.compile(max-autotune)` on GPU; VitisAI EPContext on NPU,
  ONNX Runtime 1.27.0, CPU EP fallback disabled.
- GPU batch 8, NPU batch 1, 32 admitted image slots, 8 preprocessing threads,
  2 intra-op CPU threads per model worker. The CPU reference worker exits before timing.
- JPEG decoding, preprocessing, IPC, device copies, both models, face decoding/NMS,
  pooling and result collection are timed. Model loading/compilation, file reads,
  warmup/reference generation, correctness checks and report writing are excluded.

The harness owns DINO preprocessing, runtime-library setup and reference generation.
It needs no `xdna-vision` checkout, imports, manifests or cached reference data. It
accepts a standalone compiled ONNX file plus the matching HF checkpoint and vendor
runtime. The NPU artifact in this run was copied unchanged from an existing local
compiled model; it was not recompiled by `hrx`. Its SHA-256 is recorded below.

| Model artifact | SHA-256 |
| --- | --- |
| GPU HF model.safetensors | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |
| NPU EPContext ONNX | `00f5a79a15be722ea12173a9e4fe10fc8735740938528a4437ad5293300ee494` |

The four rounds share resident models in one harness process; they are not four
independent system boots or a statistical guarantee. GPU-only rounds last about
three seconds each. This run does not establish the best possible GPU batch size,
NPU compiler, power setting, energy efficiency, or heterogeneous scheduling policy.
In particular, the mixed queue can incur partially filled GPU batches and a slow
NPU tail. The measured configurations also differ in embedding queue depth:
`mixed_balanced` admits one batch per backend, while `gpu_parallel` can queue up
to four GPU batches at depth 32. This confounds device placement with queue policy.
The result should be read as an observed gain for this configuration.

[results-lena.json](results-lena.json) contains all 16 measurement rows, quality
checks, arguments, source and binary hashes, device/runtime metadata, and the
summary. [README.md](README.md) gives the standalone reproduction command. Detailed
per-image outputs and logs remain under `artifacts/vision-bench-standalone-lena`.
