# SCRFD + DINOv3 bulk image throughput

`bench.py` measures completed images per second for a real image-indexing pipeline:
SCRFD-10GF face detection at 640×640, plus DINOv3 ViT-L/16 CLS and masked-mean
embeddings at 224×224. Every image must produce both results. It uses existing
model runtimes to test device placement; it does **not** measure the Rust `hrx`
scheduler, shared device buffers, or zero-copy execution.

The GPU face detector comes from `scrfd-loom`. DINO uses the **same ViT-L/16
checkpoint** through Hugging Face/PyTorch on ROCm or a precompiled VitisAI NPU
artifact supplied directly as an ONNX file. The existing `dinov3-loom`
implementation is ViT-S+;
using it as the GPU baseline against ViT-L on the NPU would compare different
models. No model downloads, builds, package installations, or system tuning are
performed by this harness.

## Comparisons

| Mode | Face detection | DINO embeddings | Scheduling |
| --- | --- | --- | --- |
| `gpu_serial` | GPU | GPU | Face then DINO, one microbatch at a time |
| `gpu_parallel` | GPU | GPU | Independent worker processes with bounded queued batches |
| `mixed_parallel` | GPU | NPU | Independent branches, all embeddings on the NPU |
| `mixed_balanced` | GPU | GPU and NPU | GPU batches and NPU single images take work as their previous batch completes |

The NPU artifact always executes batch one. `--gpu-batch` controls both GPU
models. The GPU DINO graph has a fixed shape, so partial batches are padded;
padded images never count toward throughput. SCRFD accepts the actual batch
length. In `mixed_balanced`, a small ready queue can feed the NPU while it fills
a GPU batch; the policy does not estimate remaining runtime or optimize the final
tail. Per-round reports include the number of embeddings each device completed.

All timed models stay resident throughout a run, including the idle model in modes
that do not use it. Separate processes accommodate incompatible NumPy/Python
versions and isolate the existing runtimes' default HIP streams. CPU memory-mapped
arrays carry inputs and embeddings between processes; these are ordinary CPU IPC
buffers, **not** GPU/NPU shared allocations.

## Timing and correctness

The wall clock starts before admitting the first image and ends after copying out
both results for the last image. It includes JPEG/PNG decoding from cached
compressed bytes, RGB/BGR conversion, both resize/normalization paths, bounded
preprocessing queues, IPC, device uploads/downloads, model execution, face decoding
and NMS, landmark delivery, DINO pooling, and queue drain. It excludes disk reads,
model loading/compilation, reference warmup, output validation, and report writing.
Parallel preprocessing uses the same worker count and slot limit in every mode.

Latency is admission-to-both-results latency within this finite bulk workload;
it is **not** live-camera latency or an unbounded arrival-rate test. CPU core
equivalents combine parent CPU time and workers' service CPU time, excluding idle
worker CPU usage outside requests. This is partial accounting, not whole-system CPU usage. Service milliseconds are batch wall time divided
by actual image count, not isolated kernel timings; preprocessing service times
include CPU contention among preprocessing threads. No energy measurements are made.

Before timing, every input image passes through both DINO backends and a separate
float32 CPU instance of the matching HF checkpoint using eager attention. CLS and
masked-mean cosine must each reach `--min-cosine` (default 0.997), both between
accelerators and against the CPU reference, with finite outputs. The CPU reference
worker is closed before timing; no stored reference-image package is needed.
SCRFD's untimed outputs become the per-image reference for the workload; this is consistency checking, not a detection-accuracy
assessment against human annotations.

Every timed output is checked after the timer: unique and complete image IDs,
branch association, embeddings against that backend's warm reference, and full
face boxes/landmarks. Slots are released only after both results are collected.
An error aborts the run; no final success summary is written. Earlier completed
rounds may remain on disk and are not evidence of a successful full run.

GPU DINO uses BF16 SDPA: FP16 produced NaNs with this checkpoint on this machine.
`--gpu-mode graph` captures a HIP graph; `compile` uses `torch.compile` with
`max-autotune`, including compilation in setup rather than timing. Neither mode
silently falls back to a different backend. The NPU session disables ONNX Runtime
CPU EP fallback and requires an open accelerator device. VitisAI's compiled
custom operator still has host-side work; disabling fallback does not imply every
instruction runs on the NPU. ORT may list `CPUExecutionProvider` even when fallback
is disabled.

## Independence

The harness owns its DINO preprocessing, pooling, runtime-library discovery and
float32 reference generation. It does not import `xdna-vision`, search its cache,
read its manifests, or need its checkout. Model compilation remains outside this
benchmark: provide a trusted compiled ONNX artifact, a matching HF checkpoint, the
vendor runtime, and a built `scrfd-loom` backend.

## Input image

The current run uses the standard 512×512 Lena JPEG from the
[OpenCV 4.10.0 samples](https://github.com/opencv/opencv/blob/4.10.0/samples/data/lena.jpg).
[sample.json](sample.json) pins its download URL and SHA-256. Fetch it explicitly:

```bash
python3 scripts/vision-bench/fetch_sample.py
```

The image stays under Git-ignored `artifacts/vision-bench-lena`; only the source
manifest and fetch script belong in the repository. `bench.py` verifies the image
against the directory's `SOURCES.json` when present and records that provenance.

This is a **single-image throughput workload**: the image is decoded, processed,
and inferred again on every iteration. Results are not cached. Repeating one image
measures execution cost; it does not establish diverse-input accuracy or production
throughput. Reports explicitly record `single_image_repeated` and one unique image.
The earlier local DINO reference photos and benchmark copies were deleted; this
sample is independent of them. Its float32 reference is computed afresh from the
matching checkpoint before each run.

## Run

Prerequisites are local, trusted artifacts and dependencies:

- Built `scrfd-loom` checkout with `scrfd_loom.py`, `build/libscrfd.so`, weights and kernels.
- Standalone, trusted DINOv3 ViT-L/16 VitisAI EPContext ONNX artifact with input
  `[1,3,224,224]` RGB float32 normalized as documented above and output
  `[1,201,1024]` hidden states (CLS, four register tokens, 196 patch tokens).
  It must use the matching HF checkpoint's weights. Any external ONNX data files
  must be supplied alongside it; the measured artifact is self-contained.
- Local HF `facebook/dinov3-vitl16-pretrain-lvd1689m` checkpoint.
- Main Python with GPU-enabled PyTorch, Transformers, NumPy, Pillow, OpenCV.
- Separate Ryzen AI Python with vendor ONNX Runtime, compatible NumPy 1.x,
  working XRT and VitisAI libraries. The benchmark uses only the selected Python's
  standard library to locate its installation paths. `--xrt-root` defaults to
  `/usr`; repeat `--npu-lib-dir` for nonstandard library directories. The child's
  `XILINX_XRT` and `LD_LIBRARY_PATH` are recorded. On Arch, a required ncurses
  loader alias is created inside the run's output directory.

```bash
python3 scripts/vision-bench/bench.py \
  --scrfd "$HOME/code/scrfd-loom" \
  --gpu-model /path/to/local/hf-dinov3-vitl16-snapshot \
  --npu-model /path/to/standalone/dinov3-vitl16.onnx \
  --npu-python "$HOME/tools/ryzen_ai-1.8.0/venv/bin/python" \
  --images artifacts/vision-bench-lena \
  --gpu-batch 8 --gpu-mode compile --depth 32 --preprocess-workers 8 \
  --rounds 4 --repeats 256 \
  --output artifacts/vision-bench-run
```

Use a new output directory for every run. `--repeats` cycles the corpus per round;
each mode processes exactly `corpus size × repeats` images. Mode order rotates
between rounds. Test batching rather than assuming batch one represents GPU bulk
throughput. `--modes gpu_serial gpu_parallel mixed_balanced` can shorten exploratory
sweeps; a speedup is only reported if that run includes a GPU-only mode.

`config.json` records arguments, checkpoint/artifact/corpus hashes, source revisions
and working-tree status, backend versions, setup times, kernel/firmware and power
policy. `quality.json`, `measurements.jsonl`, per-image round files, worker logs,
and `summary.json` preserve the results. Summary ratios divide the mixed median
by the **faster** GPU-only median. There is no speedup pass threshold.

CPU-only tests cover preprocessing geometry, normalization and pooling, plus
reordered preprocessing, partial batches, small queues, slot reuse, and backend
failure propagation:

```bash
python3 -m unittest discover -s scripts/vision-bench -v
```

See [the measured Strix Halo run](RESULTS.md) for results and limitations.

Checkpoint provenance supports both `model.safetensors` and the sharded
`model.safetensors.index.json` format, hashing the config, index and every shard.
The measured modes differ in embedding queue limits: `mixed_balanced` allows one
batch per backend, while `gpu_parallel` can queue batches up to the admitted image
depth. The published results preserve and disclose that scheduling difference.
