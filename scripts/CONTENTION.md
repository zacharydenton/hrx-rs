# GPU/NPU memory contention on Strix Halo

Does running the XDNA2 NPU alongside the gfx1151 iGPU actually add throughput, or do they
just divide one LPDDR5X memory controller between them? This is the go/no-go probe for
heterogeneous GPU+NPU model execution — it runs before any runtime integration work,
because if the devices are zero-sum there is nothing to integrate.

## Running it

```bash
cargo build --release --example gpu_load
(cd ~/code/dinov3-xdna2 && cargo build --release --bin npu_load)
python3 scripts/contention.py --seconds 8 --repeat 3
```

- `examples/gpu_load.rs` — sustained device fill/copy through hrx-rs, reports GB/s.
- `~/code/dinov3-xdna2` `npu_load` (branch `npu-contention-load`) — config3 bfp16 GEMM
  dispatched in a loop with a resident packed A, reports dispatch/s, GB/s and GFLOP/s.
- `scripts/contention.py` — each load alone, then both concurrently.

The headline metric is **concurrency efficiency**: the sum of each device's retained
throughput fraction. 1.00 means purely zero-sum; 2.00 means fully independent.

## Results

Ryzen AI MAX+ 395, gfx1151 + XDNA2, 3 trials × 8s. GPU load is a 256 MiB fill
(~162 GB/s alone); NPU load is a bfp16 GEMM m=16384 k=4096 n=1024.

| metric | alone | concurrent | retained |
|---|---:|---:|---:|
| GPU bandwidth (GB/s) | 161.9 | 136.1 | 84.0% |
| NPU dispatch (per s) | 146.6 | 104.7 | 71.4% |
| NPU bandwidth (GB/s) | 16.7 | 11.9 | |

**Concurrency efficiency: 1.55.**

A read+write copy load instead of a write-only fill shifts *who* pays but lands in the same
place overall — the GPU takes the larger hit and the NPU keeps more of its rate:

| GPU load (2 trials × 8s) | GPU retained | NPU retained | efficiency |
|---|---:|---:|---:|
| 256 MiB fill (161.9 GB/s alone) | 84.0% | 71.4% | 1.55 |
| 256 MiB copy (116.0 GB/s alone) | 66.6% | 85.1% | 1.52 |

The interference is genuinely bandwidth, not driver or host overhead. Re-running with a
cache-resident 4 MiB GPU buffer — same dispatch rate, same driver path, but little
controller traffic — leaves the NPU almost untouched:

| GPU load | NPU retained |
|---|---:|
| 256 MiB fill (~162 GB/s) | 71–77% |
| 4 MiB fill (cache-resident) | 96% |

## What this means

The devices contend, but nowhere near zero-sum. Under a GPU load heavy enough to pull
162 GB/s, the NPU still delivers ~71% of its solo rate — about **15 TFLOP/s of bfp16 GEMM
on top of 84% of the GPU's bandwidth**.

That asymmetry is the whole opportunity. This NPU workload is compute-dense: 137 GFLOP per
dispatch against only 114 MB of traffic, roughly 1200 FLOP/byte. It buys a large amount of
compute for a small slice of the contended resource, which is exactly the trade that makes
heterogeneous execution worth it for compute-bound work — prefill, speculative-decode draft
models, vision encoders.

The converse also holds: a *bandwidth*-bound NPU workload would come straight off the GPU's
budget with no compute to show for it. Decode-style offload is not the target.

## Follow-up: the mechanism is GPU cache residency

A later sweep ([CACHE-RESIDENCY.md](CACHE-RESIDENCY.md)) explains these numbers. Strix
Halo's 32 MB MALL gives the GPU ~3.8× DRAM bandwidth, with a cliff exactly at 32 MB, while
the NPU's shim DMA caps at ~27 GB/s and shows no cache sensitivity at all.

So the 4 MiB row above is not a curiosity — it is the GPU running out of MALL and never
touching the DRAM path the two engines share. Read that way, the efficiency figure is less
about arbitration between the engines than about whether the GPU is stalling on DRAM.
Tiling GPU work under 32 MB is the lever.

## Caveats

- Aggregate bandwidth falls under concurrency (178.6 → 148.0 GB/s), so the controller is
  not simply saturating and re-dividing; some throughput is lost outright. Worth
  understanding before sizing a partition.
- The 4 MiB GPU baseline is DVFS-noisy and measured *above* 100% retention when
  concurrent. Treat that row as "NPU is unaffected", not as a GPU speedup.
- Both loads are synthetic and run as separate processes with separate allocations. This
  bounds the contention question only; it says nothing yet about zero-copy handoff, which
  is the next thing to establish.
