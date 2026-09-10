# Can GPU and NPU work be scheduled to read from MALL instead of DRAM?

Strix Halo's last-level cache is **MALL** (Memory Attached Last Level) — 32 MB, reported by
KFD as L3 on the GPU node:

```
L1  16/32/256 KB (per-CU)
L2  2 MB
L3  32 MB      ← MALL
```

MALL sits at the data fabric in front of the memory controllers rather than inside the
shader complex, so unlike RDNA2's GPU-attached Infinity Cache it is not GPU-private: any
fabric client can in principle hit it. That makes "schedule both engines to work out of
MALL" a coherent idea. The question is whether each engine can actually exploit it.

Answer: **the GPU enormously, the NPU not at all.**

## GPU: a 3.8× cliff exactly at 32 MB

`examples/gpu_load.rs`, write-only fill, 3s per point:

| working set | GB/s | |
|---:|---:|---|
| 1 MiB | 212.9 | dispatch overhead dominates |
| 2 MiB | 364.6 | |
| 4 MiB | 524.4 | |
| 8 MiB | 671.1 | |
| 16 MiB | 770.7 | |
| 24 MiB | **828.7** | peak, MALL-resident |
| 32 MiB | 815.1 | still inside MALL |
| 48 MiB | 220.6 | **cliff** |
| 64 MiB | 216.9 | |
| 128 MiB | 225.9 | |
| 256 MiB | 215.7 | DRAM plateau |

The knee falls between 32 and 48 MiB — the MALL boundary, exactly where the hardware says
it should. MALL-resident work runs at roughly **3.8× DRAM bandwidth**. Absolute numbers
drift a few tens of percent between sessions with clocks and thermals; the location of the
cliff does not.

## NPU: flat, no knee, capped far below DRAM

A GEMM cannot answer this — at ~1200 FLOP/byte it is compute-bound and looks identical
either way. So `examples/mall_probe.rs` drives mlir-aie's `passthrough_dmas` instead:
shim → memtile → shim, no compute tile, so throughput is purely a function of where the
bytes come from. Working set is input + output, putting the MALL boundary at 4 Mi elements.

| elements | working set | vs MALL | GB/s | disp/s |
|---:|---:|---|---:|---:|
| 262 144 | 2 MiB | fits | 12.5 | 5950.1 |
| 1 048 576 | 8 MiB | fits | 21.7 | 2582.2 |
| 2 097 152 | 16 MiB | fits | 24.7 | 1474.9 |
| 4 194 304 | 32 MiB | fits | 25.9 | 772.8 |
| 8 388 608 | 64 MiB | exceeds | 25.7 | 382.3 |
| 16 777 216 | 128 MiB | exceeds | 26.7 | 199.2 |
| 33 554 432 | 256 MiB | exceeds | 26.9 | 100.3 |

Flat across the boundary — 25.9 GB/s inside MALL, 26.9 GB/s from DRAM. The rise at the
small end is per-dispatch fixed cost being amortised (5950 dispatches/s at 2 MiB), not
caching.

The reason is that the NPU never reaches a memory limit at all: its shim DMA tops out
around **27 GB/s**, an eighth of what the GPU pulls from DRAM and a thirtieth of MALL.
A cache cannot help a client that cannot consume bandwidth fast enough to need one.

## What follows

**Do not design tile-pipelining to keep NPU data in MALL.** There is nothing there to win.
The symmetric "both engines read from MALL" scheduling idea does not pay, because only one
of the two engines is bandwidth-sensitive.

**Do keep GPU working sets under 32 MB.** It is worth 3.8× on the GPU side on its own, and
it is also the best thing you can do for the NPU: the contention runs showed the NPU
retaining **96%** of its solo rate against a MALL-resident GPU load versus **71%** against a
DRAM-streaming one (see [CONTENTION.md](CONTENTION.md)). Tiling GPU kernels to MALL keeps
the GPU off the DRAM path, which is the only path the two engines genuinely share.

This also reframes the earlier contention result. A concurrency efficiency of 1.55 looked
like two engines dividing one memory system. It is better read as the GPU stalling on DRAM
while the NPU draws its modest 26 GB/s — the engines are more independent than a shared
bandwidth model suggests, and the lever is GPU cache residency, not arbitration between
them.

## Reproducing

Build the passthrough sweep (the wheel in `ironenv` is older than the repo, so use the
design as of the wheel's commit):

```bash
cd ~/code/mlir-aie && source ironenv/bin/activate
git show 0d49a88:programming_examples/basic/passthrough_dmas/passthrough_dmas.py > /tmp/pt/passthrough_dmas.py
cd /tmp/pt
for N in 262144 1048576 2097152 4194304 8388608 16777216 33554432; do
  python3 passthrough_dmas.py --xclbin-path=/tmp/pt/n$N.xclbin --insts-path=/tmp/pt/n$N.bin -d npu2 -n $N
done
```

Then:

```bash
cargo build --release --features npu --example mall_probe
./target/release/examples/mall_probe /tmp/pt --seconds 4
```
