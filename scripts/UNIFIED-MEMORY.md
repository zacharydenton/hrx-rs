# Zero-copy GPU↔NPU memory on Strix Halo

Both engines sit on the same LPDDR5X, but each driver pins pages into its own IOMMU
domain: the GPU through KFD, the NPU through amdxdna's DRM node. Physical sharing is not
the question — whether the two drivers will address the *same* pages is. This is what
decides whether a GPU→NPU handoff is a cache flush or a copy, and therefore what the NPU
API can promise.

## Routes tested

`examples/unified_probe.rs` and `examples/dmabuf_probe.rs`, both behind `--features npu-probe`:

```bash
cargo build --release --features npu-probe --example dmabuf_probe
./target/release/examples/dmabuf_probe ~/code/dinov3-xdna2/build/matmul/4096x1024/x.xclbin
```

| # | Route | Result |
|---|---|---|
| 1 | GPU imports anonymous host pages (`hrx_allocator_import_buffer`) | **works** — GPU reads and writes them in place |
| 2 | NPU imports those same host pages as an XRT userptr BO | fails — XRT tries `mmap(MAP_FIXED)` and gets `EINVAL` |
| 3 | GPU imports the NPU BO's own mapping | fails — `hsa_amd_memory_lock_to_pool` refuses driver-mapped pages |
| 4 | GPU exports HSA-locked host pages as a dma-buf | fails — `HSA_STATUS_ERROR_INVALID_ALLOCATION` (4099) |
| 5 | **GPU pool allocation → `hsa_amd_portable_export_dmabuf` → XRT `import_bo`** | **works, end to end** |

Route 5 is the one:

```
gpu memory pool: 126976 MiB
pool allocation: 4 MiB
hsa dma-buf export of pool memory: YES (fd 6, offset 0)
npu import of the GPU dma-buf: YES
  gpu write -> npu read over dma-buf: YES -- zero-copy confirmed
```

The GPU fills the allocation with `hsa_amd_memory_fill`; the NPU reads the same bytes back
through its imported BO. One physical allocation, two engines, no copy.

## Why the other routes fail

- **Host pointers are a dead end in both directions.** amdxdna will not pin pages another
  driver owns, and HSA will not lock pages amdxdna has mapped. Route 1 works only because
  those pages belong to nobody else yet.
- **HSA exports allocations, not registrations.** Route 4 fails because locked host memory
  is not an HSA pool allocation; only `hsa_amd_memory_pool_allocate` results are exportable.
- **The userptr failure is not a version problem.** XRT 2.23.0 (already built under
  `~/code/xdna-driver/build/Release/bins/lib`, reachable with `XILINX_XRT` pointed at a
  prefix) fails identically, and the userptr guard in `src/shim/buffer.cpp` is unchanged
  between that build and current `main`. Rebuilding will not move it.

## The one gap in the way

Route 5 currently has to bypass libhrx. `hsa_amd_portable_export_dmabuf` needs the device
pointer of a pool allocation, and `hrx_buffer_get_device_ptr` refuses device-local buffers:

```
hrx_buffer_get_device_ptr: cannot get device pointer for this buffer type
```

So the probe allocates from an HSA pool itself. Closing this means one of:

1. libhrx returns a device pointer for device-local buffers, or
2. libhrx grows `hrx_buffer_export_dmabuf` directly (the honest shape — it keeps the
   allocation owned by the runtime that made it).

Either is a small addition, and unlike the amdxdna HAL it is a GPU-side change, so it does
not depend on PR #37 landing.

## The API

`src/npu/` (feature `npu`) builds on route 5. `Npu::alloc` allocates from the GPU pool,
exports it once, and imports it into the NPU driver; `Shared` then tracks which engine last
touched the memory and does exactly the cache maintenance each transition needs:

```rust
let npu = hrx::npu::Npu::open("model.xclbin")?;
let mut activations = npu.alloc(32 << 20, npu.group_id(3)?)?;

activations.host()?.fill(0);      // host owns it
let bo = activations.npu();       // flushed; the NPU owns it and this is the kernel arg
let out = activations.host()?;    // invalidated back to the host
```

Re-entering the current owner is free, so a chain of NPU dispatches flushes once rather
than once per dispatch. `copy_from`/`copy_to` move data between a `Shared` and an ordinary
`Stream::allocate_shared` GPU buffer device-side, for the part of a pipeline that still
needs GPU kernels.

`examples/shared_roundtrip.rs` checks every transition (host→NPU, NPU→host, GPU→NPU,
GPU buffer→shared→NPU, NPU→shared→GPU buffer); all pass.

### What it costs

`examples/shared_bench.rs`, 64 MiB, 15 iterations:

| handoff | median ms | effective GB/s |
|---|---:|---:|
| shared (cache maintenance only) | 0.636 | 105.6 |
| staged (device copy, then hand off) | 2.810 | 23.9 |

**4.4× cheaper**, and the gap grows with buffer size, because the shared path moves no bytes
at all — the cost is cache maintenance, not bandwidth. Bandwidth is the resource the
contention measurements showed to be scarce, so this is spending the right currency.

## What this means for the API

Zero-copy is real, so the NPU API should be built around a buffer that is *born shared*
rather than one that is copied at the boundary — allocate once from the GPU pool, export
once, and let both engines address it for the buffer's whole life. Export is a per-buffer
setup cost, not a per-handoff cost.

Two constraints carry into the design:

- **Coherence is explicit.** The NPU side still needs `sync`/clflush around its accesses;
  the pages are shared but the caches are not. Ownership transfer has to be modelled, not
  assumed.
- **Synchronisation is host-side.** amdxdna's timeline semaphores are host-side and its
  dispatch queue is single-worker, so handoff should be coarse — whole tensors between
  stages, not per-layer ping-pong.
