# Native shared GPU/NPU memory

The 0.8 runtime uses libamdf for both devices. Allocate with
`execution::MemoryPlacement::Shared(device)` to establish access at allocation,
or adopt an owned GPU buffer through the tracked runtime. Owned adoption exports
a dma-buf and imports it for XDNA while retaining the original host mapping.
Aliases share host-access guards and cache-maintenance requirements.

`tests/fabric.rs`, `tests/heterogeneous.rs`, and the memory alias unit test cover
native allocation, changed-input replays, owner drops, visibility, and budgets.
`examples/shared_roundtrip.rs` and `examples/gemm_pipeline.rs` are self-contained
hardware probes. See [the execution guide](../docs/GPU-NPU.md).
