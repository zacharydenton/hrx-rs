# GPU execution and inference

`hrx::gpu` exposes GPU devices, streams, buffers and graphs. These types are also
available at the crate root. `hrx::execution` adds GPU/NPU scheduling and tracked
buffer access. For cross-device setup, see [GPU/NPU execution](GPU-NPU.md).

## Streams and buffers

Streams share up to eight native queues per device by default. Each stream stays
on one queue; assignment is round-robin. A long kernel or event wait can delay
other streams on that queue. Set `device.set_stream_queue_count(count)?` before
creating the first stream. Queues are created lazily. See the
[queue pool benchmarks](../benchmarks/queue-pool/README.md) for sizing details.

Buffers belong to a device, so multiple streams on that device can use them.
Order conflicting access with `record_event` and `wait_event`. Bind a whole
buffer with `buffer.binding()` or a subregion with `view.slice(offset, length)?`.
Dispatches, fills and copies take `&self`; staging transfers and synchronization
need `&mut`. Cloning a `Kernel` retains its executable.

Copies and fills accept logical ranges larger than 4 GiB. HRX splits them to fit
SDMA ring capacity or compute dispatch limits, including in recorded graphs.
One completion covers the entire transfer, and HRX retains its buffers through
retirement. Callers do not need a 64 MiB chunking loop. Ranges must be nonempty
and in bounds; copy source and destination must not overlap.

`upload` copies host bytes through reusable staging chunks of at most 64 MiB;
large uploads wait between chunks to keep staging bounded. `upload_blocking`
also waits for the final chunk. HRX checks the full destination before submission.
If submission fails after some chunks were accepted, those chunks can still
modify the destination; streams keep them on their completion timeline.
`stream.allocate_from(bytes)` initializes a new buffer directly from a nonempty
host slice, avoiding a staging copy and a separate zero-fill pass.

Ordinary GPU buffers use cacheable system memory. Stream transfers perform host
cache maintenance. Direct host-pointer access requires external synchronization
and `Buffer::cache_control` before host reads and after host writes, or coherent
storage from `allocate_shared`.

## Graphs

`Stream::graph` records operations with explicit predecessor lists. `&[]` starts
an independent branch. Pass branch endings to their consumer; use `join` when
an empty dependency node is needed. Each join adds a native partition and queue
barrier. Independent nodes can batch without intervening barriers; actual overlap
depends on resource use.

`Stream::access_graph` infers dependencies from each binding's `read()`, `write()`
or `read_write()` declaration and byte range.

`Stream::owned_graph` prepares commands during recording and retains their
buffers, kernels and allocation charges through replay. Recording does not
execute work. `Graph::binding_bytes` reports distinct bound allocation bytes.

`BufferPool` provides bounded best-fit allocation reuse; dropping a `PooledBuffer`
returns its allocation to the pool. A graph retaining native backing does not
prevent pool reuse. Keep pooled buffers leased until their recorded work no
longer needs them.

Loading kernels, dispatching them and sharing buffers across streams require
valid code, arguments, access declarations and synchronization.

## Queue engines

`fabric::Endpoint::queue_capabilities` and `hrx doctor` list native queue families.
PM4 is the default compute engine. `Device::aql_queue(maximum_private_bytes)`
creates an AQL v1 queue with fixed scratch; select it in the tracked runtime with
`RuntimeOptions::compute_engine = ComputeEngine::Aql { maximum_private_bytes: 0 }`.
Prepared dispatches retain code, arguments, scratch and buffers until completion.

`Device::sdma_queue` supports copies, fills, unaligned tails and PM4/SDMA waits.
`RuntimeOptions::copy_engine = CopyEngine::Sdma` selects SDMA for copy-only graph
regions. Allocations must support that queue; coherent backing from
`Fabric::allocate_shared` does on the supported provider. Mixed PM4/AQL/SDMA
regions use host completion checks to order dependencies.

Fabric completion `is_complete` reads cached state; `refresh` polls and retires
work, and `wait` drives progress. Stream event queries poll automatically.

## Models and composed pipelines

`hrx::model::ModelSession` owns a kernel batch, buffer arena and graph cache.
It infers dependencies from declared binding access and reuses readback storage.
Regions and kernel IDs are session-scoped. Applications validate model files,
shapes and native-code contracts.

`hrx::inference::ModelContext` shares allocation, compilation and scheduling.
`ModelSession::freeze` creates shared immutable weights and code;
`ModelDefinition::prepare` creates private inference slots. `DeviceTensor` views
retain metadata, producer completions and slot leases, preventing output reuse
while a consumer holds a view.

Use `ModelDefinition::fragment` to validate a stage, then `ModelFragment::record`
to append it to an `execution::Graph`. Pass output tensors directly to the next
stage and prepare the combined graph once. Adjacent GPU stages can then run
without intermediate copies or submissions.

`PreparedModel::prepare` takes a slot factory returning
`InferenceGraph { inputs, outputs, graph }`. Each slot owns its bindings, including
sliced or in-place I/O. Independent slots must not share writable I/O. Host staging
is allocated on first upload/readback and reused; device-only pipelines need none.
`ModelSlot::submit_host_with` writes packed inputs directly into staging.

Fragments may opt into `reuse_private_scratch` when they initialize every
temporary byte they read. Graph dependencies order reuse between stages; inputs,
outputs and weights remain distinct, and independent slots retain private scratch.

`Graph::gpu_scoped` integrates native clients with tracked buffers.
`execution::NativeSession` serves synchronous, stream-bound models with borrowed
inputs and progress callbacks. It reserves the compute lane on the calling
thread, fences errors and panics, and retains owners when completion is uncertain.
Its transfers remain part of that synchronous stage.

## Tensor and image operations

`TensorOps::linear` projects contiguous BF16 `[1, K]` inputs against `[N, K]`
weights, with K divisible by 128 and BF16 or F32 output. `linear_many` combines
up to three projections in one dispatch; `linear_fragment` records a projection
in a graph. Larger matrix products need caller-provided kernels.

`TensorOps::gather_rows` accepts host-selected indices and copies complete rows
on the device, preserving dtype bits, duplicates and order. Plans use bounded
shape caches; returned views retain output slots.

`hrx::image` provides RGB normalization, patchification, resize, affine sampling
and compositing. FP32 affine sampling uses black borders and ties-to-even byte
rounding. Plans reuse private slots across changing inverse matrices.

## Memory budgets and files

`PlanCache` bounds concurrent shape preparation and evicts idle plans by LRU.
`ResidencyManager` adds byte budgets and persistent pins. `load_budgeted` caches
units whose allocations carry their own charges; private workspaces can grow
while leased, and only idle units can be evicted.

Pass the manager's `budget()` to `RuntimeOptions::memory_budget` to charge weights,
scratch and transfer staging before allocation. Standalone streams use
`with_memory_budget`. NPU instruction buffers and tracked NPU storage also count.
Aliases and pending or quarantined work retain the backing owner's charge.

`ModelSession::in_context` applies the budget during native loading. Freezing
into the same budget does not double-charge weights; other adopted buffers are
charged at adoption. Do not count the same bytes in both a declared resource
reservation and its budgeted allocations. Tracked extents exclude native allocator
rounding and compiler/code-object memory. They are not process RSS.

Runtime statistics report transfer bytes and live/peak tracked memory.
Completion profiles and bounded traces measure host-observed latency.

`hrx::artifacts` provides shared model-file utilities:

- `hf::Resolver`: Hugging Face cache lookup, pinned revisions, offline operation,
  download progress and SHA-256 verification.
- `onnx::Model`: owned graph indexes and checked attribute/tensor decoding.
- `safetensors::FileView`: owned or mapped files, checked tensor ranges and page
  advice. See [native storage](STORAGE.md) for file-backed GPU loading.
