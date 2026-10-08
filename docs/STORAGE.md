# Native storage

Linux `storage::StorageSession` owns a GPU-authored io_uring queue, retained file
handles, a service thread, and a bounded pool of registered system pages. Select
`NativeLifetime::Process` when opening the stream or runtime:

```rust,no_run
use hrx::{
    fabric::NativeLifetime,
    storage::{StorageConfig, StorageSession},
};

fn main() -> hrx::Result<()> {
    let device = hrx::Device::open_with_lifetime(0, NativeLifetime::Process)?;
    let mut stream = device.stream()?;
    let file = std::fs::File::open("checkpoint.bin")?;
    let storage = StorageSession::new(&stream, &[file], StorageConfig::default())?;
    let ticket = storage.read(0, 0, 4096)?;
    let output = stream.allocate(4096)?;
    let mut transfer = ticket.wait()?.copy_to(&mut stream, output.binding())?;
    transfer.wait()?;
    Ok(())
}
```

## Capacity and ownership

The default pool is four 16 MiB slots with 64 native SQ entries. Identical retained
reads share a slot; tickets and consumers keep it unavailable for reuse. A full
pool returns `Error::Busy`. Payloads, ring pages, metadata and commands use the
stream's allocation budget. Timed-out native work retains its owners and charges.
Session drop drains accepted requests. The first I/O failure stops new issuance.

## I/O modes

`StorageMode::Buffered` is the default. Direct mode requires filesystem-reported `STATX_DIOALIGN`; enclosing reads hide alignment padding,
while writes require aligned offsets and lengths. Unsupported direct I/O fails.
`StorageProgress::Sqpoll` uses the kernel poller; `Wait` services deferred work on
the dedicated thread through eventfd. Write completion does not imply durability.
Host access can return `Error::Busy` while the device is using the registered allocation.
This path transfers through GPU-visible system pages; it is not NVMe-to-VRAM peer
DMA. It requires the native Linux io_uring features supported by the bundle.

## Statistics and model files

Set `StorageConfig::statistics` for host-observed queue, completion and service CPU
intervals. Counters distinguish logical reads, shared requests, physical requests,
and peak retained slots. `FileView::file_range` resolves tensor subranges to a
retained descriptor and absolute offset without faulting mapped tensor data.

## Custom GPU I/O programs

`fabric::StorageRing` exposes the ring for caller-authored GPU programs. Register
fixed regular files and payload from `Fabric::allocate_registered`, using
`NativeLifetime::Process`. `StorageLayout` distinguishes CPU payload addresses
for Linux SQEs from GPU addresses for shader access.

`StorageExecution` retains the dispatch, files, registered pages and budget
charges until GPU work and kernel I/O retire. Custom programs must handle ring
capacity, partial and error completions, and final drain. `StorageSession`
implements this protocol for ordinary reads and writes.

Run the hardware tests serially:

```sh
cargo test --all-features --test storage_session -- --ignored --test-threads=1
cargo test --all-features --test gpu_storage -- --ignored --test-threads=1
```
