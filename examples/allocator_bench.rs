//! Idle allocation latency, not inference throughput. Run with --release.
use hrx::{Gpu, sys};
fn main() -> hrx::Result<()> {
    let gpu = Gpu::open()?;
    unsafe {
        let mut device = std::ptr::null_mut();
        assert!(sys::is_ok(sys::hrx_gpu_device_get(0, &mut device)));
        let mut stream = std::ptr::null_mut();
        assert!(sys::is_ok(sys::hrx_stream_create(device, 0, &mut stream)));
        let mut queued = Vec::new();
        let mut direct = Vec::new();
        for i in 0..1100 {
            let start = std::time::Instant::now();
            let mut buffer = std::ptr::null_mut();
            assert!(sys::is_ok(sys::hrx_buffer_allocate(
                stream,
                1024 * 1024,
                sys::MEMORY_TYPE_DEVICE_LOCAL,
                sys::BUFFER_USAGE_DEFAULT,
                &mut buffer
            )));
            let elapsed = start.elapsed();
            sys::hrx_buffer_release(buffer);
            let start = std::time::Instant::now();
            let buffer = gpu.alloc(1024 * 1024)?;
            let elapsed_direct = start.elapsed();
            drop(buffer);
            if i >= 100 {
                queued.push(elapsed);
                direct.push(elapsed_direct);
            }
        }
        sys::hrx_stream_release(stream);
        queued.sort();
        direct.sort();
        println!(
            "1 MiB idle allocation, 1000 interleaved samples: native stream median {:?}; shared device allocator median {:?}",
            queued[500], direct[500]
        );
    }
    Ok(())
}
