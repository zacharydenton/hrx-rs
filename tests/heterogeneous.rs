//! Hardware validation uses a real NPU DMA program and GPU arithmetic kernels.
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract, MemoryPlacement, Runtime},
};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
struct CountAllocations;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl std::alloc::GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { std::alloc::System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

#[test]
#[ignore = "requires gfx1151, XDNA2, shared ABI 1, and HRX_TEST_NPU_DIR passthrough artifacts"]
fn gpu_arithmetic_npu_dma_gpu_arithmetic() -> Result<()> {
    let directory =
        std::path::PathBuf::from(std::env::var_os("HRX_TEST_NPU_DIR").ok_or_else(|| {
            hrx::Error::Message(
                "set HRX_TEST_NPU_DIR to the 262144-element passthrough artifact directory".into(),
            )
        })?);
    let runtime = Runtime::new()?;
    let program = unsafe { runtime.npu(0)?.load_program(directory.join("x.xclbin")) }?;
    let bytes = 1 << 20;
    let elements = bytes / 2;
    let binding = |bytes, access| BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "bf16 contiguous".into(),
    };
    let npu = unsafe {
        program.kernel(
            &std::fs::read(directory.join("x.bin"))?,
            KernelContract {
                bindings: vec![
                    binding(bytes, Access::Read),
                    binding(4096, Access::Read),
                    binding(bytes, Access::Write),
                ],
                constants: vec![],
            },
        )
    }?;
    let compiler = hrx::loom::Compiler::with_options(
        None,
        hrx::loom::CompilerOptions {
            target: runtime.gpu()?.target().clone(),
            ..Default::default()
        },
    )?;
    let module = compiler.module(include_str!("kernels/euler.loom"));
    let mut specialization = hrx::loom::Specialization::new("krea2_euler");
    specialization
        .config
        .insert("krea2.euler.grid_x".into(), (elements / 256).to_string());
    specialization
        .config
        .insert("krea2.euler.grid_y".into(), "1".into());
    let artifact = module.compile(&specialization)?;
    let stream = hrx::gpu::Stream::open()?;
    let raw = unsafe { stream.load_artifact(&artifact) }?;
    let mut constants = hrx::gpu::Constants::new();
    match raw.info().constant_byte_length {
        8 => constants.push(elements as u32)?,
        12 => constants.push(elements as u64)?,
        _ => return Err(hrx::Error::Message("unexpected Euler scalar ABI".into())),
    }
    constants.push(1f32)?;
    let gpu = unsafe {
        runtime.load_gpu_kernel(
            artifact.path(),
            "krea2_euler",
            [(elements / 256) as u32, 1, 1],
            [256, 1, 1],
            KernelContract {
                bindings: vec![
                    binding(bytes, Access::ReadWrite),
                    binding(bytes, Access::Read),
                ],
                constants: constants.as_bytes().to_vec(),
            },
        )
    }?;
    let allocate = |size| runtime.allocate(size, MemoryPlacement::Shared(program.clone()));
    let a = allocate(bytes)?;
    let unused = allocate(4096)?;
    let c = allocate(bytes)?;
    let velocity = allocate(bytes)?;
    for word in velocity.map_write()?.chunks_exact_mut(2) {
        word.copy_from_slice(&0x3f80u16.to_le_bytes());
    }
    let mut graph = runtime.graph();
    graph.gpu(&gpu, &[a.view(), velocity.view()])?;
    graph.npu(&npu, &[a.view(), unused.view(), c.view()])?;
    graph.gpu(&gpu, &[c.view(), velocity.view()])?;
    let graph = graph.prepare()?;
    // Exercise an executor-neutral future as well as the blocking fast path.
    struct WakeThread(std::thread::Thread);
    impl std::task::Wake for WakeThread {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(WakeThread(std::thread::current())));
    let wait_future = |mut completion: hrx::Completion| -> Result<()> {
        use std::future::Future;
        let mut context = std::task::Context::from_waker(&waker);
        loop {
            match std::pin::Pin::new(&mut completion).poll(&mut context) {
                std::task::Poll::Ready(result) => return result,
                std::task::Poll::Pending => std::thread::park(),
            }
        }
    };
    graph.submit()?.wait()?;
    wait_future(graph.submit()?)?; // Warm parking and waker infrastructure before counting.
    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    for iteration in 0..16 {
        // Different initialized values catch stale versions across graph reuse.
        let initial = if iteration % 2 == 0 {
            0x3f80u16
        } else {
            0x4000u16
        };
        for word in a.map_write()?.chunks_exact_mut(2) {
            word.copy_from_slice(&initial.to_le_bytes());
        }
        let expected = if iteration % 2 == 0 {
            0x4040u16
        } else {
            0x4080u16
        };
        if iteration % 2 == 0 {
            graph.submit()?.wait()?;
        } else {
            wait_future(graph.submit()?)?;
        }
        assert!(
            c.map_read()?
                .chunks_exact(2)
                .all(|word| u16::from_le_bytes([word[0], word[1]]) == expected)
        );
    }
    TRACK.store(false, Ordering::Relaxed);
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "prepared replay allocated Rust heap memory"
    );
    Ok(())
}

#[test]
#[ignore = "requires XDNA2, shared ABI 1 and HRX_TEST_NPU_DIR passthrough artifacts"]
fn npu_bindings_survive_a_different_program_context() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("HRX_TEST_NPU_DIR")
            .ok_or_else(|| hrx::Error::Message("set HRX_TEST_NPU_DIR".into()))?,
    );
    let runtime = Runtime::new()?;
    let owner = unsafe { runtime.npu(0)?.load_program(directory.join("x.xclbin")) }?;
    let bytes = 1 << 20;
    let binding = |bytes, access| BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "passthrough".into(),
    };
    let buffers = [
        MemoryPlacement::NpuLocal(owner.clone()),
        MemoryPlacement::Shared(owner.clone()),
    ]
    .into_iter()
    .map(|placement| -> Result<_> {
        Ok((
            runtime.allocate(bytes, placement.clone())?,
            runtime.allocate(4096, placement.clone())?,
            runtime.allocate(bytes, placement)?,
        ))
    })
    .collect::<Result<Vec<_>>>()?;
    // Program loads are weak-cached. Drop the program before loading again so
    // this creates a new hardware context while the BOs retain their old one.
    drop(owner);
    let consumer = unsafe { runtime.npu(0)?.load_program(directory.join("x.xclbin")) }?;
    let kernel = unsafe {
        consumer.kernel(
            &std::fs::read(directory.join("x.bin"))?,
            KernelContract {
                bindings: vec![
                    binding(bytes, Access::Read),
                    binding(4096, Access::Read),
                    binding(bytes, Access::Write),
                ],
                constants: vec![],
            },
        )
    }?;
    for (input, unused, output) in buffers {
        input.map_write()?.fill(0x6b);
        let mut graph = runtime.graph();
        graph.npu(&kernel, &[input.view(), unused.view(), output.view()])?;
        let graph = graph.prepare()?;
        graph.submit()?.wait()?;
        assert!(output.map_read()?.iter().all(|&byte| byte == 0x6b));
    }
    Ok(())
}
