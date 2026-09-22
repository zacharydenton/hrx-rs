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

fn copy_artifact() -> Result<hrx::loom::Artifact> {
    hrx::loom::Compiler::for_target(None, &hrx::Target::xdna())?
        .module(include_str!("kernels/copy.xdna.loom"))
        .compile(&hrx::loom::Specialization::new("copy").with_config("copy.packets", "256"))
}

#[test]
#[ignore = "requires native Loom, libamdf and NPU5"]
fn npu_allocations_and_instruction_leases_respect_shared_budgets() -> Result<()> {
    let artifact = copy_artifact()?;
    let image_bytes = artifact.bytes().len();
    let manager = hrx::residency::ResidencyManager::new(image_bytes + 4096)?;
    let runtime = Runtime::with_options(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    let contract = || KernelContract {
        bindings: [(1 << 20, Access::Read), (1 << 20, Access::Write)]
            .into_iter()
            .map(|(bytes, access)| BindingContract {
                bytes,
                access,
                alignment: 4,
                layout: "bf16 contiguous".into(),
            })
            .collect(),
        constants: vec![],
    };
    let program = runtime.npu(0)?;
    let kernel = unsafe { program.load_artifact(&artifact, 1, contract()) }?;
    assert_eq!(manager.statistics().reserved_bytes, image_bytes);
    let buffer = runtime.allocate(4096, MemoryPlacement::NpuLocal(program.clone()))?;
    assert_eq!(manager.statistics().reserved_bytes, image_bytes + 4096);
    assert!(
        runtime
            .allocate(1, MemoryPlacement::NpuLocal(program.clone()))
            .is_err()
    );
    assert!(matches!(
        unsafe { program.load_artifact(&artifact, 1, contract()) },
        Err(hrx::Error::Busy(_))
    ));
    // The process-wide image cache must not inherit another runtime's budget.
    let tiny = hrx::residency::ResidencyManager::new(image_bytes - 1)?;
    let other = Runtime::with_options(hrx::execution::RuntimeOptions {
        memory_budget: Some(tiny.budget()),
        ..Default::default()
    })?;
    let other_program = other.npu(0)?;
    assert!(unsafe { other_program.load_artifact(&artifact, 1, contract()) }.is_err());
    assert_eq!(tiny.statistics().reserved_bytes, 0);
    let retained_kernel = kernel.clone();
    drop((kernel, program, buffer, runtime, other_program, other));
    assert_eq!(manager.statistics().reserved_bytes, image_bytes);
    drop(retained_kernel);
    assert_eq!(manager.statistics().reserved_bytes, 0);

    // Direct fabric clients charge the same backing through cloned ownership.
    let device = hrx::fabric::Device::open(hrx::fabric::Engine::Xdna, 0)?;
    let fabric = device.fabric();
    let root = fabric.allocate_budgeted(
        image_bytes + 4096,
        std::slice::from_ref(&device),
        &manager.budget(),
    )?;
    assert!(
        fabric
            .allocate_budgeted(1, std::slice::from_ref(&device), &manager.budget())
            .is_err()
    );
    let retained = root.clone();
    drop((root, fabric, device));
    assert_eq!(manager.statistics().reserved_bytes, image_bytes + 4096);
    drop(retained);
    assert_eq!(manager.statistics().reserved_bytes, 0);
    Ok(())
}

#[test]
#[ignore = "requires native Loom, libamdf, gfx1151 and NPU5"]
fn gpu_arithmetic_npu_dma_gpu_arithmetic() -> Result<()> {
    let artifact = copy_artifact()?;
    let runtime = Runtime::new()?;
    let program = runtime.npu(0)?;
    let bytes = 1 << 20;
    let elements = bytes / 2;
    let binding = |bytes, access| BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "bf16 contiguous".into(),
    };
    let npu = unsafe {
        program.load_artifact(
            &artifact,
            1,
            KernelContract {
                bindings: vec![binding(bytes, Access::Read), binding(bytes, Access::Write)],
                constants: vec![],
            },
        )
    }?;
    let compiler = hrx::loom::Compiler::for_target(None, runtime.gpu()?.target())?;
    let module = compiler.module(include_str!("kernels/euler.loom"));
    let mut specialization = hrx::loom::Specialization::new("krea2_euler");
    specialization.set_config("krea2.euler.grid_x", (elements / 256).to_string());
    specialization.set_config("krea2.euler.grid_y", "1");
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
    let c = allocate(bytes)?;
    let velocity = allocate(bytes)?;
    for word in velocity.map_write()?.as_chunks_mut::<2>().0 {
        word.copy_from_slice(&0x3f80u16.to_le_bytes());
    }
    let mut graph = runtime.graph();
    graph.gpu(&gpu, &[a.view(), velocity.view()])?;
    graph.npu(&npu, &[a.view(), c.view()])?;
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
        for word in a.map_write()?.as_chunks_mut::<2>().0 {
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
                .as_chunks::<2>()
                .0
                .iter()
                .all(|word| u16::from_le_bytes(*word) == expected)
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
#[ignore = "requires native Loom, libamdf and NPU5"]
fn npu_bindings_survive_a_different_program_context() -> Result<()> {
    let artifact = copy_artifact()?;
    let runtime = Runtime::new()?;
    let owner = runtime.npu(0)?;
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
            runtime.allocate(bytes, placement)?,
        ))
    })
    .collect::<Result<Vec<_>>>()?;
    // Program loads are weak-cached. Drop the program before loading again so
    // this creates a new hardware context while the BOs retain their old one.
    drop(owner);
    let consumer = runtime.npu(0)?;
    let kernel = unsafe {
        consumer.load_artifact(
            &artifact,
            1,
            KernelContract {
                bindings: vec![binding(bytes, Access::Read), binding(bytes, Access::Write)],
                constants: vec![],
            },
        )
    }?;
    for (input, output) in buffers {
        input.map_write()?.fill(0x6b);
        let mut graph = runtime.graph();
        graph.npu(&kernel, &[input.view(), output.view()])?;
        let graph = graph.prepare()?;
        graph.submit()?.wait()?;
        assert!(output.map_read()?.iter().all(|&byte| byte == 0x6b));
    }
    Ok(())
}
