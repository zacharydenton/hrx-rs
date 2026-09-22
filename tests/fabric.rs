//! Tests of the new native ownership boundary without the legacy runtime.
use hrx::fabric::{Engine, Fabric};

#[test]
#[ignore = "requires native compiler, bridge, libamdf and gfx1151 hardware"]
fn gpu_arithmetic_uses_native_code_loading_and_completion_fences() -> hrx::Result<()> {
    use hrx::{
        fabric::Argument,
        loom::{Compiler, CxxSource, Specialization},
    };
    let path = std::env::var_os("HRX_AMDF_LIBRARY").expect("set HRX_AMDF_LIBRARY");
    let fabric = Fabric::load(std::path::Path::new(&path))?;
    let gpu = fabric
        .endpoints()?
        .into_iter()
        .find(|e| e.engine() == Engine::Gpu)
        .unwrap()
        .open()?;
    let compiler = Compiler::for_target(None, gpu.target())?;
    let source = CxxSource::new(
        "fill.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fill(unsigned* output) { output[threadIdx.x] = threadIdx.x * 3u + 7u; }
"#,
    );
    let artifact = compiler
        .import_cxx(source)?
        .compile(&Specialization::new("fill"))?;
    // The source above writes exactly 64 u32 values to its sole binding.
    let kernel = unsafe { gpu.load(&artifact) }?;
    let queue = gpu.queue()?;
    let output = fabric.allocate(256, std::slice::from_ref(&gpu))?;
    let prepared = unsafe {
        queue.prepare(
            &kernel,
            [1, 1, 1],
            [64, 1, 1],
            &[Argument::Buffer(&output, 0)],
        )
    }?;
    let mut earlier: Option<hrx::fabric::Completion> = None;
    for _ in 0..32 {
        output.write(0, &[0xcd; 256])?;
        let done = unsafe { prepared.dispatch() }?;
        if let Some(earlier) = &earlier {
            assert!(earlier.is_complete()?);
        }
        assert!(matches!(output.write(0, &[0]), Err(hrx::Error::Busy(_))));
        if !done.wait_timeout(std::time::Duration::from_secs(10))? {
            // Keep the hardware ownership chain live after a failed qualification.
            std::mem::forget(done);
            return Err(hrx::Error::DeviceLost(
                "native arithmetic did not retire within 10 seconds".into(),
            ));
        }
        earlier = Some(done);
        let mut bytes = [0; 256];
        output.read(0, &mut bytes)?;
        for (index, value) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(
                u32::from_le_bytes(value.try_into().unwrap()),
                index as u32 * 3 + 7
            );
        }
    }
    assert!(
        unsafe {
            queue.dispatch(
                &kernel,
                [1, 1, 1],
                [32, 1, 1],
                &[Argument::Buffer(&output, 0)],
            )
        }
        .is_err()
    );
    assert!(unsafe { queue.dispatch(&kernel, [1, 1, 1], [64, 1, 1], &[]) }.is_err());
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
#[ignore = "requires HRX_AMDF_LIBRARY and gfx1151/NPU5 hardware"]
fn native_devices_retain_instance_and_endpoint_owners() -> hrx::Result<()> {
    let path = std::env::var_os("HRX_AMDF_LIBRARY").expect("set HRX_AMDF_LIBRARY");
    let fabric = Fabric::load(std::path::Path::new(&path))?;
    let endpoints = fabric.endpoints()?;
    assert!(
        endpoints
            .iter()
            .any(|e| e.engine() == Engine::Gpu && e.target().as_str() == "gfx1151")
    );
    assert!(
        endpoints
            .iter()
            .any(|e| e.engine() == Engine::Xdna && e.target().is_xdna())
    );
    let devices = endpoints
        .iter()
        .map(|e| e.open())
        .collect::<hrx::Result<Vec<_>>>()?;
    let buffer = fabric.allocate(4096, &devices)?;
    for device in &devices {
        buffer.device_address(device)?;
    }
    buffer.write(17, &[3, 5, 7, 11])?;
    assert!(buffer.write(4095, &[1, 2]).is_err());
    assert!(
        fabric
            .allocate(1, &[devices[0].clone(), devices[0].clone()])
            .is_err()
    );
    drop(endpoints);
    drop(fabric);
    assert_eq!(devices.len(), 2);
    assert!(devices.iter().all(|d| !d.endpoint().name().is_empty()));
    drop(devices);
    let mut result = [0; 4];
    buffer.read(17, &mut result)?;
    assert_eq!(result, [3, 5, 7, 11]);
    drop(buffer);
    // Instance lifetime must support ordered teardown and subsequent recreation.
    let fabric = Fabric::load(std::path::Path::new(&path))?;
    for endpoint in fabric.endpoints()? {
        drop(endpoint.open()?);
    }
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
#[ignore = "requires native compiler, bridge, libamdf and gfx1151/NPU5 hardware"]
fn xdna_reuses_establishing_commands_on_shared_gpu_backing() -> hrx::Result<()> {
    use hrx::{
        fabric::{Argument, XdnaBinding},
        loom::{Compiler, CxxSource, Specialization},
    };
    let path = std::env::var_os("HRX_AMDF_LIBRARY").expect("set HRX_AMDF_LIBRARY");
    let fabric = Fabric::load(std::path::Path::new(&path))?;
    let endpoints = fabric.endpoints()?;
    let gpu = endpoints
        .iter()
        .find(|e| e.engine() == Engine::Gpu)
        .unwrap()
        .open()?;
    let npu = endpoints
        .iter()
        .find(|e| e.engine() == Engine::Xdna)
        .unwrap()
        .open()?;
    let devices = [gpu.clone(), npu.clone()];
    let lhs = fabric.allocate(256, &devices)?;
    let rhs = fabric.allocate(256, &devices)?;
    let output = fabric.allocate(256, &devices)?;
    let checked = fabric.allocate(256, &devices)?;
    let compiler = Compiler::for_target(None, gpu.target())?;
    let source = CxxSource::new(
        "shared.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fill(unsigned* lhs, unsigned* rhs) {
  lhs[threadIdx.x] = threadIdx.x + 1u;
  rhs[threadIdx.x] = threadIdx.x + 3u;
}
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void consume(unsigned* input, unsigned* output) {
  if (threadIdx.x < 16) output[threadIdx.x] = input[threadIdx.x] + 11u;
}
"#,
    );
    let module = compiler.import_cxx(source)?;
    let fill = unsafe { gpu.load(&module.compile(&Specialization::new("fill"))?) }?;
    let consume = unsafe { gpu.load(&module.compile(&Specialization::new("consume"))?) }?;
    let queue = gpu.queue()?;
    let compiler = Compiler::for_target(None, npu.target())?;
    let artifact = compiler
        .module(include_str!("kernels/mul_i32.xdna.loom"))
        .compile(&Specialization::new("mul_i32"))?;
    let bindings = [&lhs, &rhs, &output].map(|buffer| XdnaBinding {
        buffer,
        offset: 0,
        length: 64,
    });
    // Trusted fixture reads/writes 16 i32 elements and requires one column.
    let program = unsafe { npu.prepare_xdna(&artifact, 1, &bindings) }?;
    // Exercise separately created, time-sliced contexts between repeated uses.
    let other = unsafe { npu.prepare_xdna(&artifact, 1, &bindings) }?;
    for iteration in 0..8 {
        output.write(0, &[0xcd; 256])?;
        unsafe {
            queue.dispatch(
                &fill,
                [1, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&lhs, 0), Argument::Buffer(&rhs, 0)],
            )
        }?
        .wait()?;
        let selected = if iteration % 2 == 0 { &program } else { &other };
        let done = unsafe { selected.dispatch() }?;
        assert!(matches!(
            output.read(0, &mut [0; 4]),
            Err(hrx::Error::Busy(_))
        ));
        if !done.wait_timeout(std::time::Duration::from_secs(10))? {
            std::mem::forget(done);
            return Err(hrx::Error::DeviceLost("XDNA invocation timed out".into()));
        }
        unsafe {
            queue.dispatch(
                &consume,
                [1, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&output, 0), Argument::Buffer(&checked, 0)],
            )
        }?
        .wait()?;
        let mut result = [0; 64];
        checked.read(0, &mut result)?;
        for (index, bytes) in result.chunks_exact(4).enumerate() {
            assert_eq!(
                u32::from_le_bytes(bytes.try_into().unwrap()),
                (index as u32 + 1) * (index as u32 + 3) + 11
            );
        }
    }
    // Invalid logical bindings fail before any native command is published.
    let short = [&lhs, &rhs, &output].map(|buffer| XdnaBinding {
        buffer,
        offset: 0,
        length: 4,
    });
    assert!(unsafe { npu.prepare_xdna(&artifact, 1, &short) }.is_err());
    // The program retains its device and external bindings across caller drops.
    drop(devices);
    drop(lhs);
    drop(rhs);
    drop(output);
    drop(gpu);
    drop(npu);
    drop(endpoints);
    drop(fabric);
    unsafe { program.dispatch() }?.wait()?;
    drop(other);
    drop(program);
    Ok(())
}

#[test]
#[ignore = "requires native compiler, bridge, libamdf and gfx1151 hardware"]
fn private_segment_scratch_is_backed_and_retained() -> hrx::Result<()> {
    use hrx::{
        fabric::Argument,
        loom::{Compiler, ReportMode, Specialization},
    };
    let path = std::env::var_os("HRX_AMDF_LIBRARY").expect("set HRX_AMDF_LIBRARY");
    let fabric = Fabric::load(std::path::Path::new(&path))?;
    let gpu = fabric
        .endpoints()?
        .into_iter()
        .find(|e| e.engine() == Engine::Gpu)
        .unwrap()
        .open()?;
    let compiler = Compiler::for_target(None, gpu.target())?;
    let artifact = compiler
        .module(include_str!("kernels/scratch.loom"))
        .compile(&Specialization::new("scratch").with_report(ReportMode::Summary))?;
    assert!(
        artifact
            .report()
            .unwrap()
            .entries()?
            .iter()
            .any(|row| row.private_bytes.is_some_and(|bytes| bytes > 0)),
        "fixture must exercise private storage"
    );
    let kernel = unsafe { gpu.load(&artifact) }?;
    let queue = gpu.queue()?;
    let output = fabric.allocate(256, std::slice::from_ref(&gpu))?;
    let index = fabric.allocate(256, std::slice::from_ref(&gpu))?;
    let indices: Vec<u8> = (0u32..64)
        .flat_map(|i| ((i * 7) % 64).to_le_bytes())
        .collect();
    index.write(0, &indices)?;
    for _ in 0..4 {
        let done = unsafe {
            queue.dispatch(
                &kernel,
                [1, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&output, 0), Argument::Buffer(&index, 0)],
            )
        }?;
        if !done.wait_timeout(std::time::Duration::from_secs(10))? {
            std::mem::forget(done);
            return Err(hrx::Error::DeviceLost("scratch kernel timed out".into()));
        }
        let mut bytes = [0; 256];
        output.read(0, &mut bytes)?;
        for (i, bytes) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(
                u32::from_le_bytes(bytes.try_into().unwrap()),
                ((i as u32 * 7) % 64) * 3 + i as u32
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires native compiler, bridge, libamdf and gfx1151 hardware"]
fn native_transfers_preserve_unaligned_ranges_and_guards() -> hrx::Result<()> {
    use hrx::fabric::Device;
    let gpu = Device::open(Engine::Gpu, 0)?;
    assert_eq!(gpu.id(), Device::open(Engine::Gpu, 0)?.id());
    let fabric = gpu.fabric();
    let source = fabric.allocate(8192, std::slice::from_ref(&gpu))?;
    let destination = fabric.allocate(8192, std::slice::from_ref(&gpu))?;
    let queue = gpu.queue()?;
    source.write(0, &(0..8192).map(|i| (i % 251) as u8).collect::<Vec<_>>())?;
    destination.write(0, &[0xa5; 8192])?;
    let copy = queue.prepare_copy(&destination, 5, &source, 3, 4097)?;
    unsafe { copy.dispatch() }?.wait()?;
    let mut bytes = [0; 8192];
    destination.read(0, &mut bytes)?;
    assert_eq!(&bytes[..5], &[0xa5; 5]);
    assert!(bytes[4102..].iter().all(|byte| *byte == 0xa5));
    for (index, byte) in bytes[5..4102].iter().enumerate() {
        assert_eq!(*byte, ((index + 3) % 251) as u8);
    }
    let fill = queue.prepare_fill(&destination, 7, 4099, 0x37)?;
    unsafe { fill.dispatch() }?.wait()?;
    destination.read(0, &mut bytes)?;
    assert!(bytes[7..4106].iter().all(|byte| *byte == 0x37));
    assert!(bytes[4106..].iter().all(|byte| *byte == 0xa5));
    assert!(queue.prepare_copy(&source, 1, &source, 0, 2).is_err());
    assert!(queue.prepare_fill(&source, 8190, 4, 0).is_err());
    Ok(())
}

#[test]
#[ignore = "requires native compiler, bridge, libamdf and gfx1151"]
fn malformed_images_are_rejected_before_dispatch_and_valid_loads_recover() -> hrx::Result<()> {
    use hrx::{
        fabric::Device,
        loom::{Compiler, Specialization},
    };
    let gpu = Device::open(Engine::Gpu, 0)?;
    let artifact = Compiler::for_target(None, gpu.target())?
        .module(include_str!("kernels/scratch.loom"))
        .compile(&Specialization::new("scratch"))?;
    let bytes = artifact.bytes();
    for length in [0, 4, 63, bytes.len() / 2] {
        assert!(unsafe { gpu.load_bytes(&bytes[..length], "scratch") }.is_err());
    }
    for (offset, replacement) in [(0, 0), (4, 1), (5, 2), (16, 2), (18, 0), (54, 0), (58, 0)] {
        let mut corrupt = bytes.to_vec();
        corrupt[offset] = replacement;
        assert!(
            unsafe { gpu.load_bytes(&corrupt, "scratch") }.is_err(),
            "offset {offset}"
        );
    }
    for offset in [32, 40] {
        let mut corrupt = bytes.to_vec();
        corrupt[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(unsafe { gpu.load_bytes(&corrupt, "scratch") }.is_err());
    }
    assert!(unsafe { gpu.load_bytes(bytes, "missing_entry") }.is_err());
    drop(unsafe { gpu.load(&artifact) }?);
    Ok(())
}
