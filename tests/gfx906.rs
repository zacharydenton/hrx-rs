//! Radeon VII qualification. Run with the candidate native libraries selected.
use hrx::{
    Error, Result,
    fabric::{Argument, Device, Engine},
    loom::{Compiler, CxxSource, Specialization},
};
use std::time::Duration;

#[test]
#[ignore = "requires gfx906 and the matching native compiler/runtime"]
fn fp64_device_local_transfers_and_repeated_dispatch() -> Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    assert_eq!(device.target().as_str(), "gfx906");
    let compiler = Compiler::for_target(None, device.target())?;
    let source = CxxSource::new(
        "gfx906.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(4, 1, 1)]]
void fp64(const double* input, double* output) {
    unsigned i = blockIdx.x * 64 + threadIdx.x;
    double x = input[i];
    output[i] = x / (double)(i + 1) + x * 0.125;
}
"#,
    );
    let artifact = compiler
        .import_cxx(source)?
        .compile(&Specialization::new("fp64"))?;
    // SAFETY: authored scratch-free kernel has two 256-double buffer arguments.
    let kernel = unsafe { device.load_bytes(artifact.bytes(), "fp64") }?;
    assert_eq!(kernel.subgroup_size(), 64);
    let fabric = device.fabric();
    let queue = device.queue()?;
    let staging = fabric.allocate_shared(2048, std::slice::from_ref(&device))?;
    let readback = fabric.allocate_shared(2176, std::slice::from_ref(&device))?;
    // SAFETY: every input byte is initialized by a completed copy before use;
    // the output is fully initialized by the kernel before readback.
    let input = unsafe { fabric.allocate_device_local(2048, &device) }?;
    // SAFETY: the kernel writes the full allocation before the copy reads it.
    let output = unsafe { fabric.allocate_device_local(2048, &device) }?;
    assert!(matches!(
        input.read(0, &mut [0; 8]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        input.write(0, &[0; 8]),
        Err(Error::Unsupported(_))
    ));
    for pass in 1..=16 {
        let values: Vec<f64> = (1..=256).map(|i| f64::from(i * pass) * 2.5).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        staging.write(0, &bytes)?;
        readback.write(0, &[0xa5; 2176])?;
        // SAFETY: initialized source, checked disjoint extents, ordered waits.
        let uploaded = unsafe { queue.prepare_copy(&input, 0, &staging, 0, 2048)?.dispatch() }?;
        assert!(uploaded.wait_timeout(Duration::from_secs(10))?);
        // SAFETY: wave64 geometry covers each initialized input/output element once.
        let done = unsafe {
            queue.dispatch(
                &kernel,
                [4, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&input, 0), Argument::Buffer(&output, 0)],
            )
        }?;
        assert!(done.wait_timeout(Duration::from_secs(10))?);
        // SAFETY: the complete output retired; the copy preserves both guards.
        let copied = unsafe {
            queue
                .prepare_copy(&readback, 64, &output, 0, 2048)?
                .dispatch()
        }?;
        assert!(copied.wait_timeout(Duration::from_secs(10))?);
        let mut actual = [0; 2176];
        readback.read(0, &mut actual)?;
        assert!(
            actual[..64]
                .iter()
                .chain(&actual[2112..])
                .all(|&x| x == 0xa5)
        );
        for (i, word) in actual[64..2112].chunks_exact(8).enumerate() {
            let observed = f64::from_le_bytes(word.try_into().unwrap());
            let expected = f64::from(pass) * 2.5 + values[i] * 0.125;
            assert!(
                (observed - expected).abs() <= expected.abs() * 1e-14,
                "pass={pass}, lane={i}: {observed} != {expected}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx906 and the matching native compiler/runtime"]
fn unaligned_views_copy_fill_and_reject_overlap() -> Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    assert_eq!(device.target().as_str(), "gfx906");
    let queue = device.queue()?;
    let root = device
        .fabric()
        .allocate_shared(256, std::slice::from_ref(&device))?;
    root.write(0, &[0xa5; 256])?;
    let source = root.view(1, 64)?;
    let destination = root.view(129, 64)?;
    // SAFETY: checked disjoint view ranges; the fill retires before copying.
    let fill = unsafe { queue.prepare_fill(&source, 0, 64, 0x37)?.dispatch() }?;
    assert!(fill.wait_timeout(Duration::from_secs(10))?);
    // SAFETY: initialized disjoint ranges, retained backing and ordered access.
    let copy = unsafe {
        queue
            .prepare_copy(&destination, 0, &source, 0, 64)?
            .dispatch()
    }?;
    assert!(copy.wait_timeout(Duration::from_secs(10))?);
    // These overlapping physical ranges have different view-relative offsets.
    let alias = root.view(9, 64)?;
    assert!(queue.prepare_copy(&alias, 0, &source, 16, 32).is_err());
    let mut actual = [0; 256];
    root.read(0, &mut actual)?;
    let mut expected = [0xa5; 256];
    expected[1..65].fill(0x37);
    expected[129..193].fill(0x37);
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
#[ignore = "requires gfx906 and the matching native compiler/runtime"]
fn recorded_batch_replays_pooled_immutable_arguments() -> Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    assert_eq!(device.target().as_str(), "gfx906");
    let compiler = Compiler::for_target(None, device.target())?;
    let source = CxxSource::new(
        "batch.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void add(const int* input, int* output, int bias) {
    output[threadIdx.x] = input[threadIdx.x] + bias;
}
"#,
    );
    let artifact = compiler
        .import_cxx(source)?
        .compile(&Specialization::new("add"))?;
    // SAFETY: authored scratch-free kernel has two 64-i32 buffers and one i32.
    let kernel = unsafe { device.load_bytes(artifact.bytes(), "add") }?;
    let queue = device.queue()?;
    let allocate = || {
        device
            .fabric()
            .allocate_shared(256, std::slice::from_ref(&device))
    };
    let input = allocate()?;
    let intermediate = allocate()?;
    let output = allocate()?;
    let mut recording = queue.record_batch()?;
    for (source, destination, bias, dependency) in [
        (&input, &intermediate, 113i32, false),
        (&intermediate, &output, -421i32, true),
    ] {
        // SAFETY: fixed complete extents. The dependency orders intermediate
        // reads after writes; each replay follows the previous retirement.
        unsafe {
            recording.dispatch(
                &kernel,
                [1, 1, 1],
                [64, 1, 1],
                &[
                    Argument::Buffer(source, 0),
                    Argument::Buffer(destination, 0),
                    Argument::Value(&bias.to_le_bytes()),
                ],
                dependency,
            )
        }?;
    }
    assert_eq!(recording.len(), 2);
    let batch = recording.finish()?;
    for pass in 0..16i32 {
        let values: Vec<i32> = (0..64).map(|i| i * 31 - pass * 17).collect();
        input.write(
            0,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // SAFETY: initialized input, ordered dependencies, retained full buffers.
        let done = unsafe { batch.dispatch() }?;
        assert!(done.wait_timeout(Duration::from_secs(10))?);
        let mut actual = [0; 256];
        output.read(0, &mut actual)?;
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|v| (v - 308).to_le_bytes())
            .collect();
        assert_eq!(actual.as_slice(), expected);
    }
    Ok(())
}
