//! Qualify gfx906 native allocation and scratch-free dispatch independently of
//! Loom bringup. Accepts a trusted probe HSACO, writing 42 into 64 u32 lanes.
use hrx::{
    Error, Result,
    fabric::{Argument, Device, Engine, Fabric},
};

fn main() -> Result<()> {
    let fabric = Fabric::resolve()?;
    for endpoint in fabric.endpoints()? {
        println!(
            "endpoint: {} ({})",
            endpoint.name(),
            endpoint.target().as_str()
        );
    }
    let gpu = Device::open(Engine::Gpu, 0)?;
    if gpu.target().as_str() != "gfx906" {
        return Err(Error::Message(
            "qualification requires a Radeon VII gfx906".into(),
        ));
    }
    println!("device opened");
    let output = fabric.allocate_shared(64 * 4, std::slice::from_ref(&gpu))?;
    output.write(0, &[0; 64 * 4])?;
    let mut readback = [1; 64 * 4];
    output.read(0, &mut readback)?;
    assert_eq!(readback, [0; 64 * 4]);
    println!("host-visible allocation and CPU access passed");
    let Some(path) = std::env::args_os().nth(1) else {
        return Ok(());
    };
    let bytes = std::fs::read(path)?;
    // SAFETY: this maintainer qualification takes an explicitly supplied,
    // trusted probe. Its sole binding is the 64-u32 output allocation.
    let kernel = unsafe { gpu.load_bytes(&bytes, "probe") }?;
    assert_eq!(kernel.subgroup_size(), 64);
    let queue = gpu.queue()?;
    for _ in 0..4 {
        output.write(0, &[0; 64 * 4])?;
        // SAFETY: the probe writes one u32 per lane of the 64-lane workgroup.
        // The allocation and loaded kernel remain alive through completion.
        unsafe {
            queue.dispatch(
                &kernel,
                [1, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&output, 0)],
            )
        }?
        .wait()?;
        output.read(0, &mut readback)?;
        for word in readback.as_chunks::<4>().0 {
            assert_eq!(u32::from_le_bytes(*word), 42);
        }
    }
    println!("four repeated wave64 dispatches and readback passed");
    let input = fabric.allocate_shared(64 * 4, std::slice::from_ref(&gpu))?;
    // SAFETY: the same trusted qualification object contains transform with
    // two 64-u32 bindings and one inline i32; the source is checked in.
    let transform = unsafe { gpu.load_bytes(&bytes, "transform") }?;
    for pass in 0..64_i32 {
        let mut values = [0; 64 * 4];
        for (index, word) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            word.copy_from_slice(&(index as i32 - pass).to_le_bytes());
        }
        input.write(0, &values)?;
        output.write(0, &[0; 64 * 4])?;
        let bias = (pass * 7).to_le_bytes();
        // SAFETY: all bindings match transform's declared ABI and extent.
        // CPU writes precede submission; readback follows terminal completion.
        unsafe {
            queue.dispatch(
                &transform,
                [1, 1, 1],
                [64, 1, 1],
                &[
                    Argument::Buffer(&input, 0),
                    Argument::Buffer(&output, 0),
                    Argument::Value(&bias),
                ],
            )
        }?
        .wait()?;
        output.read(0, &mut readback)?;
        for (index, word) in readback.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                i32::from_le_bytes(*word),
                (index as i32 - pass) * 3 + pass * 7
            );
        }
    }
    println!("64 changed-input, inline-argument, and cache-visibility passes succeeded");
    Ok(())
}
