//! Verify native GPU/XDNA access to one owned system-memory allocation.
use hrx::{
    Result,
    fabric::{Device, Engine, Fabric},
};
fn main() -> Result<()> {
    let fabric = Fabric::resolve()?;
    let gpu = Device::open(Engine::Gpu, 0)?;
    let npu = Device::open(Engine::Xdna, 0)?;
    let buffer = fabric.allocate(1 << 20, &[gpu.clone(), npu.clone()])?;
    println!(
        "GPU address={:#x}, XDNA DMA address={:#x}",
        buffer.device_address(&gpu)?,
        buffer.device_address(&npu)?
    );
    let fill = gpu.queue()?.prepare_fill(&buffer, 0, buffer.len(), 0x5a)?;
    unsafe { fill.dispatch() }?.wait()?;
    let mut bytes = vec![0; buffer.len()];
    buffer.read(0, &mut bytes)?;
    assert!(bytes.iter().all(|&byte| byte == 0x5a));
    println!("native shared allocation: {} bytes verified", bytes.len());
    Ok(())
}
