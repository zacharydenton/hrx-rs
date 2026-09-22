//! Compile the bundled native XDNA copy kernel without opening a device.
//! Usage: compile_npu [packets], where each packet is 4096 bytes.
use hrx::{
    Result, Target,
    loom::{Compiler, ReportMode, Specialization},
};
fn main() -> Result<()> {
    let packets = std::env::args().nth(1).unwrap_or_else(|| "256".into());
    let artifact = Compiler::for_target(None, &Target::xdna())?
        .module(include_str!("../tests/kernels/copy.xdna.loom"))
        .compile(
            &Specialization::new("copy")
                .with_config("copy.packets", packets)
                .with_report(ReportMode::Summary),
        )?;
    println!("{}", artifact.path().display());
    Ok(())
}
