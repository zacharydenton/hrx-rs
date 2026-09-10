//! Compile a fixed passthrough design through the public Rust compiler API.
//! Usage: compile_npu <toolchain.json> <elements>
use hrx::{
    Result,
    execution::{Access, BindingContract, KernelContract},
    npu::compiler::{Compiler, CompilerOptions, Project, Source, Toolchain},
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err(hrx::Error::Message(
            "usage: compile_npu <toolchain.json> <elements>".into(),
        ));
    }
    let n: usize = args[2]
        .parse()
        .map_err(|e: std::num::ParseIntError| hrx::Error::Message(e.to_string()))?;
    let bytes = n
        .checked_mul(4)
        .ok_or_else(|| hrx::Error::Message("element count overflow".into()))?;
    let binding = |bytes, access| BindingContract {
        bytes,
        alignment: 4,
        access,
        layout: "u32".into(),
    };
    let compiler = Compiler::new(Toolchain::load(&args[1])?, CompilerOptions::new()?)?;
    let artifact = compiler.compile(&Project {
        root: std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/npu/fixtures"),
        source: Source::Iron("passthrough.py".into()),
        dependencies: vec![],
        arguments: vec!["-d".into(), "npu2".into(), "-n".into(), n.to_string()],
        cacheable: true,
        contract: KernelContract {
            bindings: vec![
                binding(bytes, Access::Read),
                binding(4096, Access::Read),
                binding(bytes, Access::Write),
            ],
            constants: vec![],
        },
    })?;
    println!("{}", artifact.path().display());
    Ok(())
}
