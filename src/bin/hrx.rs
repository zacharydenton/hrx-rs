//! Native bundle provisioning and Loom compilation CLI.
use hrx::{Error, Result};
#[path = "hrx/pack.rs"]
mod pack;
fn main() -> std::process::ExitCode {
    if let Err(e) = run() {
        eprintln!("{e}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("pack") if args.len() == 5 || args.len() == 6 => {
            pack::pack(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
                &args[3],
                &args[4],
                &args
                    .get(5)
                    .map(|key| hrx::Target::new(key))
                    .transpose()?
                    .unwrap_or_default(),
            )?;
        }
        Some("prepare") => {
            let manifest = hrx::bundle::default_manifest()?;
            let root = hrx::bundle::cache_root()?.join("runtime");
            let path = if let Some(archive) = args.get(1) {
                manifest.install(std::path::Path::new(archive), &root)?
            } else {
                hrx::bundle::resolve()?
            };
            println!("{}", path.display());
        }
        Some("info") => {
            let directory = hrx::bundle::resolve()?;
            let device = hrx::Device::open(0)?;
            println!(
                "runtime: {}\ntarget: {}",
                directory.display(),
                device.target().as_str()
            );
        }
        Some("compile") if args.len() >= 3 => {
            let source = std::fs::read_to_string(&args[1])?;
            let compiler = hrx::loom::Compiler::resolve(None)?;
            let module = compiler.module(&source);
            let mut request = hrx::loom::Specialization::new(&args[2]);
            for arg in &args[3..] {
                let (k, v) = arg
                    .split_once('=')
                    .ok_or_else(|| Error::Message("config must be key=value".into()))?;
                request.config.insert(k.into(), v.into());
            }
            println!(
                "{}",
                module
                    .compile(&request, &hrx::bundle::cache_root()?.join("kernels"))?
                    .path()
                    .display()
            );
        }
        _ => {
            return Err(Error::Message(
                "usage: hrx pack RUNTIME OUTPUT URL REVISION [TARGET] | prepare [bundle.tar.gz] | info | compile SOURCE SYMBOL [key=value ...]"
                    .into(),
            ));
        }
    }
    Ok(())
}
