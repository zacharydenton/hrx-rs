use hrx::{Error, Result};
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
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
            let _device = hrx::Device::open(0)?;
            println!(
                "runtime: {}\ntarget: {}",
                directory.display(),
                hrx::TARGET_KEY
            );
        }
        Some("compile") if args.len() >= 3 => {
            let source = std::fs::read_to_string(&args[1])?;
            let compiler = hrx::loom::Compiler::resolve(None)?;
            let mut request = hrx::loom::Request::new(&source, &args[2]);
            for arg in &args[3..] {
                let (k, v) = arg
                    .split_once('=')
                    .ok_or_else(|| Error("config must be key=value".into()))?;
                request.config.insert(k.into(), v.into());
            }
            println!(
                "{}",
                compiler
                    .compile(&request, &hrx::bundle::cache_root()?.join("kernels"))?
                    .display()
            );
        }
        _ => {
            return Err(Error(
                "usage: hrx prepare [bundle.tar.gz] | info | compile SOURCE SYMBOL [key=value ...]"
                    .into(),
            ));
        }
    }
    Ok(())
}
