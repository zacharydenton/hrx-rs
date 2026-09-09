//! Native bundle provisioning and Loom compilation CLI.
use hrx::{Error, Result};
#[path = "hrx/pack.rs"]
mod pack;
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The launcher owns its own exit codes: usage errors exit 64, not 1.
    if args.first().is_some_and(|a| a == "run") {
        return hrx::runner::run(&args[1..]);
    }
    if let Err(e) = dispatch(&args) {
        eprintln!("{e}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
fn dispatch(args: &[String]) -> Result<()> {
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
        Some("gc") => {
            let days: u64 = match args.get(1) {
                Some(value) => value
                    .parse()
                    .map_err(|_| Error::Message("gc takes a day count".into()))?,
                None => 30,
            };
            let cache = hrx::bundle::cache_root()?;
            let manifest = hrx::bundle::default_manifest()?;
            let reclaimed = hrx::bundle::collect(
                &cache,
                &manifest.archive_sha256,
                std::time::Duration::from_secs(days * 24 * 60 * 60),
            )?;
            println!(
                "removed {} superseded runtime bundle(s) ({:.1} MB)\nremoved {} kernel artifact(s) unread for >{days}d ({:.1} MB)\nkept {} (pinned by bundle.json)",
                reclaimed.bundles,
                reclaimed.bundle_bytes as f64 / 1e6,
                reclaimed.artifacts,
                reclaimed.artifact_bytes as f64 / 1e6,
                &manifest.archive_sha256[..8],
            );
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
            println!("{}", module.compile(&request)?.path().display());
        }
        _ => {
            return Err(Error::Message(
                "usage: hrx run --hsaco FILE --kernel NAME [...] | pack RUNTIME OUTPUT URL REVISION [TARGET] | prepare [bundle.tar.gz] | gc [DAYS] | info | compile SOURCE SYMBOL [key=value ...]"
                    .into(),
            ));
        }
    }
    Ok(())
}
