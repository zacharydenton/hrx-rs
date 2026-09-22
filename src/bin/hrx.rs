//! Native bundle provisioning and Loom compilation CLI.
use hrx::{Error, Result};
#[path = "hrx/compile.rs"]
mod compile;
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
        Some("doctor") => {
            for path in ["/dev/kfd", "/dev/dri/renderD128", "/dev/accel/accel0"] {
                let accessible = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path);
                println!(
                    "{path}: {}",
                    match accessible {
                        Ok(_) => "accessible".into(),
                        Err(error) => error.to_string(),
                    }
                );
            }
            let manifest = hrx::bundle::default_manifest()?;
            let cached = hrx::bundle::cache_root()?
                .join("runtime")
                .join(&manifest.archive_sha256);
            if std::env::var_os("HRX_RUNTIME_DIR").is_some()
                || std::env::var_os("HRX_AMDF_LIBRARY").is_some()
                || manifest.verify(&cached).is_ok()
            {
                match hrx::fabric::Fabric::resolve().and_then(|fabric| fabric.endpoints()) {
                    Ok(endpoints) => {
                        for endpoint in endpoints {
                            println!(
                                "{:?}: {} ({})",
                                endpoint.engine(),
                                endpoint.name(),
                                endpoint.target().as_str()
                            );
                        }
                    }
                    Err(error) => println!("native device discovery: {error}"),
                }
                let directory = std::env::var_os("HRX_RUNTIME_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or(cached);
                let library = std::env::var_os("HRX_LOOM_LIBRARY")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| directory.join("libloomc.so"));
                match hrx::loom::Compiler::resolve(Some(&library)) {
                    Ok(_) => println!("Loom compiler: loaded"),
                    Err(error) => println!("Loom compiler probe failed: {error}"),
                }
            } else {
                println!("Native runtime: not prepared (doctor does not download)");
            }
            println!(
                "native bundle: {}",
                hrx::bundle::default_manifest()?.revision
            );
            println!(
                "Native override: {}",
                std::env::var("HRX_RUNTIME_DIR").unwrap_or_else(|_| "none".into())
            );
        }
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
        Some("prepare") if args.len() <= 2 => {
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
        Some("compile") if args.len() >= 3 => compile::compile(&args[1..])?,
        Some("report") => compile::report(&args[1..])?,
        _ => {
            return Err(Error::Message(
                "usage: hrx run --hsaco FILE --kernel NAME [...] | pack RUNTIME OUTPUT URL REVISION [TARGET] | prepare [native.tar.gz] | doctor | gc [DAYS] | info | compile SOURCE SYMBOL [key=value ...] | report show FILE | report diff BEFORE AFTER"
                    .into(),
            ));
        }
    }
    Ok(())
}
