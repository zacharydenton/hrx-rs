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
            if std::env::var_os("HRX_RUNTIME_DIR").is_some() || manifest.verify(&cached).is_ok() {
                match hrx::gpu::Device::open(0) {
                    Ok(device) => println!(
                        "GPU target: {}; shared interop ABI 1: {}",
                        device.target().as_str(),
                        match device.supports_shared_interop() {
                            Ok(supported) => supported.to_string(),
                            Err(error) => format!("probe failed: {error}"),
                        }
                    ),
                    Err(error) => println!("GPU initialization: {error}"),
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
                println!("GPU runtime: not prepared (doctor does not download)");
            }
            println!("gpu bundle: {}", hrx::bundle::default_manifest()?.revision);
            println!(
                "GPU override: {}",
                std::env::var("HRX_RUNTIME_DIR").unwrap_or_else(|_| "none".into())
            );
            #[cfg(feature = "npu")]
            {
                let probe = || -> Result<()> {
                    let manifest = hrx::npu::provision::runtime_manifest()?;
                    println!("NPU bundle: {}", manifest.revision);
                    let directory = match std::env::var_os("HRX_NPU_RUNTIME_DIR") {
                        Some(path) => path.into(),
                        None => {
                            let cached = hrx::bundle::cache_root()?
                                .join(&manifest.component)
                                .join(&manifest.archive_sha256);
                            if let Err(error) = manifest.verify(&cached) {
                                println!(
                                    "NPU runtime: not prepared or corrupt ({error}); run hrx prepare"
                                );
                                return Ok(());
                            }
                            cached
                        }
                    };
                    hrx::npu::provision::probe_runtime(&directory)?;
                    println!(
                        "NPU runtime ABI 1: loaded from {} (device access checked above)",
                        directory.display()
                    );
                    Ok(())
                };
                if let Err(error) = probe() {
                    println!("NPU runtime probe failed: {error}");
                }
                println!(
                    "NPU override: {}",
                    std::env::var("HRX_NPU_RUNTIME_DIR").unwrap_or_else(|_| "none".into())
                );
            }
            #[cfg(not(feature = "npu"))]
            println!("NPU support: not compiled; install with --features runner,npu");
            println!(
                "NPU kernel compilation uses an explicit toolchain manifest; doctor does not install drivers or compilers."
            );
        }
        #[cfg(feature = "npu")]
        Some("prepare-npu") if args.len() <= 3 => {
            let manifest = match args.get(1) {
                Some(path) => hrx::npu::provision::Manifest::load(path)?,
                None => {
                    println!("{}", hrx::npu::provision::resolve()?.display());
                    return Ok(());
                }
            };
            let directory = if let Some(archive) = args.get(2) {
                manifest.install(
                    std::path::Path::new(archive),
                    &hrx::bundle::cache_root()?.join(&manifest.component),
                )?
            } else {
                manifest.prepare(std::env::var_os("HRX_OFFLINE").is_some())?
            };
            println!("{}", directory.display());
        }
        #[cfg(feature = "npu-compile")]
        Some("compile-npu") if args.len() == 3 => {
            use hrx::npu::compiler::{Compiler, CompilerOptions, Project, Toolchain};
            let project: Project = serde_json::from_slice(&std::fs::read(&args[2])?)?;
            let compiler = Compiler::new(Toolchain::load(&args[1])?, CompilerOptions::new()?)?;
            println!("{}", compiler.compile(&project)?.path().display());
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
        Some("prepare") if args.len() <= 3 => {
            let manifest = hrx::bundle::default_manifest()?;
            let root = hrx::bundle::cache_root()?.join("runtime");
            let path = if let Some(archive) = args.get(1) {
                manifest.install(std::path::Path::new(archive), &root)?
            } else {
                hrx::bundle::resolve()?
            };
            // Report the usable GPU directory even if NPU provisioning fails.
            println!("{}", path.display());
            std::io::Write::flush(&mut std::io::stdout())?;
            #[cfg(feature = "npu")]
            {
                let npu = if let Some(archive) = args.get(2) {
                    let manifest = hrx::npu::provision::runtime_manifest()?;
                    manifest.install(
                        std::path::Path::new(archive),
                        &hrx::bundle::cache_root()?.join(&manifest.component),
                    )?
                } else {
                    hrx::npu::provision::resolve()?
                };
                eprintln!("NPU runtime: {}", npu.display());
            }
            #[cfg(not(feature = "npu"))]
            if args.len() == 3 {
                return Err(Error::Unsupported(
                    "NPU archive requires the npu feature".into(),
                ));
            }
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
                "usage: hrx run --hsaco FILE --kernel NAME [...] | pack RUNTIME OUTPUT URL REVISION [TARGET] | prepare [gpu.tar.gz [npu.tar.gz]] | prepare-npu [MANIFEST [ARCHIVE]] | doctor | gc [DAYS] | info | compile SOURCE SYMBOL [key=value ...] | compile-npu TOOLCHAIN PROJECT"
                    .into(),
            ));
        }
    }
    Ok(())
}
