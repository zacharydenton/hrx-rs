//! Fresh-process startup and compilation costs. Use an empty XDG_CACHE_HOME
//! for each invocation; this does not flush the operating system's page cache.
//! Pass --compiler-only to measure CPU compilation without opening a GPU.
use hrx::{Result, Stream, loom::Specialization};
use serde_json::{Value, json};
use std::time::Instant;

fn memory() -> Result<Value> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kib = |key: &str| -> Result<u64> {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| hrx::Error::Message(format!("missing {key} in process status")))
    };
    Ok(json!({"rss_kib": kib("VmRSS:")?, "peak_rss_kib": kib("VmHWM:")?}))
}

fn main() -> Result<()> {
    let compiler_only = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--compiler-only") => true,
        Some(_) => {
            return Err(hrx::Error::Message(
                "usage: startup_bench [--compiler-only]".into(),
            ));
        }
    };
    let initial = memory()?;
    let start = Instant::now();
    let stream = if compiler_only {
        None
    } else {
        Some(Stream::open()?)
    };
    let open_ms = stream
        .as_ref()
        .map(|_| start.elapsed().as_secs_f64() * 1000.0);
    let after_open = if compiler_only { None } else { Some(memory()?) };
    let start = Instant::now();
    let compiler = match &stream {
        Some(stream) => hrx::loom::Compiler::for_stream(None, stream)?,
        None => hrx::loom::Compiler::resolve(None)?,
    };
    let compiler_ms = start.elapsed().as_secs_f64() * 1000.0;
    let after_compiler = memory()?;
    let module = compiler.module(include_str!("../tests/kernels/euler.loom"));
    let mut spec = Specialization::new("krea2_euler");
    spec.set_config("krea2.euler.grid_x", "1");
    spec.set_config("krea2.euler.grid_y", "1");
    let start = Instant::now();
    let artifact = module.compile(&spec)?;
    let first_compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    let after_compile = memory()?;
    let start = Instant::now();
    let cached = module.compile(&spec)?;
    let cache_hit_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(artifact.bytes(), cached.bytes());
    println!(
        "{}",
        json!({"compiler_only": compiler_only, "open_ms": open_ms, "compiler_ms": compiler_ms,
            "first_compile_ms": first_compile_ms, "cache_hit_ms": cache_hit_ms,
            "initial": initial, "after_open": after_open,
            "after_compiler": after_compiler, "after_compile": after_compile,
            "compiler_identity": artifact.compiler_identity()})
    );
    Ok(())
}
