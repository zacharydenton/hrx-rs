#![cfg(feature = "loom")]
//! CPU-only native compiler tests. Set HRX_LOOM_LIBRARY and run --ignored.
use hrx::loom::{Compiler, CompilerOptions, Specialization};
use std::{fs, num::NonZeroUsize};
const SOURCE: &str = include_str!("kernels/euler.loom");
fn spec() -> Specialization {
    let mut s = Specialization::new("krea2_euler");
    s.config.insert("krea2.euler.grid_x".into(), "1".into());
    s.config.insert("krea2.euler.grid_y".into(), "1".into());
    s
}
#[test]
fn missing_library_is_an_io_error() {
    assert!(Compiler::resolve(Some(std::path::Path::new("/nonexistent/hrx/libloomc.so"))).is_err());
}
#[test]
#[ignore = "requires public libloomc.so, no GPU"]
fn index_reuse_cache_repair_and_diagnostics() -> hrx::Result<()> {
    let c = Compiler::resolve(None)?;
    let m = c.module(SOURCE);
    let cache = tempfile::tempdir()?;
    let mut request = spec();
    request.report = true;
    let a = m.compile(&request, cache.path())?;
    assert!(a.bytes().starts_with(b"\x7fELF"));
    assert!(a.report().is_some());
    assert_eq!(a.bytes(), m.compile(&request, cache.path())?.bytes());
    fs::write(a.path(), "corrupt")?;
    assert_eq!(a.bytes(), m.compile(&request, cache.path())?.bytes());
    fs::write(a.path().with_file_name("artifact.json"), "broken")?;
    assert_eq!(a.bytes(), m.compile(&request, cache.path())?.bytes());
    assert_ne!(
        m.key(&request)?,
        c.module(&format!("{SOURCE}\n")).key(&request)?
    );
    let mut other = request.clone();
    other.config.insert("krea2.euler.grid_x".into(), "2".into());
    assert_ne!(m.key(&request)?, m.key(&other)?);
    assert_ne!(a.path(), m.compile(&other, cache.path())?.path());
    let mut invalid = request.clone();
    invalid
        .config
        .insert("krea2.euler.grid_x".into(), "invalid".into());
    let e = m.compile(&invalid, cache.path()).unwrap_err().to_string();
    assert!(e.contains("Loom compilation failed"), "{e}");
    assert!(
        m.compile(&Specialization::new("missing"), cache.path())
            .is_err()
    );
    assert!(
        c.module("not Loom")
            .compile(&request, cache.path())
            .is_err()
    );
    assert_eq!(a.bytes(), m.compile(&request, cache.path())?.bytes());
    c.trim();
    assert_eq!(a.bytes(), m.compile(&request, cache.path())?.bytes());
    drop(m);
    drop(c);
    assert!(a.bytes().starts_with(b"\x7fELF"));
    Ok(())
}
#[test]
#[ignore = "requires public libloomc.so, no GPU"]
fn concurrent_specializations_use_exclusive_workspaces() -> hrx::Result<()> {
    let c = Compiler::with_options(
        None,
        CompilerOptions {
            workers: NonZeroUsize::new(2).unwrap(),
            ..CompilerOptions::default()
        },
    )?;
    let m = c.module(SOURCE);
    let cache = tempfile::tempdir()?;
    std::thread::scope(|scope| {
        let tasks: Vec<_> = (0..8)
            .map(|i| {
                let m = &m;
                let cache = cache.path();
                scope.spawn(move || {
                    let mut s = spec();
                    s.config
                        .insert("krea2.euler.grid_x".into(), (i % 4 + 1).to_string());
                    let a = m.compile(&s, cache).unwrap();
                    assert_eq!(a.bytes(), m.compile(&s, cache).unwrap().bytes());
                })
            })
            .collect();
        for t in tasks {
            t.join().unwrap();
        }
    });
    c.trim();
    Ok(())
}

#[test]
#[ignore = "requires public libloomc.so, no GPU"]
fn replaced_library_cannot_relabel_a_resident_compiler() -> hrx::Result<()> {
    let original = Compiler::resolve(None)?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("libloomc.so");
    fs::copy(original.library_path(), &path)?;
    let loaded = Compiler::resolve(Some(&path))?;
    assert_eq!(loaded.identity(), original.identity());
    // Change the digest without changing the ELF code. Replace atomically: never
    // write into an inode that the dynamic loader has already mapped.
    let mut bytes = fs::read(&path)?;
    bytes.push(0);
    let replacement = dir.path().join("replacement.so");
    fs::write(&replacement, &bytes)?;
    fs::rename(&replacement, &path)?;
    let error = Compiler::resolve(Some(&path)).unwrap_err().to_string();
    assert!(error.contains("changed after loading"), "{error}");
    let cache = tempfile::tempdir()?;
    assert!(
        loaded
            .module(SOURCE)
            .compile(&spec(), cache.path())?
            .bytes()
            .starts_with(b"\x7fELF")
    );
    drop(loaded);
    assert!(Compiler::resolve(Some(&path)).is_err());
    let new_path = dir.path().join("new-version.so");
    fs::write(&new_path, bytes)?;
    let upgraded = Compiler::resolve(Some(&new_path))?;
    assert_ne!(original.identity(), upgraded.identity());
    assert!(
        upgraded
            .module(SOURCE)
            .compile(&spec(), cache.path())?
            .bytes()
            .starts_with(b"\x7fELF")
    );
    Ok(())
}
