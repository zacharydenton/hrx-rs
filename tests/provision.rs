use hrx::bundle::{Manifest, digest, prepare};
use std::{collections::BTreeMap, fs, io::Write};
fn fixture(dir: &std::path::Path) -> Manifest {
    let names = ["loom-compile", "libhrx.so", "libhsa-runtime64.so.1"];
    let archive = dir.join("bundle.tar.gz");
    let encoder = flate2::write::GzEncoder::new(
        fs::File::create(&archive).unwrap(),
        flate2::Compression::default(),
    );
    let mut tar = tar::Builder::new(encoder);
    let mut files = BTreeMap::new();
    for name in names {
        let bytes = name.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        tar.append_data(&mut header, name, bytes).unwrap();
        files.insert(name.into(), digest(bytes));
    }
    tar.into_inner().unwrap().finish().unwrap().flush().unwrap();
    Manifest {
        schema: 1,
        target: "x86_64-unknown-linux-gnu-gfx1151".into(),
        revision: "test".into(),
        url: format!("file://{}", archive.display()),
        archive_sha256: hrx::bundle::file_digest(&archive).unwrap(),
        files,
    }
}
#[test]
fn installation_is_atomic_verified_and_offline_reusable() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = fixture(dir.path());
    let cache = dir.path().join("cache");
    assert!(prepare(&manifest, &cache, true).is_err());
    let destination = prepare(&manifest, &cache, false).unwrap();
    assert_eq!(prepare(&manifest, &cache, true).unwrap(), destination);
    fs::write(destination.join("libhrx.so"), "corrupt").unwrap();
    assert!(prepare(&manifest, &cache, true).is_err());
    prepare(&manifest, &cache, false).unwrap();
    manifest.verify(&destination).unwrap();
    fs::remove_dir_all(&destination).unwrap();
    fs::write(dir.path().join("bundle.tar.gz"), "truncated").unwrap();
    assert!(prepare(&manifest, &cache, false).is_err());
    assert!(!destination.exists());
}
#[test]
fn simultaneous_installers_share_one_complete_bundle() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = fixture(dir.path());
    let cache = dir.path().join("cache");
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|_| scope.spawn(|| prepare(&manifest, &cache, false).unwrap()))
            .collect();
        for worker in workers {
            manifest.verify(&worker.join().unwrap()).unwrap();
        }
    });
}
#[test]
fn invalid_manifest_paths_and_missing_components_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = fixture(dir.path());
    manifest.files.insert("../evil".into(), digest(b"evil"));
    assert!(Manifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
    manifest.files.remove("../evil");
    manifest.files.remove("loom-compile");
    assert!(Manifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
}
#[cfg(feature = "loom")]
#[test]
fn compiler_cache_includes_content_and_repairs_corrupt_outputs() {
    use hrx::loom::{Compiler, Request};
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let executable = dir.path().join("compiler");
    fs::write(&executable, "#!/bin/sh\nfor arg in \"$@\"; do case \"$arg\" in --output=*) printf artifact > \"${arg#--output=}\";; esac; done\n").unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    let compiler = Compiler::resolve(Some(&executable)).unwrap();
    let source = Request::new("source", "kernel");
    let cache = dir.path().join("cache");
    let first = compiler.compile(&source, &cache).unwrap();
    fs::write(&first, "corrupt").unwrap();
    let repaired = compiler.compile(&source, &cache).unwrap();
    assert_eq!(fs::read(repaired).unwrap(), b"artifact");
    assert_ne!(
        compiler.key(&source).unwrap(),
        compiler.key(&Request::new("changed", "kernel")).unwrap()
    );
    let mut changed = Request::new("source", "kernel");
    changed.config.insert("k.value".into(), "one\ntwo".into());
    assert_ne!(
        compiler.key(&source).unwrap(),
        compiler.key(&changed).unwrap()
    );
    let original = fs::read(&executable).unwrap();
    fs::write(
        &executable,
        String::from_utf8(original)
            .unwrap()
            .replace("artifact", "ARTIFACT"),
    )
    .unwrap();
    let new_compiler = Compiler::resolve(Some(&executable)).unwrap();
    assert_ne!(
        compiler.key(&source).unwrap(),
        new_compiler.key(&source).unwrap()
    );
    assert!(
        compiler
            .compile(&Request::new("uncached", "kernel"), &cache)
            .is_err()
    );
}

#[cfg(feature = "runner")]
#[test]
fn separate_processes_provision_the_same_cache() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = fixture(dir.path());
    let manifest_path = dir.path().join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let cache = dir.path().join("shared-cache");
    let mut children: Vec<_> = (0..4)
        .map(|_| {
            std::process::Command::new(env!("CARGO_BIN_EXE_hrx"))
                .arg("prepare")
                .env("HRX_CACHE_DIR", &cache)
                .env("HRX_BUNDLE_MANIFEST", &manifest_path)
                .env_remove("HRX_OFFLINE")
                .env_remove("HRX_RUNTIME_DIR")
                .env_remove("KREA2_RUNTIME")
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap()
        })
        .collect();
    for child in &mut children {
        assert!(child.wait().unwrap().success());
    }
    manifest
        .verify(&cache.join("runtime").join(manifest.archive_sha256.clone()))
        .unwrap();
}
