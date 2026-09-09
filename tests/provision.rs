//! Provisioning integrity, path validation and compiler cache tests.
use hrx::bundle::{Manifest, digest, prepare};
use std::{collections::BTreeMap, fs, io::Write};
fn fixture(dir: &std::path::Path) -> Manifest {
    let names = ["libloomc.so", "libhrx.so", "libhsa-runtime64.so.1"];
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
    manifest.files.remove("libloomc.so");
    assert!(Manifest::parse(&serde_json::to_vec(&manifest).unwrap()).is_err());
}

#[test]
fn unchecked_manifests_are_rejected_before_any_installation_io() {
    let dir = tempfile::tempdir().unwrap();
    let valid = fixture(dir.path());
    for name in ["../escaped", "/tmp/escaped", "nested/file", ".", ".."] {
        let mut manifest = valid.clone();
        manifest.files.insert(name.into(), digest(b"escaped"));
        // Derived Deserialize intentionally remains available; install must
        // enforce the same invariant as parse even for this construction path.
        let manifest: Manifest =
            serde_json::from_slice(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        let root = dir.path().join("uncreated");
        assert!(
            manifest
                .install(&dir.path().join("bundle.tar.gz"), &root)
                .is_err()
        );
        assert!(prepare(&manifest, &root, false).is_err());
        assert!(manifest.verify(dir.path()).is_err());
        assert!(!root.exists());
        assert!(!dir.path().join("escaped").exists());
    }
    let mut manifest = valid;
    manifest.archive_sha256 = "../escaped".into();
    assert!(prepare(&manifest, &dir.path().join("uncreated"), false).is_err());
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
