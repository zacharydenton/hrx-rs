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
fn manifest_targets_are_runtime_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = fixture(dir.path());
    for key in ["gfx1100", "gfx1151", "gfx90a", "gfx942"] {
        manifest.target = hrx::Target::new(key).unwrap().manifest_key();
        manifest.validate().unwrap();
        assert_eq!(manifest.gpu_target().unwrap().as_str(), key);
    }
    for key in ["gfx", "gfx11/../", "gfx1151\0", "GFX1151", "gfxzzzz"] {
        assert!(hrx::Target::new(key).is_err());
    }
    manifest.target = "aarch64-unknown-linux-gnu-gfx1151".into();
    assert!(manifest.validate().is_err());
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
    #[cfg(feature = "npu")]
    let npu = {
        let archive = dir.path().join("npu.tar.gz");
        let mut tar = tar::Builder::new(flate2::write::GzEncoder::new(
            fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        ));
        let bytes = b"test shim";
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        tar.append_data(&mut header, "libhrx_npu.so.1", &bytes[..])
            .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        let npu = hrx::npu::provision::Manifest {
            schema: 1,
            component: "npu-runtime".into(),
            revision: "test".into(),
            url: format!("file://{}", archive.display()),
            archive_sha256: hrx::bundle::file_digest(&archive).unwrap(),
            files: BTreeMap::from([("libhrx_npu.so.1".into(), digest(bytes))]),
        };
        fs::write(
            dir.path().join("npu.json"),
            serde_json::to_vec(&npu).unwrap(),
        )
        .unwrap();
        npu
    };
    // The cache location follows XDG, so isolate it by pointing XDG_CACHE_HOME
    // at a temporary root; hrx appends its own directory under that.
    let xdg = dir.path().join("xdg-cache");
    let cache = xdg.join("hrx");
    let mut children: Vec<_> = (0..4)
        .map(|_| {
            std::process::Command::new(env!("CARGO_BIN_EXE_hrx"))
                .arg("prepare")
                .env("XDG_CACHE_HOME", &xdg)
                .env("HRX_BUNDLE_MANIFEST", &manifest_path)
                .env("HRX_NPU_BUNDLE_MANIFEST", dir.path().join("npu.json"))
                .env_remove("HRX_OFFLINE")
                .env_remove("HRX_RUNTIME_DIR")
                .env_remove("HRX_NPU_RUNTIME_DIR")
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
    #[cfg(feature = "npu")]
    npu.verify(&cache.join("npu-runtime").join(&npu.archive_sha256))
        .unwrap();
    // The CLI must reuse both components without consulting their download URLs.
    fs::remove_file(dir.path().join("bundle.tar.gz")).unwrap();
    #[cfg(feature = "npu")]
    fs::remove_file(dir.path().join("npu.tar.gz")).unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_hrx"))
        .arg("prepare")
        .env("XDG_CACHE_HOME", &xdg)
        .env("HRX_BUNDLE_MANIFEST", &manifest_path)
        .env("HRX_NPU_BUNDLE_MANIFEST", dir.path().join("npu.json"))
        .env("HRX_OFFLINE", "1")
        .env_remove("HRX_RUNTIME_DIR")
        .env_remove("HRX_NPU_RUNTIME_DIR")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[cfg(all(feature = "runner", feature = "npu"))]
#[test]
fn prepare_reports_gpu_directory_when_npu_provisioning_fails() {
    for offline in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let gpu = fixture(dir.path());
        let npu = hrx::npu::provision::Manifest {
            schema: 1,
            component: "npu-runtime".into(),
            revision: "test".into(),
            url: format!("file://{}", dir.path().join("missing-npu.tar.gz").display()),
            archive_sha256: digest(b"unavailable archive"),
            files: BTreeMap::from([("libhrx_npu.so.1".into(), digest(b"test shim"))]),
        };
        let gpu_manifest = dir.path().join("gpu.json");
        let npu_manifest = dir.path().join("npu.json");
        fs::write(&gpu_manifest, serde_json::to_vec(&gpu).unwrap()).unwrap();
        fs::write(&npu_manifest, serde_json::to_vec(&npu).unwrap()).unwrap();
        let xdg = dir.path().join("xdg-cache");
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_hrx"));
        command
            .arg("prepare")
            .arg(dir.path().join("bundle.tar.gz"))
            .env("XDG_CACHE_HOME", &xdg)
            .env("HRX_BUNDLE_MANIFEST", &gpu_manifest)
            .env("HRX_NPU_BUNDLE_MANIFEST", &npu_manifest)
            .env_remove("HRX_RUNTIME_DIR")
            .env_remove("HRX_NPU_RUNTIME_DIR")
            .env_remove("HRX_OFFLINE");
        if offline {
            command.env("HRX_OFFLINE", "1");
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
        let destination = xdg.join("hrx/runtime").join(&gpu.archive_sha256);
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("{}\n", destination.display()),
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        gpu.verify(&destination).unwrap();
        assert!(
            !xdg.join("hrx/npu-runtime")
                .join(&npu.archive_sha256)
                .exists()
        );
    }
}
