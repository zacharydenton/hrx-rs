//! Verified, separately provisioned NPU components. No Cargo build-time setup.
use crate::{Error, Result, bundle};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};
/// Contents and source identity of an NPU native runtime or compiler component.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Format version, currently 1.
    pub schema: u32,
    /// Component name: npu-runtime or npu-compiler.
    pub component: String,
    /// Pinned source revision and build toolchain description.
    pub revision: String,
    /// HTTPS download or local file URL.
    pub url: String,
    /// SHA-256 of the gzip-compressed tar archive.
    pub archive_sha256: String,
    /// Relative regular files and their SHA-256 digests; links are forbidden.
    pub files: BTreeMap<PathBuf, String>,
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// The runtime shipped with this crate, including its matching XRT libraries.
pub fn default_manifest() -> Result<Manifest> {
    let manifest: Manifest = serde_json::from_str(include_str!("../../npu-bundle.json"))?;
    manifest.validate()?;
    Ok(manifest)
}

/// Select the pinned runtime manifest, respecting `HRX_NPU_BUNDLE_MANIFEST`.
pub fn runtime_manifest() -> Result<Manifest> {
    let manifest = match std::env::var_os("HRX_NPU_BUNDLE_MANIFEST") {
        Some(path) => Manifest::load(path)?,
        None => default_manifest()?,
    };
    if manifest.component != "npu-runtime" {
        return Err(Error::Message("expected an npu-runtime manifest".into()));
    }
    Ok(manifest)
}

/// Resolve the runtime override or prepare the pinned user-space runtime.
/// `HRX_OFFLINE` prevents downloads. No drivers or compiler tools are installed.
pub fn resolve() -> Result<PathBuf> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(Error::Unsupported(
            "the NPU runtime supports Linux x86_64".into(),
        ));
    }
    if let Some(path) = std::env::var_os("HRX_NPU_RUNTIME_DIR") {
        return Ok(PathBuf::from(path));
    }
    runtime_manifest()?.prepare(std::env::var_os("HRX_OFFLINE").is_some())
}

/// Load an installed native runtime and check its ABI without downloading files
/// or loading a program. This checks the shim and linked dependencies, not device access.
pub fn probe_runtime(directory: impl AsRef<Path>) -> Result<()> {
    super::load_shim_from(directory.as_ref()).map(|_| ())
}

impl Manifest {
    /// Read and structurally validate a component manifest.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(&fs::read(path)?)?;
        manifest.validate()?;
        Ok(manifest)
    }
    fn validate(&self) -> Result<()> {
        if self.schema != 1
            || !matches!(self.component.as_str(), "npu-runtime" | "npu-compiler")
            || !hash(&self.archive_sha256)
            || self.files.is_empty()
        {
            return Err(Error::Message("invalid NPU component manifest".into()));
        }
        for (path, digest) in &self.files {
            if path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_)))
                || !hash(digest)
            {
                return Err(Error::Message("invalid component file or digest".into()));
            }
        }
        if self.component == "npu-runtime" && !self.files.contains_key(Path::new("libhrx_npu.so.1"))
        {
            return Err(Error::Message("NPU runtime lacks ABI shim".into()));
        }
        Ok(())
    }
    /// Verify all regular files in an installed component.
    pub fn verify(&self, root: &Path) -> Result<()> {
        self.validate()?;
        for (path, expected) in &self.files {
            let mut parent = root.to_path_buf();
            for component in path.components() {
                parent.push(component);
                if fs::symlink_metadata(&parent)?.file_type().is_symlink() {
                    return Err(Error::Message("component contains a symbolic link".into()));
                }
            }
            if !fs::metadata(root.join(path))?.is_file()
                || bundle::file_digest(&root.join(path))? != *expected
            {
                return Err(Error::Message(format!(
                    "corrupt component file {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
    /// Install a verified archive atomically under a caller-selected cache root.
    pub fn install(&self, archive: &Path, cache: &Path) -> Result<PathBuf> {
        self.validate()?;
        bundle::create_cache_dir(cache)?;
        let _lock = bundle::Lock::acquire(&cache.join(format!("{}.lock", self.archive_sha256)))?;
        let destination = cache.join(&self.archive_sha256);
        if self.verify(&destination).is_ok() {
            return Ok(destination);
        }
        if bundle::file_digest(archive)? != self.archive_sha256 {
            return Err(Error::Message("NPU archive digest mismatch".into()));
        }
        let staging = tempfile::tempdir_in(cache)?;
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(fs::File::open(archive)?));
        let mut seen = BTreeSet::new();
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.into_owned();
            if !entry.header().entry_type().is_file()
                || !self.files.contains_key(&path)
                || !seen.insert(path.clone())
            {
                return Err(Error::Message(
                    "unexpected, duplicate or nonregular component archive entry".into(),
                ));
            }
            let destination = staging.path().join(&path);
            fs::create_dir_all(destination.parent().unwrap())?;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(destination)?;
            std::io::copy(&mut entry, &mut file)?;
            file.flush()?;
            use std::os::unix::fs::PermissionsExt;
            // Preserve executable bits; never preserve setuid or special modes.
            file.set_permissions(fs::Permissions::from_mode(entry.header().mode()? & 0o777))?;
        }
        self.verify(staging.path())?;
        fs::write(
            staging.path().join("component.json"),
            serde_json::to_vec_pretty(self)?,
        )?;
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        bundle::publish(staging, &destination)?;
        Ok(destination)
    }
    /// Reuse cached files or download a pinned component. Offline mode never downloads.
    pub fn prepare(&self, offline: bool) -> Result<PathBuf> {
        self.validate()?;
        let root = bundle::cache_root()?.join(&self.component);
        let destination = root.join(&self.archive_sha256);
        if self.verify(&destination).is_ok() {
            return Ok(destination);
        }
        if offline {
            return Err(Error::Message(format!(
                "{} is not prepared for offline use",
                self.component
            )));
        }
        bundle::create_cache_dir(&root)?;
        let mut archive = tempfile::NamedTempFile::new_in(&root)?;
        if let Some(path) = self.url.strip_prefix("file://") {
            std::io::copy(&mut fs::File::open(path)?, &mut archive)?;
        } else {
            if !self.url.starts_with("https://") {
                return Err(Error::Message(
                    "NPU component URL must use HTTPS or file://".into(),
                ));
            }
            #[cfg(feature = "download")]
            {
                let mut response = ureq::get(&self.url)
                    .call()
                    .map_err(|e| Error::Download(Box::new(e)))?;
                std::io::copy(&mut response.body_mut().as_reader(), &mut archive)?;
            }
            #[cfg(not(feature = "download"))]
            {
                return Err(Error::Unsupported(
                    "enable download or supply a local component archive".into(),
                ));
            }
        }
        archive.flush()?;
        self.install(archive.path(), &root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn archive(root: &Path) -> (Manifest, PathBuf) {
        let archive = root.join("component.tar.gz");
        let gzip = flate2::write::GzEncoder::new(
            fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(gzip);
        let contents = b"test native ABI shim";
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        tar.append_data(&mut header, "libhrx_npu.so.1", &contents[..])
            .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        (
            Manifest {
                schema: 1,
                component: "npu-runtime".into(),
                revision: "fixture".into(),
                url: format!("file://{}", archive.display()),
                archive_sha256: bundle::file_digest(&archive).unwrap(),
                files: BTreeMap::from([("libhrx_npu.so.1".into(), bundle::digest(contents))]),
            },
            archive,
        )
    }
    #[test]
    fn pinned_runtime_includes_its_xrt_and_driver() {
        let manifest = default_manifest().unwrap();
        assert_eq!(manifest.component, "npu-runtime");
        assert!(manifest.url.starts_with("https://"));
        for path in [
            "libhrx_npu.so.1",
            "lib/libxrt_coreutil.so.2",
            "lib/libxrt_core.so.2",
            "lib/libxrt_driver_xdna.so.2",
            "lib/libuuid.so.1",
            "THIRD-PARTY.json",
            "NOTICE",
        ] {
            assert!(
                manifest.files.contains_key(Path::new(path)),
                "missing {path}"
            );
        }
    }
    #[test]
    fn verified_install_reuses_and_repairs_without_exposing_partial_files() {
        let root = tempfile::tempdir().unwrap();
        let (manifest, archive) = archive(root.path());
        let cache = root.path().join("cache");
        let installed = manifest.install(&archive, &cache).unwrap();
        manifest.verify(&installed).unwrap();
        fs::write(installed.join("libhrx_npu.so.1"), "corrupt").unwrap();
        assert!(manifest.verify(&installed).is_err());
        assert_eq!(manifest.install(&archive, &cache).unwrap(), installed);
        manifest.verify(&installed).unwrap();
        fs::write(&archive, "wrong archive").unwrap();
        fs::remove_dir_all(&installed).unwrap();
        assert!(manifest.install(&archive, &cache).is_err());
        assert!(!installed.exists());
    }
    #[test]
    fn manifest_traversal_and_installed_links_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let (mut manifest, archive) = archive(root.path());
        manifest
            .files
            .insert("../outside".into(), bundle::digest(b"bad"));
        assert!(manifest.validate().is_err());
        manifest.files.remove(Path::new("../outside"));
        let installed = manifest
            .install(&archive, &root.path().join("cache"))
            .unwrap();
        fs::remove_file(installed.join("libhrx_npu.so.1")).unwrap();
        std::os::unix::fs::symlink(&archive, installed.join("libhrx_npu.so.1")).unwrap();
        assert!(manifest.verify(&installed).is_err());
    }
}
