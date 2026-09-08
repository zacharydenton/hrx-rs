//! Reproducible native provisioning. No build script downloads or native link flags.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub target: String,
    pub revision: String,
    pub url: String,
    pub archive_sha256: String,
    pub files: BTreeMap<String, String>,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file_digest(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn cache_root() -> Result<PathBuf> {
    let base = std::env::var_os("HRX_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                .map(|p| p.join("hrx"))
        })
        .ok_or_else(|| Error("set HRX_CACHE_DIR: no home or cache directory available".into()))?;
    fs::create_dir_all(&base)?;
    Ok(base)
}

/// An independently opened flock also serializes separate Rust copies in cdylibs.
/// Never unlink a lock file while another process might be waiting on its inode.
pub struct Lock(File);
impl Lock {
    pub fn acquire(path: &Path) -> Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        let file = OpenOptions::new()
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        loop {
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
                return Ok(Self(file));
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(e.into());
            }
        }
    }
    pub(crate) fn file(&mut self) -> &mut File {
        &mut self.0
    }
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let m: Self =
            serde_json::from_slice(bytes).map_err(|e| Error(format!("runtime manifest: {e}")))?;
        if m.schema != 1 || m.target != "x86_64-unknown-linux-gnu-gfx1151" {
            return Err(Error(
                "unsupported runtime manifest schema or target".into(),
            ));
        }
        fn sha(s: &str) -> bool {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        }
        if !sha(&m.archive_sha256) || m.files.is_empty() {
            return Err(Error("invalid bundle digest or empty bundle".into()));
        }
        for (name, hash) in &m.files {
            if !valid_name(name) || !sha(hash) {
                return Err(Error(format!("invalid bundle entry {name}")));
            }
        }
        for name in ["libhrx.so", "libhsa-runtime64.so.1", "loom-compile"] {
            if !m.files.contains_key(name) {
                return Err(Error(format!("bundle is missing {name}")));
            }
        }
        Ok(m)
    }
    pub fn verify(&self, directory: &Path) -> Result<()> {
        for (name, expected) in &self.files {
            let path = directory.join(name);
            if !fs::symlink_metadata(&path)?.is_file() || &file_digest(&path)? != expected {
                return Err(Error(format!(
                    "runtime integrity check failed: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
    /// Verify the compressed archive before unpacking. Reject links, paths and
    /// undeclared files, then publish the entire directory with one rename.
    pub fn install(&self, archive: &Path, root: &Path) -> Result<PathBuf> {
        fs::create_dir_all(root)?;
        let _lock = Lock::acquire(&root.join(format!("{}.lock", self.archive_sha256)))?;
        let destination = root.join(&self.archive_sha256);
        if self.verify(&destination).is_ok() {
            return Ok(destination);
        }
        if file_digest(archive)? != self.archive_sha256 {
            return Err(Error("runtime archive SHA-256 mismatch".into()));
        }
        let staging = tempfile::tempdir_in(root)?;
        let decoder = flate2::read::GzDecoder::new(File::open(archive)?);
        let mut tar = tar::Archive::new(decoder);
        let mut seen = std::collections::BTreeSet::new();
        for entry in tar.entries()? {
            let mut entry = entry?;
            let name = entry.path()?.to_string_lossy().into_owned();
            if !entry.header().entry_type().is_file()
                || !self.files.contains_key(&name)
                || !seen.insert(name.clone())
            {
                return Err(Error(format!("unexpected runtime archive entry: {name}")));
            }
            let path = staging.path().join(&name);
            let mut out = File::create(&path)?;
            std::io::copy(&mut entry, &mut out)?;
            out.sync_all()?;
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        }
        self.verify(staging.path())?;
        fs::write(
            staging.path().join("manifest.json"),
            serde_json::to_vec_pretty(self).map_err(|e| Error(e.to_string()))?,
        )?;
        // Failed/incomplete installations never become candidates for dlopen.
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(staging.path(), &destination)?;
        File::open(root)?.sync_all()?;
        Ok(destination)
    }
}
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

pub fn default_manifest() -> Result<Manifest> {
    if let Some(path) = std::env::var_os("HRX_BUNDLE_MANIFEST") {
        return Manifest::parse(&fs::read(path)?);
    }
    Manifest::parse(include_bytes!("../bundle.json"))
}

/// Explicit runtime directories are trusted developer overrides. Packaged
/// directories with a manifest are verified just like the default cache.
pub fn resolve() -> Result<PathBuf> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(Error(
            "this HRX release supports Linux x86_64 / gfx1151".into(),
        ));
    }
    if let Some(path) =
        std::env::var_os("HRX_RUNTIME_DIR").or_else(|| std::env::var_os("KREA2_RUNTIME"))
    {
        let path = fs::canonicalize(path)?;
        if path.join("manifest.json").is_file() {
            Manifest::parse(&fs::read(path.join("manifest.json"))?)?.verify(&path)?;
        }
        return Ok(path);
    }
    prepare(
        &default_manifest()?,
        &cache_root()?.join("runtime"),
        std::env::var_os("HRX_OFFLINE").is_some(),
    )
}

pub fn prepare(manifest: &Manifest, root: &Path, offline: bool) -> Result<PathBuf> {
    let destination = root.join(&manifest.archive_sha256);
    if manifest.verify(&destination).is_ok() {
        return Ok(destination);
    }
    fs::create_dir_all(root)?;
    // Download lock is distinct from the installation lock, and always taken first.
    let _download =
        Lock::acquire(&root.join(format!("{}.download.lock", manifest.archive_sha256)))?;
    if manifest.verify(&destination).is_ok() {
        return Ok(destination);
    }
    if offline {
        return Err(Error(format!(
            "runtime bundle is missing or corrupt in {}; run `hrx prepare` online or set HRX_RUNTIME_DIR",
            destination.display()
        )));
    }
    let mut archive = tempfile::NamedTempFile::new_in(root)?;
    if let Some(path) = manifest.url.strip_prefix("file://") {
        std::io::copy(&mut File::open(path)?, &mut archive)?;
    } else {
        if !manifest.url.starts_with("https://") {
            return Err(Error("runtime bundle URL must use HTTPS or file://".into()));
        }
        #[cfg(feature = "download")]
        {
            let mut response = ureq::get(&manifest.url)
                .call()
                .map_err(|e| Error(format!("download {}: {e}", manifest.url)))?;
            std::io::copy(&mut response.body_mut().as_reader(), &mut archive)?;
        }
        #[cfg(not(feature = "download"))]
        return Err(Error(
            "network provisioning disabled; enable `download` or prepare the cache offline".into(),
        ));
    }
    archive.flush()?;
    manifest.install(archive.path(), root)
}
