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
/// Archive identity and expected contents; validated again at every installation entry point.
pub struct Manifest {
    /// Manifest format version (currently one).
    pub schema: u32,
    /// Supported platform and GPU target.
    pub target: String,
    /// Native build revision or provenance description.
    pub revision: String,
    /// HTTPS download URL or local file URL.
    pub url: String,
    /// Lowercase SHA-256 of the compressed archive.
    pub archive_sha256: String,
    /// Flat archive filenames mapped to lowercase SHA-256 digests.
    pub files: BTreeMap<String, String>,
}

/// Hash bytes as lowercase SHA-256 hex.
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
/// Stream a file through SHA-256 without loading it all into memory.
pub fn file_digest(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
/// Resolve the cache path without creating directories.
///
/// Follows the XDG Base Directory specification: `$XDG_CACHE_HOME/hrx`, or
/// `$HOME/.cache/hrx` when that is unset or empty. The specification requires
/// these variables to hold absolute paths and says a relative one is invalid and
/// must be ignored, so a relative `XDG_CACHE_HOME` falls back to `$HOME` rather
/// than resolving against the working directory.
pub fn cache_root() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".cache"))
        })
        .ok_or_else(|| {
            Error::Message(
                "set XDG_CACHE_HOME or HOME to an absolute path: no cache directory available"
                    .into(),
            )
        })?;
    Ok(base.join("hrx"))
}

/// The one cache for compiled kernels.
///
/// Artifacts are content-addressed — the key covers compiler identity, source,
/// export, target and canonical configuration — so there is nothing for a
/// per-consumer location to distinguish. Separate directories could only
/// duplicate identical artifacts and hide them from `hrx gc`.
pub fn kernel_cache() -> Result<PathBuf> {
    Ok(cache_root()?.join("kernels"))
}

/// An independently opened flock also serializes separate Rust copies in cdylibs.
/// Never unlink a lock file while another process might be waiting on its inode.
pub struct Lock(File);
impl Lock {
    /// Open and exclusively lock a file, refusing symlinks; unlocks on drop.
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
    /// Deserialize a manifest and validate its platform, digests and filenames.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let m: Self = serde_json::from_slice(bytes)
            .map_err(|e| Error::from(e).context("runtime manifest"))?;
        m.validate()?;
        Ok(m)
    }
    /// Validate public fields even when constructed directly or deserialized.
    pub fn validate(&self) -> Result<()> {
        let m = self;
        if m.schema != 1 {
            return Err(Error::Message(
                "unsupported runtime manifest schema or target".into(),
            ));
        }
        m.gpu_target()?;
        if !valid_digest(&m.archive_sha256) || m.files.is_empty() {
            return Err(Error::Message(
                "invalid bundle digest or empty bundle".into(),
            ));
        }
        for (name, hash) in &m.files {
            if !valid_name(name) || !valid_digest(hash) {
                return Err(Error::Message(format!("invalid bundle entry {name}")));
            }
        }
        for name in ["libhrx.so", "libhsa-runtime64.so.1", "libloomc.so"] {
            if !m.files.contains_key(name) {
                return Err(Error::Message(format!("bundle is missing {name}")));
            }
        }
        Ok(())
    }
    /// Parse the GPU architecture from the supported host platform identifier.
    pub fn gpu_target(&self) -> Result<crate::Target> {
        let key = self
            .target
            .strip_prefix("x86_64-unknown-linux-gnu-")
            .ok_or_else(|| Error::Message("unsupported bundle platform".into()))?;
        crate::Target::new(key)
    }
    /// Validate the manifest and verify every expected regular file in a directory.
    pub fn verify(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        for (name, expected) in &self.files {
            let path = directory.join(name);
            if !fs::symlink_metadata(&path)?.is_file() || &file_digest(&path)? != expected {
                return Err(Error::Message(format!(
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
        self.validate()?;
        create_cache_dir(root)?;
        let _lock = Lock::acquire(&root.join(format!("{}.lock", self.archive_sha256)))?;
        let destination = root.join(&self.archive_sha256);
        if self.verify(&destination).is_ok() {
            return Ok(destination);
        }
        if file_digest(archive)? != self.archive_sha256 {
            return Err(Error::Message("runtime archive SHA-256 mismatch".into()));
        }
        let staging = tempfile::tempdir_in(root)?;
        let decoder = flate2::read::GzDecoder::new(File::open(archive)?);
        let mut tar = tar::Archive::new(decoder);
        let mut seen = std::collections::BTreeSet::new();
        for entry in tar.entries()? {
            let mut entry = entry?;
            let name = entry.path()?.to_string_lossy().into_owned();
            if !valid_name(&name)
                || !entry.header().entry_type().is_file()
                || !self.files.contains_key(&name)
                || !seen.insert(name.clone())
            {
                return Err(Error::Message(format!(
                    "unexpected runtime archive entry: {name}"
                )));
            }
            let path = staging.path().join(&name);
            let mut out = File::create(&path)?;
            std::io::copy(&mut entry, &mut out)?;
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            out.sync_all()?;
        }
        self.verify(staging.path())?;
        fs::write(
            staging.path().join("manifest.json"),
            serde_json::to_vec_pretty(self)?,
        )?;
        // Failed/incomplete installations never become candidates for dlopen.
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        File::open(staging.path().join("manifest.json"))?.sync_all()?;
        File::open(staging.path())?.sync_all()?;
        publish(staging, &destination)?;
        File::open(root)?.sync_all()?;
        Ok(destination)
    }
}
/// What a cache sweep reclaimed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reclaimed {
    /// Superseded runtime bundle directories removed.
    pub bundles: usize,
    /// Bytes freed by removing those bundles.
    pub bundle_bytes: u64,
    /// Compiled kernel artifacts removed.
    pub artifacts: usize,
    /// Bytes freed by removing those artifacts.
    pub artifact_bytes: u64,
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => directory_bytes(&entry.path()),
            Ok(kind) if kind.is_file() => entry.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// The most recent access to anything in `path`, including `path` itself.
///
/// Reading a cached artifact updates the file's atime, not the directory's, so
/// the entry's own timestamp would report when it was created. `None` means no
/// timestamp could be read, which is treated as recently used: a collector that
/// deletes what it cannot date is not honouring the age it was given.
fn last_access(path: &Path) -> Option<std::time::SystemTime> {
    let own = fs::metadata(path).and_then(|m| m.accessed()).ok();
    let Ok(entries) = fs::read_dir(path) else {
        return own;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().and_then(|m| m.accessed()).ok())
        .chain(own)
        .max()
}

/// Remove cached runtime bundles other than `keep`, and compiled kernel
/// artifacts not accessed for longer than `unused_for`.
///
/// Provisioning publishes but never evicts, so a developer machine accumulates
/// every bundle it has ever prepared. This is deliberately explicit: nothing
/// here runs from `prepare`, because another checkout may still be using a
/// bundle this one has moved past. Removing a directory whose libraries are
/// already mapped is safe; the mapping holds the inode open.
pub fn collect(cache: &Path, keep: &str, unused_for: std::time::Duration) -> Result<Reclaimed> {
    let cutoff = std::time::SystemTime::now()
        .checked_sub(unused_for)
        .ok_or_else(|| Error::Message("cache age exceeds the system clock".into()))?;
    let mut reclaimed = Reclaimed::default();
    let runtime = cache.join("runtime");
    if let Ok(entries) = fs::read_dir(&runtime) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            // Only published digest directories are candidates. Temporary
            // directories belong to active installers and use the digest's lock,
            // not a lock named after the temporary directory.
            // Lock files stay: another process may be waiting on the inode.
            if name == keep
                || !valid_digest(name)
                || !entry.file_type().is_ok_and(|kind| kind.is_dir())
            {
                continue;
            }
            let _lock = Lock::acquire(&runtime.join(format!("{name}.lock")))?;
            if !entry.path().is_dir() {
                continue; // Another collector removed it while we waited.
            }
            let bytes = directory_bytes(&entry.path());
            fs::remove_dir_all(entry.path())?;
            reclaimed.bundles += 1;
            reclaimed.bundle_bytes += bytes;
        }
    }
    // `cache` is the root, not necessarily the process's own: tests and tools
    // collect a cache they were handed. `kernel_cache` names the default.
    let kernels = cache.join("kernels");
    if let Ok(entries) = fs::read_dir(&kernels) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !valid_digest(name) || !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                continue;
            }
            // Serialize with compilation, cache repair, and other collectors.
            // Read the timestamp only after a waiting writer has finished.
            let _lock = Lock::acquire(&kernels.join(format!("{name}.lock")))?;
            if !entry.path().is_dir() {
                continue;
            }
            // Last access, not age, and taken from the filesystem rather than from
            // any file this layout happens to contain: entries written by an older
            // release are dated the same way as current ones.
            if last_access(&entry.path()).is_none_or(|used| used >= cutoff) {
                continue;
            }
            let bytes = directory_bytes(&entry.path());
            fs::remove_dir_all(entry.path())?;
            reclaimed.artifacts += 1;
            reclaimed.artifact_bytes += bytes;
        }
    }
    Ok(reclaimed)
}

fn valid_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

/// Read the configured manifest or the embedded release manifest.
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
        return Err(Error::Message(
            "this HRX release supports Linux x86_64".into(),
        ));
    }
    if let Some(path) = std::env::var_os("HRX_RUNTIME_DIR") {
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

/// Reuse a verified bundle or provision it; offline mode never downloads.
pub fn prepare(manifest: &Manifest, root: &Path, offline: bool) -> Result<PathBuf> {
    manifest.validate()?;
    let destination = root.join(&manifest.archive_sha256);
    if manifest.verify(&destination).is_ok() {
        return Ok(destination);
    }
    create_cache_dir(root)?;
    // Download lock is distinct from the installation lock, and always taken first.
    let _download =
        Lock::acquire(&root.join(format!("{}.download.lock", manifest.archive_sha256)))?;
    if manifest.verify(&destination).is_ok() {
        return Ok(destination);
    }
    if offline {
        return Err(Error::Message(format!(
            "runtime bundle is missing or corrupt in {}; run `hrx prepare` online or set HRX_RUNTIME_DIR",
            destination.display()
        )));
    }
    let mut archive = tempfile::NamedTempFile::new_in(root)?;
    if let Some(path) = manifest.url.strip_prefix("file://") {
        std::io::copy(&mut File::open(path)?, &mut archive)?;
    } else {
        if !manifest.url.starts_with("https://") {
            return Err(Error::Message(
                "runtime bundle URL must use HTTPS or file://".into(),
            ));
        }
        #[cfg(feature = "download")]
        {
            let mut response = ureq::get(&manifest.url).call().map_err(|e| {
                Error::Download(Box::new(e)).context(format!("download {}", manifest.url))
            })?;
            std::io::copy(&mut response.body_mut().as_reader(), &mut archive)?;
        }
        #[cfg(not(feature = "download"))]
        return Err(Error::Message(
            "network provisioning disabled; enable `download` or prepare the cache offline".into(),
        ));
    }
    archive.flush()?;
    manifest.install(archive.path(), root)
}

/// Atomically hand a temporary directory to its final owner. Keep cleanup on
/// rename failure, and disarm TempDir only after the handoff succeeds.
pub(crate) fn publish(staging: tempfile::TempDir, destination: &Path) -> Result<()> {
    fs::rename(staging.path(), destination)?;
    let _ = staging.keep();
    Ok(())
}

pub(crate) fn create_cache_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)?;
    Ok(())
}

#[cfg(test)]
mod gc_tests {
    use super::*;

    #[test]
    fn active_installation_and_compilation_staging_survive_collection() {
        let cache = tempfile::tempdir().unwrap();
        for name in ["runtime", "kernels"] {
            fs::create_dir(cache.path().join(name)).unwrap();
        }
        let installation = tempfile::tempdir_in(cache.path().join("runtime")).unwrap();
        let compilation = tempfile::tempdir_in(cache.path().join("kernels")).unwrap();
        fs::write(installation.path().join("libhrx.so"), b"in progress").unwrap();
        fs::write(compilation.path().join("kernel.hsaco"), b"in progress").unwrap();

        let reclaimed = collect(
            cache.path(),
            &digest(b"pinned"),
            std::time::Duration::from_secs(30 * 24 * 60 * 60),
        )
        .unwrap();
        assert_eq!((reclaimed.bundles, reclaimed.artifacts), (0, 0));
        assert!(installation.path().join("libhrx.so").is_file());
        assert!(compilation.path().join("kernel.hsaco").is_file());
    }

    #[test]
    fn artifact_collection_waits_for_the_writer_and_rechecks_age() {
        let cache = tempfile::tempdir().unwrap();
        let kernels = cache.path().join("kernels");
        let key = digest(b"being repaired");
        let dir = kernels.join(&key);
        fs::create_dir_all(&dir).unwrap();
        let lock = Lock::acquire(&kernels.join(format!("{key}.lock"))).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).unwrap();
                let result = collect(
                    cache.path(),
                    &digest(b"pinned"),
                    std::time::Duration::from_secs(30 * 24 * 60 * 60),
                );
                done_tx.send(result).unwrap();
            });
            started_rx.recv().unwrap();
            let early = done_rx.recv_timeout(std::time::Duration::from_millis(100));
            // A compiler holding this key's lock is publishing repaired metadata.
            let write = fs::write(dir.join("artifact.json"), b"{}");
            drop(lock);
            assert!(matches!(
                early,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            write.unwrap();
            let reclaimed = done_rx.recv().unwrap().unwrap();
            assert_eq!(reclaimed.artifacts, 0);
            assert!(dir.join("artifact.json").is_file());
        });
    }

    #[test]
    fn sweeps_superseded_bundles_and_stale_artifacts_but_never_the_pinned_one() {
        let cache = tempfile::tempdir().unwrap();
        let runtime = cache.path().join("runtime");
        for name in ["keepme", "oldone", "olderone"] {
            fs::create_dir_all(runtime.join(digest(name.as_bytes()))).unwrap();
            fs::write(
                runtime.join(digest(name.as_bytes())).join("libhrx.so"),
                vec![0u8; 1024],
            )
            .unwrap();
        }
        let kernels = cache.path().join("kernels");
        // Eviction reads access time, so an entry is dated without regard to which
        // files it holds: "legacy" carries none of the current layout at all.
        for (name, age_days, extra) in [
            ("fresh", 0u64, "artifact.json"),
            ("stale", 90, "artifact.json"),
            ("legacy", 90, "kernel.sha256"),
        ] {
            let dir = kernels.join(digest(name.as_bytes()));
            fs::create_dir_all(&dir).unwrap();
            let when = std::time::SystemTime::now()
                - std::time::Duration::from_secs(age_days * 24 * 60 * 60);
            for file in ["kernel.hsaco", extra] {
                let path = dir.join(file);
                fs::write(&path, vec![0u8; 512]).unwrap();
                File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_times(fs::FileTimes::new().set_accessed(when).set_modified(when))
                    .unwrap();
            }
            File::open(&dir)
                .unwrap()
                .set_times(fs::FileTimes::new().set_accessed(when).set_modified(when))
                .unwrap();
        }
        let reclaimed = collect(
            cache.path(),
            &digest(b"keepme"),
            std::time::Duration::from_secs(30 * 24 * 60 * 60),
        )
        .unwrap();
        assert_eq!(reclaimed.bundles, 2);
        assert!(reclaimed.bundle_bytes >= 2048);
        assert_eq!(reclaimed.artifacts, 2, "stale and legacy entries both go");
        assert!(
            runtime.join(digest(b"keepme")).is_dir(),
            "pinned bundle survives"
        );
        assert!(!runtime.join(digest(b"oldone")).exists());
        assert!(kernels.join(digest(b"fresh")).is_dir());
        assert!(!kernels.join(digest(b"stale")).exists());
        assert!(
            !kernels.join(digest(b"legacy")).exists(),
            "an older layout is dated by access time like any other entry"
        );
        // A second sweep is a no-op rather than an error.
        let again = collect(
            cache.path(),
            &digest(b"keepme"),
            std::time::Duration::from_secs(30 * 24 * 60 * 60),
        )
        .unwrap();
        assert_eq!((again.bundles, again.artifacts), (0, 0));
    }
}
