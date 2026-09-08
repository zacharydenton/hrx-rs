//! Content-addressed Loom compilation. Prepared kernels stay outside this cold
//! path; no compiler process, file read, or hash is necessary during dispatch.
use crate::{
    Error, Result,
    bundle::{self, Lock},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Debug)]
pub struct Compiler {
    executable: PathBuf,
    identity: String,
}
impl Compiler {
    /// Resolve an explicit compiler, LOOM_COMPILE, or the pinned native bundle.
    pub fn resolve(override_path: Option<&Path>) -> Result<Self> {
        let path = if let Some(path) = override_path {
            resolve_executable(path)?
        } else if let Some(path) = std::env::var_os("LOOM_COMPILE") {
            resolve_executable(Path::new(&path))?
        } else {
            bundle::resolve()?.join("loom-compile")
        };
        let identity = bundle::file_digest(&path)?;
        Ok(Self {
            executable: path,
            identity,
        })
    }
    pub fn path(&self) -> &Path {
        &self.executable
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn key(&self, request: &Request<'_>) -> Result<String> {
        request.validate()?;
        // JSON arrays preserve boundaries (newline/equals in strings cannot
        // create collisions), and BTreeMap canonicalizes config ordering.
        let identity = serde_json::to_vec(&(
            1,
            &self.identity,
            request.source,
            request.symbol,
            request.backend,
            request.target,
            &request.config,
        ))
        .map_err(|e| Error(e.to_string()))?;
        Ok(bundle::digest(&identity))
    }
    /// Return a verified artifact, compiling once under a per-key process lock.
    /// Failures clean up staging and never publish partial output.
    pub fn compile(&self, request: &Request<'_>, cache: &Path) -> Result<PathBuf> {
        let key = self.key(request)?;
        let directory = cache.join(&key);
        let output = directory.join("kernel.hsaco");
        if verified(&directory) {
            return Ok(output);
        }
        fs::create_dir_all(cache)?;
        let _lock = Lock::acquire(&cache.join(format!("{key}.lock")))?;
        if verified(&directory) {
            return Ok(output);
        }
        if bundle::file_digest(&self.executable)? != self.identity {
            return Err(Error("Loom compiler changed during this session".into()));
        }
        let temporary = tempfile::tempdir_in(cache)?;
        let source = temporary.path().join("kernel.loom");
        let result = temporary.path().join("kernel.hsaco");
        fs::write(&source, request.source)?;
        let mut command = Command::new(&self.executable);
        command
            .arg(&source)
            .arg(format!("--backend={}", request.backend))
            .arg(format!("--target={}", request.target))
            .arg(format!("--root=@{}", request.symbol))
            .arg(format!("--output={}", result.display()));
        for (key, value) in &request.config {
            command.arg(format!("--config={key}={value}"));
        }
        // Write diagnostics to disk rather than accumulating unbounded compiler output.
        let log_path = temporary.path().join("compile.log");
        let log = fs::File::create(&log_path)?;
        let status = command
            .stdout(log.try_clone()?)
            .stderr(log)
            .status()
            .map_err(|e| Error(format!("starting {}: {e}", self.executable.display())))?;
        if !status.success() || !result.is_file() || fs::metadata(&result)?.len() == 0 {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = fs::File::open(&log_path)?;
            let length = file.metadata()?.len();
            file.seek(SeekFrom::Start(length.saturating_sub(8192)))?;
            let mut tail = Vec::new();
            file.read_to_end(&mut tail)?;
            return Err(Error(format!(
                "Loom compilation failed for {} ({status}):\n{}",
                request.symbol,
                String::from_utf8_lossy(&tail)
            )));
        }
        if bundle::file_digest(&self.executable)? != self.identity {
            return Err(Error("Loom compiler changed while compiling".into()));
        }
        fs::write(
            temporary.path().join("kernel.sha256"),
            bundle::file_digest(&result)?,
        )?;
        fs::File::open(&result)?.sync_all()?;
        if directory.exists() {
            fs::remove_dir_all(&directory)?;
        }
        fs::rename(temporary.path(), &directory)?;
        Ok(output)
    }
}
fn verified(path: &Path) -> bool {
    match (
        fs::read_to_string(path.join("kernel.sha256")),
        bundle::file_digest(&path.join("kernel.hsaco")),
    ) {
        (Ok(want), Ok(got)) => want == got,
        _ => false,
    }
}
fn resolve_executable(path: &Path) -> Result<PathBuf> {
    if path.components().count() == 1
        && let Some(search) = std::env::var_os("PATH")
    {
        for dir in std::env::split_paths(&search) {
            let candidate = dir.join(path);
            if candidate.is_file() {
                return Ok(fs::canonicalize(candidate)?);
            }
        }
    }
    fs::canonicalize(path).map_err(|e| Error(format!("compiler {}: {e}", path.display())))
}

pub struct Request<'a> {
    pub source: &'a str,
    pub symbol: &'a str,
    pub backend: &'a str,
    pub target: &'a str,
    /// Fully qualified keys, e.g. `h3.gemm.k_size` or `krea2.gemm.cols`.
    pub config: BTreeMap<String, String>,
}
impl<'a> Request<'a> {
    pub fn new(source: &'a str, symbol: &'a str) -> Self {
        Self {
            source,
            symbol,
            backend: "amdgpu-hal",
            target: crate::TARGET_KEY,
            config: BTreeMap::new(),
        }
    }
    fn validate(&self) -> Result<()> {
        for value in [self.symbol, self.backend, self.target] {
            if value.is_empty()
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            {
                return Err(Error("invalid Loom symbol, backend or target".into()));
            }
        }
        for key in self.config.keys() {
            if key.is_empty()
                || !key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            {
                return Err(Error(format!("invalid Loom configuration key {key}")));
            }
        }
        Ok(())
    }
}
