//! In-process Loom compilation with reusable indexed modules and verified disk artifacts.
#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::all
)]
mod ffi;
mod native;
use crate::{
    Error, Result,
    bundle::{self, Lock},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// A compiler diagnostic copied out of native result storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Native diagnostic severity (remark, warning, or error).
    pub severity: u32,
    /// Stable compiler diagnostic code.
    pub code: String,
    /// Rendered diagnostic message.
    pub message: String,
    /// One-based source line, or zero when unavailable.
    pub line: u32,
    /// One-based source column, or zero when unavailable.
    pub column: u32,
}
/// Limits for reusable compiler scratch. No background work is started.
#[derive(Clone, Debug)]
pub struct CompilerOptions {
    /// Maximum number of simultaneous compilations through this compiler.
    pub workers: NonZeroUsize,
}
impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            workers: NonZeroUsize::new(
                std::thread::available_parallelism()
                    .map_or(1, NonZeroUsize::get)
                    .min(4),
            )
            .unwrap(),
        }
    }
}
struct Inner {
    // Indexes are released before prepared compiler state when the session ends.
    modules: Mutex<HashMap<String, Arc<ModuleData>>>,
    native: native::Prepared,
    path: PathBuf,
    identity: String,
}
/// A pinned compiler library and reusable native state, shared by cheap clones.
#[derive(Clone)]
pub struct Compiler(Arc<Inner>);
impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler")
            .field("library", &self.0.path)
            .field("identity", &self.0.identity)
            .finish_non_exhaustive()
    }
}
impl Compiler {
    /// Select an explicit library, HRX_LOOM_LIBRARY, or the pinned native bundle.
    /// This loads CPU compiler code but never opens a GPU. Library mappings stay
    /// resident until process exit; contexts and workspaces remain session-owned.
    /// To upgrade a loaded library, select a new path or restart the process.
    pub fn resolve(library: Option<&Path>) -> Result<Self> {
        Self::with_options(library, CompilerOptions::default())
    }
    /// Resolve a compiler with a bounded number of exclusive workspaces.
    pub fn with_options(library: Option<&Path>, options: CompilerOptions) -> Result<Self> {
        let path = match library
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("HRX_LOOM_LIBRARY").map(PathBuf::from))
        {
            Some(p) => fs::canonicalize(&p)
                .map_err(|e| Error::from(e).context(format!("compiler library {}", p.display())))?,
            None => fs::canonicalize(bundle::resolve()?.join("libloomc.so"))?,
        };
        let identity = bundle::file_digest(&path)?;
        let native = native::Prepared::open(&path, &identity, options.workers.get())?;
        if bundle::file_digest(&path)? != identity {
            return Err(Error::Message(
                "compiler library changed while loading".into(),
            ));
        }
        Ok(Self(Arc::new(Inner {
            modules: Mutex::new(HashMap::new()),
            native,
            path,
            identity,
        })))
    }
    /// Content identity of the loaded compiler library.
    pub fn identity(&self) -> &str {
        &self.0.identity
    }
    /// Selected compiler library path.
    pub fn library_path(&self) -> &Path {
        &self.0.path
    }
    /// Retain a source module, sharing its parsed index across specializations.
    /// Parsing is deferred until the first artifact-cache miss.
    pub fn module(&self, source: &str) -> Module {
        let digest = bundle::digest(source.as_bytes());
        let mut modules = self.0.modules.lock().unwrap_or_else(|e| e.into_inner());
        let data = modules
            .entry(digest.clone())
            .or_insert_with(|| {
                Arc::new(ModuleData {
                    source: source.into(),
                    digest,
                    index: Mutex::new(None),
                })
            })
            .clone();
        Module {
            compiler: self.clone(),
            data,
        }
    }
    /// Release idle workspace blocks and cached module references. Existing
    /// Module handles remain valid; active compilations are unaffected.
    pub fn trim(&self) {
        self.0
            .modules
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.0.native.trim();
    }
}
struct ModuleData {
    source: Arc<str>,
    digest: String,
    index: Mutex<Option<(Arc<native::Index>, Vec<Diagnostic>)>>,
}
/// Immutable source plus compiler ownership. Clones share the frozen native index.
#[derive(Clone)]
pub struct Module {
    // Release the index before the last compiler/context owner.
    data: Arc<ModuleData>,
    compiler: Compiler,
}
/// One export and its exact configuration within a module.
#[derive(Clone, Debug, Default)]
pub struct Specialization {
    /// Export/root symbol to compile.
    pub symbol: String,
    /// Fully qualified configuration keys and Loom value spellings.
    pub config: BTreeMap<String, String>,
    /// Request a native resource manifest in the resulting artifact.
    pub report: bool,
}
impl Specialization {
    /// Select an export with no configuration overrides.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ..Self::default()
        }
    }
    fn validate(&self) -> Result<()> {
        let valid = |s: &str| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        };
        if !valid(&self.symbol) || self.config.keys().any(|k| !valid(k)) {
            return Err(Error::Message(
                "invalid Loom export or configuration key".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    key: String,
    compiler: String,
    target: String,
    symbol: String,
    sha256: String,
    diagnostics: Vec<Diagnostic>,
    report: Option<serde_json::Value>,
}
/// Owned native executable bytes and their verified compilation metadata.
#[derive(Clone, Debug)]
pub struct Artifact {
    bytes: Arc<[u8]>,
    path: PathBuf,
    record: Record,
}
impl Artifact {
    /// Native executable bytes, independent of compiler/result lifetimes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Durable verified cache artifact path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Compiler library content identity.
    pub fn compiler_identity(&self) -> &str {
        &self.record.compiler
    }
    /// Exact GPU target key.
    pub fn target(&self) -> &str {
        &self.record.target
    }
    /// Export compiled into this artifact.
    pub fn symbol(&self) -> &str {
        &self.record.symbol
    }
    /// Diagnostics retained from successful preparation and compilation.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.record.diagnostics
    }
    /// Optional native resource manifest.
    pub fn report(&self) -> Option<&serde_json::Value> {
        self.record.report.as_ref()
    }
}
struct Compiled {
    bytes: Vec<u8>,
    diagnostics: Vec<Diagnostic>,
    report: Option<serde_json::Value>,
}
impl Module {
    /// Identity of the exact source bytes.
    pub fn identity(&self) -> &str {
        &self.data.digest
    }
    /// Cache identity including compiler, pipeline contract, source and specialization.
    pub fn key(&self, spec: &Specialization) -> Result<String> {
        spec.validate()?;
        Ok(bundle::digest(&serde_json::to_vec(&(
            "loomc-v1",
            self.compiler.identity(),
            self.identity(),
            &spec.symbol,
            crate::TARGET_KEY,
            &spec.config,
            spec.report,
        ))?))
    }
    /// Compile in process or return verified cached bytes. Publication is atomic
    /// and serialized per key across threads and processes; failures are retryable.
    pub fn compile(&self, spec: &Specialization, cache: &Path) -> Result<Artifact> {
        let key = self.key(spec)?;
        let dir = cache.join(&key);
        if let Some(a) = cached(&dir, &key) {
            return Ok(a);
        }
        bundle::create_cache_dir(cache)?;
        let _lock = Lock::acquire(&cache.join(format!("{key}.lock")))?;
        if let Some(a) = cached(&dir, &key) {
            return Ok(a);
        }
        let (index, mut diagnostics) = {
            let mut slot = self.data.index.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                let (index, diagnostics) = self.compiler.0.native.index(&self.data.source)?;
                *slot = Some((Arc::new(index), diagnostics));
            }
            slot.as_ref().unwrap().clone()
        };
        let compiled = self.compiler.0.native.compile(&index, spec).map_err(|e| {
            e.context(format!(
                "compiling {} with {}",
                spec.symbol,
                self.compiler.library_path().display()
            ))
        })?;
        diagnostics.extend(compiled.diagnostics);
        let record = Record {
            key,
            compiler: self.compiler.identity().into(),
            target: crate::TARGET_KEY.into(),
            symbol: spec.symbol.clone(),
            sha256: bundle::digest(&compiled.bytes),
            diagnostics,
            report: compiled.report,
        };
        let staging = tempfile::tempdir_in(cache)?;
        let output = staging.path().join("kernel.hsaco");
        fs::write(&output, &compiled.bytes)?;
        let metadata = serde_json::to_vec(&record)?;
        fs::write(staging.path().join("artifact.json"), &metadata)?;
        fs::write(
            staging.path().join("artifact.sha256"),
            bundle::digest(&metadata),
        )?;
        for file in ["kernel.hsaco", "artifact.json", "artifact.sha256"] {
            fs::File::open(staging.path().join(file))?.sync_all()?;
        }
        fs::File::open(staging.path())?.sync_all()?;
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        bundle::publish(staging, &dir)?;
        fs::File::open(cache)?.sync_all()?;
        Ok(Artifact {
            bytes: compiled.bytes.into(),
            path: dir.join("kernel.hsaco"),
            record,
        })
    }
}
fn cached(dir: &Path, key: &str) -> Option<Artifact> {
    let metadata = fs::read(dir.join("artifact.json")).ok()?;
    if fs::read_to_string(dir.join("artifact.sha256")).ok()? != bundle::digest(&metadata) {
        return None;
    }
    let record: Record = serde_json::from_slice(&metadata).ok()?;
    if record.key != key {
        return None;
    }
    let path = dir.join("kernel.hsaco");
    let bytes = fs::read(&path).ok()?;
    if bytes.is_empty() || bundle::digest(&bytes) != record.sha256 {
        return None;
    }
    Some(Artifact {
        bytes: bytes.into(),
        path,
        record,
    })
}
