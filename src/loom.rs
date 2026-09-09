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

/// How seriously the compiler meant a diagnostic. Ordered, so callers can
/// filter with a comparison rather than a magic number: a spill remark and a
/// type error arrive in the same list, and only one of them is worth a reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// A remark: informational, including backend resource notes.
    Note,
    /// A warning: compilation continued.
    Warning,
    /// An error: compilation of this specialization failed.
    Error,
}
impl From<u32> for Severity {
    /// Unknown native severities are treated as errors, so a future compiler
    /// cannot make a diagnostic disappear from a caller's error filter.
    fn from(native: u32) -> Self {
        match native {
            0 => Self::Note,
            1 => Self::Warning,
            _ => Self::Error,
        }
    }
}

/// A compiler diagnostic copied out of native result storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    /// How seriously the compiler meant this diagnostic.
    pub severity: Severity,
    /// Stable compiler diagnostic code.
    pub code: String,
    /// Rendered diagnostic message.
    pub message: String,
    /// One-based source line, or zero when unavailable.
    pub line: u32,
    /// One-based source column, or zero when unavailable.
    pub column: u32,
}
/// Compiler target and limits for concurrent workspaces and cached modules.
#[derive(Clone, Debug)]
pub struct CompilerOptions {
    /// Maximum number of simultaneous compilations through this compiler.
    pub workers: NonZeroUsize,
    /// Architecture used by the Loom target profile and artifact cache.
    pub target: crate::Target,
    /// Maximum retained source modules; zero disables module caching.
    /// Eviction chooses an arbitrary entry, not the least recently used module.
    pub module_cache_capacity: usize,
}
impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            target: crate::Target::default(),
            module_cache_capacity: 64,
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
    target: crate::Target,
    workers: NonZeroUsize,
    module_cache_capacity: usize,
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
        let native =
            native::Prepared::open(&path, &identity, options.workers.get(), &options.target)?;
        if bundle::file_digest(&path)? != identity {
            return Err(Error::Message(
                "compiler library changed while loading".into(),
            ));
        }
        Ok(Self(Arc::new(Inner {
            modules: Mutex::new(HashMap::new()),
            native,
            target: options.target,
            workers: options.workers,
            module_cache_capacity: options.module_cache_capacity,
            path,
            identity,
        })))
    }
    /// Maximum simultaneous compilations, and the workspace pool's bound.
    pub fn workers(&self) -> NonZeroUsize {
        self.0.workers
    }

    /// Compile several specializations, up to [`CompilerOptions::workers`] at a
    /// time, and return their outcomes in request order.
    ///
    /// [`Module::compile`] blocks, so a single-threaded caller never reaches the
    /// workspace pool's bound. This is the entry point that does: it owns the
    /// thread budget rather than leaving every consumer to rebuild the same pool.
    /// Each request is independent; one failure does not cancel the others.
    pub fn compile_all(
        &self,
        requests: &[(&Module, &Specialization)],
        cache: &Path,
    ) -> Vec<Result<Artifact>> {
        let limit = self.0.workers.get().min(requests.len());
        if limit <= 1 {
            return requests
                .iter()
                .map(|(module, spec)| module.compile(spec, cache))
                .collect();
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<Result<Artifact>>>> =
            requests.iter().map(|_| Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..limit {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((module, spec)) = requests.get(index) else {
                            return;
                        };
                        let outcome = module.compile(spec, cache);
                        *slots[index].lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
                    }
                });
            }
        });
        slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .unwrap_or_else(|e| e.into_inner())
                    .expect("every request slot is filled before the scope ends")
            })
            .collect()
    }

    /// Architecture selected for every compilation by this compiler.
    pub fn target(&self) -> &crate::Target {
        &self.0.target
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
        let data = if let Some(data) = modules.get(&digest) {
            data.clone()
        } else {
            let data = Arc::new(ModuleData {
                source: source.into(),
                digest: digest.clone(),
                index: Mutex::new(None),
            });
            if self.0.module_cache_capacity != 0 {
                if modules.len() >= self.0.module_cache_capacity {
                    let victim = modules.keys().next().cloned().unwrap();
                    modules.remove(&victim);
                }
                modules.insert(digest, data.clone());
            }
            data
        };
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
/// Render the diagnostic a caller should act on, not the whole cascade.
///
/// A single rejected construct makes the rest of its block unparseable, so the
/// compiler reports one real error followed by a run of consequences. Leading
/// with all of them buries the cause; every diagnostic stays available through
/// [`Error::Compile::diagnostics`].
fn summarize(diagnostics: &[Diagnostic]) -> String {
    let errors = || diagnostics.iter().filter(|d| d.severity >= Severity::Error);
    let Some(first) = errors().next() else {
        return diagnostics
            .iter()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
    };
    let mut message = if first.line == 0 {
        first.message.clone()
    } else {
        format!("{}:{}: {}", first.line, first.column, first.message)
    };
    if let Some(hint) = hint(&first.message) {
        message.push_str("\nhint: ");
        message.push_str(hint);
    }
    let suppressed = errors().count() - 1;
    if suppressed > 0 {
        message.push_str(&format!(
            "\n({suppressed} later error(s) suppressed as cascade; see Error::Compile diagnostics)"
        ));
    }
    message
}

/// Diagnostics whose cause is a configuration mistake rather than a source bug.
fn hint(message: &str) -> Option<&'static str> {
    if message.contains("Low representation contract") && message.contains(".generic.") {
        return Some(
            "a generic target has no low-asm contract; name a bare architecture \
             such as gfx1151 in amdgpu.target<...> to use hand-written asm",
        );
    }
    None
}

#[cfg(test)]
mod diagnostic_tests {
    use super::{Diagnostic, Severity, summarize};
    fn diagnostic(severity: u32, line: u32, message: &str) -> Diagnostic {
        Diagnostic {
            severity: Severity::from(severity),
            code: String::new(),
            message: message.into(),
            line,
            column: 1,
        }
    }
    #[test]
    fn the_first_error_leads_and_its_cascade_is_counted_not_printed() {
        let mut all = vec![
            diagnostic(1, 3, "unused binding"),
            diagnostic(
                2,
                7,
                "unknown Low representation contract amdgpu.gfx11.generic.core",
            ),
        ];
        all.extend((0..12).map(|i| diagnostic(2, 8 + i, "expected expression")));
        let rendered = summarize(&all);
        assert!(rendered.starts_with("7:1: unknown Low representation contract"));
        assert!(rendered.contains("hint: a generic target has no low-asm contract"));
        assert!(rendered.contains("12 later error(s) suppressed"));
        assert!(!rendered.contains("expected expression"));
        // A cascade-free failure gains no suppression line and no hint.
        let single = summarize(&[diagnostic(2, 4, "type mismatch")]);
        assert_eq!(single, "4:1: type mismatch");
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
    bytes: Arc<[u8]>,
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
            self.compiler.0.target.as_str(),
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
            target: self.compiler.0.target.as_str().into(),
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
            bytes: compiled.bytes,
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
    // Refresh the timestamp `bundle::collect` prunes by, so it means last use.
    if let Ok(file) = fs::File::open(dir.join("artifact.json")) {
        let now = std::time::SystemTime::now();
        let _ = file.set_times(fs::FileTimes::new().set_accessed(now).set_modified(now));
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires libloomc.so, no GPU"]
    fn module_cache_eviction_preserves_live_modules() -> Result<()> {
        let compiler = Compiler::with_options(
            None,
            CompilerOptions {
                module_cache_capacity: 2,
                ..Default::default()
            },
        )?;
        let first = compiler.module(include_str!("../tests/kernels/euler.loom"));
        let first_id = first.identity().to_owned();
        for i in 0..8 {
            compiler.module(&format!("// source {i}"));
            assert!(compiler.0.modules.lock().unwrap().len() <= 2);
        }
        compiler.trim();
        assert!(compiler.0.modules.lock().unwrap().is_empty());
        assert_eq!(first.identity(), first_id);
        let cache = tempfile::tempdir()?;
        let mut spec = Specialization::new("krea2_euler");
        spec.config.insert("krea2.euler.grid_x".into(), "1".into());
        spec.config.insert("krea2.euler.grid_y".into(), "1".into());
        assert!(!first.compile(&spec, cache.path())?.bytes().is_empty());
        Ok(())
    }
}
