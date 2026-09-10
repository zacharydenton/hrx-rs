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
    /// The compiler for this library and target, resolved once per process.
    ///
    /// Resolving is not cheap: it canonicalizes the library path and digests
    /// the whole shared object, twice. A consumer that asks for a compiler per
    /// kernel pays that per kernel, so every consumer memoized it themselves.
    ///
    /// The cache is keyed by the resolved path and target, so `HRX_LOOM_LIBRARY`
    /// is honoured on each call rather than frozen at the first.
    pub fn shared(library: Option<&Path>, options: CompilerOptions) -> Result<Self> {
        static RESOLVED: Mutex<Option<HashMap<(PathBuf, String), Compiler>>> = Mutex::new(None);
        let path = Self::resolved_library(library)?;
        let key = (path, options.target.as_str().to_owned());
        let mut cache = RESOLVED
            .lock()
            .map_err(|_| Error::Message("compiler cache poisoned".into()))?;
        let cache = cache.get_or_insert_with(HashMap::new);
        if let Some(compiler) = cache.get(&key) {
            return Ok(compiler.clone());
        }
        let compiler = Self::with_options(library, options)?;
        cache.insert(key, compiler.clone());
        Ok(compiler)
    }

    /// The library a given override, `HRX_LOOM_LIBRARY` or the bundle selects.
    fn resolved_library(library: Option<&Path>) -> Result<PathBuf> {
        match library
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("HRX_LOOM_LIBRARY").map(PathBuf::from))
        {
            Some(p) => fs::canonicalize(&p)
                .map_err(|e| Error::from(e).context(format!("compiler library {}", p.display()))),
            None => Ok(fs::canonicalize(bundle::resolve()?.join("libloomc.so"))?),
        }
    }

    /// Resolve a compiler with a bounded number of exclusive workspaces.
    pub fn with_options(library: Option<&Path>, options: CompilerOptions) -> Result<Self> {
        let path = Self::resolved_library(library)?;
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
    pub fn compile_all(&self, requests: &[(&Module, &Specialization)]) -> Vec<Result<Artifact>> {
        let limit = self.0.workers.get().min(requests.len());
        if limit <= 1 {
            return requests
                .iter()
                .map(|(module, spec)| module.compile(spec))
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
                        let outcome = module.compile(spec);
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
    pub fn compile(&self, spec: &Specialization) -> Result<Artifact> {
        let key = self.key(spec)?;
        let cache = &bundle::kernel_cache()?;
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
    // `bundle::collect` evicts on access time. Reading the artifact below updates
    // it on a normal mount, but a `noatime` mount never would, so set it here:
    // utimensat is unaffected by the mount option that suppresses the implicit
    // update, and without this a busy cache would look idle to the collector.
    if let Ok(file) = fs::File::open(dir.join("artifact.json")) {
        let _ = file.set_times(fs::FileTimes::new().set_accessed(std::time::SystemTime::now()));
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
        let mut spec = Specialization::new("krea2_euler");
        spec.config.insert("krea2.euler.grid_x".into(), "1".into());
        spec.config.insert("krea2.euler.grid_y".into(), "1".into());
        assert!(!first.compile(&spec)?.bytes().is_empty());
        Ok(())
    }
}

/// Loaded kernels, keyed by the artifact their specialization compiles to.
///
/// The compiler already caches *compilation* on disk. This caches the loaded
/// executable, which is what a dispatch loop needs: turning a cached artifact
/// into a [`crate::Kernel`] is a native load every time, and a model that dispatches
/// the same kernel per block would pay it per block. Every consumer of this
/// crate wrote this cache, so it lives here instead.
///
/// The key is the artifact's own identity — the same digest the disk cache uses
/// — rather than whatever the caller derived a request from, so two requests
/// that compile to the same artifact share one executable.
///
/// Kernels are device-scoped, so a cache serves one device and refuses another.
#[derive(Clone)]
pub struct Kernels(Arc<Cache>);

struct Cache {
    compiler: Compiler,
    state: Mutex<Loaded>,
    report: Option<Reporter>,
}

/// Called with each artifact a [`Kernels`] builds. See [`Kernels::reporting`].
type Reporter = Arc<dyn Fn(&Artifact) + Send + Sync>;

#[derive(Default)]
struct Loaded {
    /// Set by the first request; kernels from one device are useless on another.
    device: Option<usize>,
    ready: HashMap<String, crate::Kernel>,
    /// Requested but not yet built, in request order.
    waiting: Vec<(String, String, Specialization)>,
}

/// A kernel that has been asked for but not yet built.
///
/// Compiling is what a cold cache costs and requests are independent, so
/// [`Kernels::request`] records one and returns this rather than stopping to
/// build it. The outstanding set is then built together, on as many threads as
/// the compiler allows, by [`Kernels::build`] or by the first
/// [`Pending::resolve`] that needs any one of them.
pub struct Pending {
    key: String,
    cache: Arc<Cache>,
    /// This handle's own copy, so a built kernel can be borrowed for as long as
    /// the handle lives. A graph records `&'g Kernel`, and the cache's own copy
    /// is behind a lock with no lifetime to lend.
    cell: std::sync::OnceLock<crate::Kernel>,
}

impl Clone for Pending {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            cache: self.cache.clone(),
            cell: self.cell.clone(),
        }
    }
}

impl Pending {
    /// The kernel, if its batch has already been built.
    ///
    /// For callers with no stream to build one on — recording into a graph,
    /// which cannot load an executable. Everywhere else wants
    /// [`Pending::resolve`].
    pub fn built(&self) -> Option<&crate::Kernel> {
        if self.cell.get().is_none() {
            let found = self.cache.locked().ok()?.ready.get(&self.key).cloned()?;
            let _ = self.cell.set(found);
        }
        self.cell.get()
    }

    /// The kernel, building everything outstanding if this is the first call
    /// that needs it.
    ///
    /// # Safety
    /// As [`Kernels::get`], for every source passed to [`Kernels::request`].
    pub unsafe fn resolve(&self, stream: &crate::Stream) -> Result<&crate::Kernel> {
        if self.built().is_some() {
            return Ok(self.cell.get().expect("just built"));
        }
        // A batch reports the first kernel in it that would not build, which is
        // not necessarily this one, so ask again before passing that failure on
        // as though it were ours.
        // Safety: the caller vouched for every requested source.
        let outcome = unsafe { self.cache.build(stream) };
        match self.built() {
            Some(_) => Ok(self.cell.get().expect("just built")),
            None => Err(outcome
                .err()
                .unwrap_or_else(|| Error::Message("kernel was never requested".into()))),
        }
    }
}

impl Kernels {
    /// Index this cache by a caller's compact request key.
    ///
    /// Source hashing and specialization construction happen only on a key
    /// miss. Distinct caller keys that name the same artifact still share one
    /// loaded executable. Include every input that changes the source or
    /// specialization in the key.
    pub fn keyed<K>(self) -> KeyedKernels<K> {
        KeyedKernels {
            kernels: self,
            keys: Mutex::new(HashMap::new()),
        }
    }

    /// An empty cache over a compiler. Clone it to share one cache.
    pub fn new(compiler: Compiler) -> Self {
        Self(Arc::new(Cache {
            compiler,
            state: Mutex::new(Loaded::default()),
            report: None,
        }))
    }

    /// Call `report` with each artifact as it is built, once, before it loads.
    ///
    /// A cache that returned only kernels would swallow what the compiler said
    /// about them: warnings, and the backend remarks that are the only warning
    /// a kernel is spilling registers. Cache hits report nothing, having built
    /// nothing.
    pub fn reporting(self, report: impl Fn(&Artifact) + Send + Sync + 'static) -> Self {
        let cache = Arc::try_unwrap(self.0).unwrap_or_else(|shared| Cache {
            compiler: shared.compiler.clone(),
            state: Mutex::new(Loaded::default()),
            report: shared.report.clone(),
        });
        Self(Arc::new(Cache {
            report: Some(Arc::new(report)),
            ..cache
        }))
    }

    /// The compiler these were built with.
    pub fn compiler(&self) -> &Compiler {
        &self.0.compiler
    }

    /// How many kernels are loaded.
    pub fn len(&self) -> usize {
        self.0.locked().map_or(0, |state| state.ready.len())
    }

    /// Whether any kernel is loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compile `spec` from `source` if needed and load it, or return the
    /// loaded kernel. Outstanding [`Kernels::request`]s are left alone.
    ///
    /// # Safety
    /// `source` must be trusted Loom: its compiled output is loaded as native
    /// code, as for [`crate::Stream::load_artifact`].
    pub unsafe fn get(
        &self,
        stream: &crate::Stream,
        source: &str,
        spec: &Specialization,
    ) -> Result<crate::Kernel> {
        let module = self.0.compiler.module(source);
        let key = module.key(spec)?;
        // Safety: the caller vouched for this module's source.
        unsafe { self.get_module(stream, &module, key, spec) }
    }

    unsafe fn get_module(
        &self,
        stream: &crate::Stream,
        module: &Module,
        key: String,
        spec: &Specialization,
    ) -> Result<crate::Kernel> {
        {
            let mut state = self.0.locked()?;
            state.claim(stream)?;
            if let Some(kernel) = state.ready.get(&key) {
                return Ok(kernel.clone());
            }
        }
        let artifact = module.compile(spec)?;
        self.0.reported(&artifact);
        // Safety: the caller vouched for the source this artifact came from.
        let kernel = unsafe { stream.load_artifact(&artifact) }?;
        self.0.locked()?.ready.insert(key, kernel.clone());
        Ok(kernel)
    }

    /// Ask for a kernel without building it. Nothing is compiled until
    /// [`Kernels::build`] or the first [`Pending::resolve`], and then every
    /// outstanding request is built at once across the compiler's workspaces —
    /// so a consumer can name its whole set first and pay for it in one batch.
    pub fn request(&self, source: &str, spec: &Specialization) -> Result<Pending> {
        let key = self.0.compiler.module(source).key(spec)?;
        let mut state = self.0.locked()?;
        let known = state.ready.contains_key(&key)
            || state.waiting.iter().any(|(waiting, ..)| *waiting == key);
        if !known {
            state
                .waiting
                .push((key.clone(), source.to_owned(), spec.clone()));
        }
        drop(state);
        Ok(Pending {
            key,
            cache: self.0.clone(),
            cell: std::sync::OnceLock::new(),
        })
    }

    /// Build every outstanding request, in one batch.
    ///
    /// # Safety
    /// As [`Kernels::get`], for every source passed to [`Kernels::request`].
    pub unsafe fn build(&self, stream: &crate::Stream) -> Result<()> {
        // Safety: the caller vouched for every requested source.
        unsafe { self.0.build(stream) }
    }
}

impl Cache {
    fn locked(&self) -> Result<std::sync::MutexGuard<'_, Loaded>> {
        self.state
            .lock()
            .map_err(|_| Error::Message("kernel cache poisoned".into()))
    }

    fn reported(&self, artifact: &Artifact) {
        if let Some(report) = &self.report {
            report(artifact);
        }
    }

    /// Compile off the device, then load in order.
    ///
    /// A specialization that will not build is that one's failure and no one
    /// else's, so the batch carries on and the rest is loaded. The failure goes
    /// back on the queue still wanted: dropping it would leave the handles that
    /// asked for it waiting on nothing, with no way to say why, while returning
    /// it means the next attempt reports the same failure again.
    unsafe fn build(&self, stream: &crate::Stream) -> Result<()> {
        let waiting = {
            let mut state = self.locked()?;
            state.claim(stream)?;
            std::mem::take(&mut state.waiting)
        };
        if waiting.is_empty() {
            return Ok(());
        }
        let modules: Vec<Module> = waiting
            .iter()
            .map(|(_, source, _)| self.compiler.module(source))
            .collect();
        let requests: Vec<(&Module, &Specialization)> = modules
            .iter()
            .zip(waiting.iter().map(|(_, _, spec)| spec))
            .collect();
        let built = self.compiler.compile_all(&requests);

        let mut failure = None;
        let mut unbuilt = Vec::new();
        for (request, outcome) in waiting.into_iter().zip(built) {
            let loaded = outcome.and_then(|artifact| {
                self.reported(&artifact);
                // Safety: the caller vouched for the source when requesting it.
                unsafe { stream.load_artifact(&artifact) }
            });
            match loaded {
                Ok(kernel) => {
                    self.locked()?.ready.insert(request.0.clone(), kernel);
                }
                Err(error) => {
                    unbuilt.push(request);
                    failure = failure.or(Some(error));
                }
            }
        }
        if !unbuilt.is_empty() {
            let mut state = self.locked()?;
            unbuilt.append(&mut state.waiting);
            state.waiting = unbuilt;
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// A caller-key index over HRX's artifact-keyed loaded kernels.
///
/// Create with [`Kernels::keyed`]. A hit checks device ownership and returns a
/// shared executable without hashing source, serializing a specialization or
/// invoking the request factory. The index stores artifact identities, not a
/// second collection of executables.
pub struct KeyedKernels<K> {
    kernels: Kernels,
    keys: Mutex<HashMap<K, String>>,
}

impl<K: Eq + std::hash::Hash> KeyedKernels<K> {
    /// Return the kernel named by `key`, constructing its request only on a miss.
    ///
    /// Failed requests are not indexed and can be retried. The factory is run
    /// under the index lock; it and the reporting callback must not reenter
    /// this index. The underlying cache's device restriction applies to hits too.
    ///
    /// # Safety
    /// As [`Kernels::get`]. Equal keys must always describe the same source and
    /// specialization, including compiler-report settings. The factory is not
    /// evaluated on a hit, so changing it does not change a cached kernel.
    pub unsafe fn get_or_insert_with<'s>(
        &self,
        stream: &crate::Stream,
        key: K,
        request: impl FnOnce(&K) -> Result<(&'s str, Specialization)>,
    ) -> Result<crate::Kernel> {
        let mut keys = self
            .keys
            .lock()
            .map_err(|_| Error::Message("kernel key index poisoned".into()))?;
        {
            let mut state = self.kernels.0.locked()?;
            state.claim(stream)?;
            if let Some(artifact) = keys.get(&key) {
                return Ok(state.ready[artifact].clone());
            }
        }
        let (source, spec) = request(&key)?;
        let module = self.kernels.compiler().module(source);
        let artifact = module.key(&spec)?;
        // Safety: the caller vouched for the source and the request key.
        let kernel = unsafe {
            self.kernels
                .get_module(stream, &module, artifact.clone(), &spec)
        }?;
        keys.insert(key, artifact);
        Ok(kernel)
    }
}

impl Loaded {
    /// Bind this cache to a device on first use, and hold it there.
    fn claim(&mut self, stream: &crate::Stream) -> Result<()> {
        let device = stream.device_id();
        match self.device {
            Some(owner) if owner != device => Err(Error::Message(
                "a kernel cache serves one device: an executable loaded on another is not \
                 dispatchable here"
                    .into(),
            )),
            Some(_) => Ok(()),
            None => {
                self.device = Some(device);
                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels")
            .field("loaded", &self.len())
            .finish_non_exhaustive()
    }
}
