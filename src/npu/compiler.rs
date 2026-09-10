//! Compile IRON/AIE projects in isolated subprocesses with verified toolchains.
//!
//! Compilation never opens an accelerator. Project generators are trusted host
//! programs: isolation prevents output collisions, not arbitrary code execution.
use crate::{Error, Result, bundle, execution::KernelContract};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

// Kill and reap the compiler process tree on every early-return path.
struct Process {
    child: Child,
    reaped: bool,
}
impl Drop for Process {
    fn drop(&mut self) {
        if !self.reaped {
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
    }
}

/// Explicit tile compiler selection; backends are never silently substituted.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Backend {
    /// LLVM-AIE/Peano.
    Peano,
    /// User-installed AMD Chess.
    Chess,
}
/// Pinned tools and environment for one compiler installation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Toolchain {
    /// Python executable inside the pinned IRON environment (preserve venv symlinks).
    pub python: PathBuf,
    /// Pinned aiecc executable for direct AIE MLIR projects.
    pub aiecc: PathBuf,
    /// Per-core compiler selection.
    pub backend: Backend,
    /// Every compiler/package/support file affecting output, with its expected SHA-256.
    pub files: BTreeMap<PathBuf, String>,
    /// Child-only environment, including tool search paths and Chess setup if used.
    pub environment: BTreeMap<String, String>,
}
impl Toolchain {
    /// Read and verify a toolchain manifest. File names are absolute.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let tools: Self = serde_json::from_slice(&fs::read(path)?)?;
        tools.verify()?;
        Ok(tools)
    }
    /// Verify actual tools and support files against their pinned digests.
    pub fn verify(&self) -> Result<()> {
        if self.files.is_empty()
            || !self.python.is_absolute()
            || !self.aiecc.is_absolute()
            || !self.files.contains_key(&self.python)
            || !self.files.contains_key(&self.aiecc)
        {
            return Err(Error::Message(
                "toolchain must pin absolute Python and aiecc paths and their dependencies".into(),
            ));
        }
        for (path, expected) in &self.files {
            if !path.is_absolute() || bundle::file_digest(path)? != *expected {
                return Err(Error::Message(format!(
                    "toolchain file changed: {}",
                    path.display()
                )));
            }
        }
        if matches!(self.backend, Backend::Chess) && !self.environment.contains_key("AIETOOLS_ROOT")
        {
            return Err(Error::Message(
                "Chess requires explicit AIETOOLS_ROOT and pinned support files".into(),
            ));
        }
        Ok(())
    }
    /// Content identity including backend and environment.
    pub fn identity(&self) -> Result<String> {
        Ok(bundle::digest(&serde_json::to_vec(self)?))
    }
}
/// Compiler process and cache limits.
#[derive(Clone, Debug)]
pub struct CompilerOptions {
    /// Maximum simultaneous subprocess workspaces.
    pub workers: usize,
    /// Time limit for one generator/compiler process tree.
    pub timeout: Duration,
    /// Artifact root; defaults to the common HRX kernel cache.
    pub cache: PathBuf,
}
impl CompilerOptions {
    /// Conservative defaults with four compile workers and a ten-minute timeout.
    pub fn new() -> Result<Self> {
        Ok(Self {
            workers: 4,
            timeout: Duration::from_secs(600),
            cache: bundle::kernel_cache()?.join("npu"),
        })
    }
}
/// The source entry point consumed by the compiler adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Source {
    /// IRON generator accepting --xclbin-path and --insts-path.
    Iron(PathBuf),
    /// AIE MLIR passed directly to aiecc.
    Mlir(PathBuf),
}
/// A fixed project specialization. Paths are relative to `root`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    /// Source directory. Compilation occurs in a private copy.
    pub root: PathBuf,
    /// Generator or MLIR entry point.
    pub source: Source,
    /// Complete dependency list, including tile sources and headers.
    pub dependencies: Vec<PathBuf>,
    /// Explicit generator/aiecc arguments, including target and shape.
    pub arguments: Vec<String>,
    /// Whether the dependency list covers every non-toolchain input. False disables reuse.
    pub cacheable: bool,
    /// Fixed dispatch ABI metadata; trust is asserted separately at kernel loading.
    pub contract: KernelContract,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    schema: u32,
    key: String,
    compiler: String,
    xclbin: String,
    instructions: String,
    contract: KernelContract,
}
/// Immutable compiled device image, instruction stream, and ABI metadata.
#[derive(Clone)]
pub struct Artifact {
    _temporary: Option<Arc<tempfile::TempDir>>,
    directory: PathBuf,
    key: String,
}
impl Artifact {
    /// Directory containing x.xclbin, x.bin, manifest.json, and build.log.
    pub fn path(&self) -> &Path {
        &self.directory
    }
    /// Content-addressed specialization identity.
    pub fn identity(&self) -> &str {
        &self.key
    }
    fn metadata(&self) -> Result<Metadata> {
        let metadata: Metadata =
            serde_json::from_slice(&fs::read(self.directory.join("manifest.json"))?)?;
        if metadata.schema != 1
            || metadata.key != self.key
            || bundle::file_digest(&self.directory.join("x.xclbin"))? != metadata.xclbin
            || bundle::file_digest(&self.directory.join("x.bin"))? != metadata.instructions
        {
            return Err(Error::Message(
                "NPU artifact is corrupt or has a different identity".into(),
            ));
        }
        metadata.contract.validate()?;
        Ok(metadata)
    }
    /// Verify output digests and ABI metadata before reuse.
    pub fn verify(&self) -> Result<()> {
        self.metadata().map(|_| ())
    }
    /// Trusted loading of a compiled program and fixed instruction specialization.
    /// # Safety
    /// The generated image and instructions must satisfy the stored contract;
    /// successful compilation and hashes are not a memory-safety proof.
    pub unsafe fn load(&self, device: i32) -> Result<(super::NpuProgram, super::NpuKernel)> {
        let metadata = self.metadata()?;
        let program = unsafe { super::NpuProgram::load(device, self.directory.join("x.xclbin")) }?;
        let kernel =
            unsafe { program.kernel(&fs::read(self.directory.join("x.bin"))?, metadata.contract) }?;
        Ok((program, kernel))
    }
}
struct Inner {
    tools: Toolchain,
    options: CompilerOptions,
    active: Mutex<usize>,
    available: Condvar,
}
/// A cloneable compiler with bounded workspaces and a verified persistent cache.
#[derive(Clone)]
pub struct Compiler {
    inner: Arc<Inner>,
}
struct Permit<'a>(&'a Inner);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        self.0.available.notify_one();
    }
}
fn relative(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Error::Message(format!(
            "source path must be a relative file: {}",
            path.display()
        )));
    }
    Ok(())
}
impl Compiler {
    /// Verify the pinned installation without initializing a native runtime.
    pub fn new(tools: Toolchain, options: CompilerOptions) -> Result<Self> {
        tools.verify()?;
        if options.workers == 0 || options.timeout.is_zero() {
            return Err(Error::Message(
                "compiler worker count and timeout must be nonzero".into(),
            ));
        }
        fs::create_dir_all(&options.cache)?;
        Ok(Self {
            inner: Arc::new(Inner {
                tools,
                options,
                active: Mutex::new(0),
                available: Condvar::new(),
            }),
        })
    }
    /// Compile or reuse one immutable specialization. Failures preserve build.log.
    pub fn compile(&self, project: &Project) -> Result<Artifact> {
        project.contract.validate()?;
        let mut active = self.inner.active.lock().unwrap_or_else(|e| e.into_inner());
        while *active >= self.inner.options.workers {
            active = self
                .inner
                .available
                .wait(active)
                .unwrap_or_else(|e| e.into_inner());
        }
        *active += 1;
        drop(active);
        let _permit = Permit(&self.inner);
        self.inner.tools.verify()?;
        let entry = match &project.source {
            Source::Iron(path) | Source::Mlir(path) => path,
        };
        let mut inputs = BTreeMap::new();
        for path in std::iter::once(entry).chain(&project.dependencies) {
            relative(path)?;
            inputs.insert(path.clone(), fs::read(project.root.join(path))?);
        }
        let compiler = self.inner.tools.identity()?;
        let input_hashes: BTreeMap<_, _> = inputs
            .iter()
            .map(|(path, bytes)| (path, bundle::digest(bytes)))
            .collect();
        let key = bundle::digest(&serde_json::to_vec(&(
            &compiler,
            &project.source,
            &project.arguments,
            &project.contract,
            input_hashes,
        ))?);
        let cache = &self.inner.options.cache;
        let _lock = bundle::Lock::acquire(&cache.join(format!("{key}.lock")))?;
        let cached = Artifact {
            _temporary: None,
            directory: cache.join(&key),
            key: key.clone(),
        };
        if project.cacheable && cached.verify().is_ok() {
            return Ok(cached);
        }
        let work = tempfile::Builder::new()
            .prefix("compile-")
            .tempdir_in(cache)?;
        let sources = work.path().join("source");
        fs::create_dir(&sources)?;
        for (path, bytes) in inputs {
            let destination = sources.join(path);
            fs::create_dir_all(destination.parent().unwrap())?;
            fs::write(destination, bytes)?;
        }
        let mut command = match &project.source {
            Source::Iron(_) => {
                let mut c = Command::new(&self.inner.tools.python);
                c.arg(entry)
                    .args(&project.arguments)
                    .arg(format!(
                        "--xclbin-path={}",
                        work.path().join("x.xclbin").display()
                    ))
                    .arg(format!(
                        "--insts-path={}",
                        work.path().join("x.bin").display()
                    ));
                c
            }
            Source::Mlir(_) => {
                let mut c = Command::new(&self.inner.tools.aiecc);
                c.args(&project.arguments)
                    .arg("--no-compile-host")
                    .arg(match self.inner.tools.backend {
                        Backend::Peano => "--no-xchesscc",
                        Backend::Chess => "--xchesscc",
                    })
                    .arg("--aie-generate-xclbin")
                    .arg("--aie-generate-npu-insts")
                    .arg(format!(
                        "--xclbin-name={}",
                        work.path().join("x.xclbin").display()
                    ))
                    .arg(format!(
                        "--npu-insts-name={}",
                        work.path().join("x.bin").display()
                    ))
                    .arg(entry);
                c
            }
        };
        command
            .current_dir(&sources)
            .env_clear()
            .envs(&self.inner.tools.environment);
        // IRON delegates tile compiler selection to the project. The recorded
        // backend is explicit and its executable/support files are pinned.
        command.env(
            "HRX_AIE_BACKEND",
            match self.inner.tools.backend {
                Backend::Chess => "chess",
                Backend::Peano => "peano",
            },
        );
        let log = fs::File::create(work.path().join("build.log"))?;
        command
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        use std::os::unix::process::CommandExt;
        command.process_group(0);
        let result = (|| -> Result<()> {
            let mut child = Process {
                child: command.spawn()?,
                reaped: false,
            };
            let started = Instant::now();
            loop {
                if let Some(status) = child.child.try_wait()? {
                    child.reaped = true;
                    if !status.success() {
                        return Err(Error::Message(format!("NPU compiler exited with {status}")));
                    }
                    break;
                }
                if started.elapsed() >= self.inner.options.timeout {
                    return Err(Error::Message("NPU compilation timed out".into()));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            drop(child);
            self.inner.tools.verify()?;
            let instructions = fs::read(work.path().join("x.bin"))?;
            if instructions.is_empty()
                || instructions.len() % 4 != 0
                || fs::metadata(work.path().join("x.xclbin"))?.len() == 0
            {
                return Err(Error::Message(
                    "compiler emitted empty image or malformed instructions".into(),
                ));
            }
            let metadata = Metadata {
                schema: 1,
                key: key.clone(),
                compiler,
                xclbin: bundle::file_digest(&work.path().join("x.xclbin"))?,
                instructions: bundle::digest(&instructions),
                contract: project.contract.clone(),
            };
            fs::write(
                work.path().join("manifest.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            let directory = work.keep();
            return Err(Error::Message(format!(
                "{error}: diagnostics in {}",
                directory.join("build.log").display()
            )));
        }
        // Uncacheable artifacts own their private workspace until the last
        // Artifact clone drops. They never accumulate as persistent cache hits.
        if !project.cacheable {
            return Ok(Artifact {
                directory: work.path().to_path_buf(),
                key,
                _temporary: Some(Arc::new(work)),
            });
        }
        let destination = cached.directory;
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        bundle::publish(work, &destination)?;
        Ok(Artifact {
            directory: destination,
            key,
            _temporary: None,
        })
    }
    /// Compile concurrently up to the configured worker limit, in input order.
    pub fn compile_all(&self, projects: &[Project]) -> Vec<Result<Artifact>> {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Mutex<Vec<Option<Result<Artifact>>>> =
            Mutex::new((0..projects.len()).map(|_| None).collect());
        std::thread::scope(|scope| {
            for _ in 0..self.inner.options.workers.min(projects.len()) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= projects.len() {
                            break;
                        }
                        let result = self.compile(&projects[index]);
                        results.lock().unwrap_or_else(|e| e.into_inner())[index] = Some(result);
                    }
                });
            }
        });
        results
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .into_iter()
            .map(|r| r.expect("worker result"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(root: &Path) -> (Compiler, Project) {
        let tool = root.join("driver");
        fs::write(&tool, b"#!/bin/sh\nexec /bin/sh \"$@\"\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tool, fs::Permissions::from_mode(0o755)).unwrap();
        let source = root.join("generator.sh");
        fs::write(&source, "for arg in \"$@\"; do case $arg in --xclbin-path=*) image=${arg#*=};; --insts-path=*) insts=${arg#*=};; esac; done\nprintf 'image' > \"$image\"\nprintf '1234' > \"$insts\"\n").unwrap();
        let tools = Toolchain {
            python: tool.clone(),
            aiecc: tool.clone(),
            backend: Backend::Peano,
            files: BTreeMap::from([(tool.clone(), bundle::file_digest(&tool).unwrap())]),
            environment: BTreeMap::new(),
        };
        let compiler = Compiler::new(
            tools,
            CompilerOptions {
                workers: 2,
                timeout: Duration::from_secs(2),
                cache: root.join("cache"),
            },
        )
        .unwrap();
        let project = Project {
            root: root.into(),
            source: Source::Iron("generator.sh".into()),
            dependencies: vec![],
            arguments: vec![],
            cacheable: true,
            contract: KernelContract {
                bindings: vec![],
                constants: vec![],
            },
        };
        (compiler, project)
    }
    #[test]
    fn cache_repairs_output_and_invalidates_changed_inputs() {
        let root = tempfile::tempdir().unwrap();
        let (compiler, mut project) = fixture(root.path());
        let first = compiler.compile(&project).unwrap();
        first.verify().unwrap();
        assert_eq!(first.path(), compiler.compile(&project).unwrap().path());
        fs::write(first.path().join("x.bin"), b"bad").unwrap();
        assert!(first.verify().is_err());
        compiler.compile(&project).unwrap().verify().unwrap();
        project.arguments.push("shape=2".into());
        assert_ne!(
            first.identity(),
            compiler.compile(&project).unwrap().identity()
        );
        fs::write(root.path().join("driver"), b"changed compiler").unwrap();
        assert!(compiler.compile(&project).is_err());
    }
    #[test]
    fn concurrent_builds_publish_once_and_uncached_projects_do_not_reuse() {
        let root = tempfile::tempdir().unwrap();
        let (compiler, mut project) = fixture(root.path());
        let results = compiler.compile_all(&[project.clone(), project.clone()]);
        assert_eq!(
            results[0].as_ref().unwrap().path(),
            results[1].as_ref().unwrap().path()
        );
        project.cacheable = false;
        assert_ne!(
            compiler.compile(&project).unwrap().path(),
            compiler.compile(&project).unwrap().path()
        );
        project.dependencies.push("../outside".into());
        assert!(compiler.compile(&project).is_err());
    }
    #[test]
    fn invalid_outputs_and_timeouts_keep_diagnostics() {
        let root = tempfile::tempdir().unwrap();
        let (mut compiler, project) = fixture(root.path());
        for (script, expected) in [
            ("echo missing-output\n", "No such file"),
            ("echo waiting; /bin/sleep 30\n", "timed out"),
        ] {
            fs::write(root.path().join("generator.sh"), script).unwrap();
            Arc::get_mut(&mut compiler.inner).unwrap().options.timeout = Duration::from_millis(100);
            let error = match compiler.compile(&project) {
                Ok(_) => panic!("unexpected compile success"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(expected), "{error}");
            let log = error.split("diagnostics in ").nth(1).unwrap();
            assert!(!fs::read_to_string(log).unwrap().is_empty());
        }
    }
    #[test]
    fn failed_compiles_keep_diagnostics_and_do_not_publish_artifacts() {
        let root = tempfile::tempdir().unwrap();
        let (compiler, project) = fixture(root.path());
        fs::write(
            root.path().join("generator.sh"),
            "echo useful-diagnostic >&2\nexit 7\n",
        )
        .unwrap();
        let error = match compiler.compile(&project) {
            Ok(_) => panic!("unexpected compile success"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("diagnostics in"));
        let log = error.split("diagnostics in ").nth(1).unwrap();
        assert!(
            fs::read_to_string(log)
                .unwrap()
                .contains("useful-diagnostic")
        );
    }
}
