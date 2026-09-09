//! Ownership boundary for the public Loom C ABI. No runtime/GPU dependency.
use super::ffi::*;
use super::{Diagnostic, Error, Result, Specialization};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Condvar, Mutex, OnceLock},
};

// Keep native code mapped for the process lifetime. In particular, dlopen may
// retain C++ libraries after dlclose, so a weak registry cannot safely associate
// a replacement file with the old loader mapping. Compiler contexts and scratch
// remain session-owned and are released normally.
fn library(path: &Path, identity: &str) -> Result<Arc<Loomc>> {
    struct Loaded {
        identity: Option<String>,
        api: Arc<Loomc>,
    }
    static LIBRARIES: OnceLock<Mutex<HashMap<PathBuf, Loaded>>> = OnceLock::new();
    let mut libraries = LIBRARIES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(loaded) = libraries.get(path) {
        if loaded.identity.as_deref() != Some(identity) {
            return Err(Error::Message(format!(
                "compiler library {} changed after loading; use a new path or restart the process",
                path.display()
            )));
        }
        return Ok(loaded.api.clone());
    }
    // Loading a compiler library trusts the selected native bundle/override.
    let api = Arc::new(unsafe { Loomc::new(path)? });
    // Register even if the post-load check fails: the loader may keep this
    // mapping resident, and a subsequent attempt must not silently reuse it.
    let verified = crate::bundle::file_digest(path).is_ok_and(|actual| actual == identity);
    libraries.insert(
        path.to_path_buf(),
        Loaded {
            identity: verified.then(|| identity.into()),
            api: api.clone(),
        },
    );
    if !verified {
        return Err(Error::Message(
            "compiler library changed while loading".into(),
        ));
    }
    Ok(api)
}

fn view(s: &str) -> loomc_string_view_t {
    loomc_string_view_t {
        data: s.as_ptr().cast(),
        size: s.len(),
    }
}
unsafe fn string(v: loomc_string_view_t) -> String {
    if v.size == 0 {
        return String::new();
    }
    // Native views remain valid while their owning result is retained.
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(v.data.cast(), v.size) })
        .into_owned()
}
fn status(api: &Loomc, value: loomc_status_t) -> Result<()> {
    if value.is_null() {
        return Ok(());
    }
    unsafe {
        let mut size = 0;
        api.loomc_status_format(value, 0, ptr::null_mut(), &mut size);
        let mut bytes = vec![0u8; size.saturating_add(1)];
        api.loomc_status_format(value, bytes.len(), bytes.as_mut_ptr().cast(), &mut size);
        api.loomc_status_free(value);
        Err(Error::Message(
            String::from_utf8_lossy(&bytes[..size.min(bytes.len())]).into_owned(),
        ))
    }
}
struct Handle<T> {
    raw: *mut T,
    api: Arc<Loomc>,
    release: unsafe extern "C" fn(*mut T),
}
impl<T> Drop for Handle<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.release)(self.raw) }
        }
    }
}
fn output<T>(
    api: &Arc<Loomc>,
    release: unsafe extern "C" fn(*mut T),
    call: impl FnOnce(*mut *mut T) -> loomc_status_t,
) -> Result<Handle<T>> {
    let mut h = Handle {
        raw: ptr::null_mut(),
        api: api.clone(),
        release,
    };
    status(api, call(&mut h.raw))?;
    if h.raw.is_null() {
        return Err(Error::Message("Loom returned a null handle".into()));
    }
    Ok(h)
}
fn result_call<T>(
    api: &Arc<Loomc>,
    release: unsafe extern "C" fn(*mut T),
    call: impl FnOnce(*mut *mut T, *mut *mut loomc_result_t) -> loomc_status_t,
) -> Result<(Handle<T>, Vec<Diagnostic>)> {
    let mut value = Handle {
        raw: ptr::null_mut(),
        api: api.clone(),
        release,
    };
    let mut result = Handle {
        raw: ptr::null_mut(),
        api: api.clone(),
        release: api.loomc_result_release,
    };
    status(api, call(&mut value.raw, &mut result.raw))?;
    let diagnostics = result.check()?;
    if value.raw.is_null() {
        return Err(Error::Message(
            "Loom returned no value after a successful result".into(),
        ));
    }
    Ok((value, diagnostics))
}
impl Handle<loomc_result_t> {
    fn check(&self) -> Result<Vec<Diagnostic>> {
        if self.raw.is_null() {
            return Err(Error::Message("Loom returned no result".into()));
        }
        unsafe {
            let diagnostics: Vec<_> = (0..self.api.loomc_result_diagnostic_count(self.raw))
                .map(|i| {
                    let d = &*self.api.loomc_result_diagnostic_at(self.raw, i);
                    Diagnostic {
                        severity: d.severity.into(),
                        code: string(d.code),
                        message: string(d.message),
                        line: d.range.start_line,
                        column: d.range.start_column,
                    }
                })
                .collect();
            if !self.api.loomc_result_succeeded(self.raw) {
                return Err(Error::Compile {
                    message: super::summarize(&diagnostics),
                    diagnostics,
                });
            }
            Ok(diagnostics)
        }
    }
}

pub(super) struct Prepared {
    compiler: Handle<loomc_compiler_t>,
    pipeline: Handle<loomc_pass_program_t>,
    linker: Handle<loomc_linker_t>,
    profile: Handle<loomc_target_profile_t>,
    context: Handle<loomc_context_t>,
    environment: Handle<loomc_target_environment_t>,
    api: Arc<Loomc>,
    pool: Pool,
}
// These handles are immutable after construction; Loom documents concurrent
// compilation/linking with independent modules and exclusive workspaces.
unsafe impl Send for Prepared {}
unsafe impl Sync for Prepared {}
pub(super) struct Index(Handle<loomc_link_index_t>);
// Frozen indexes retain their immutable module storage and support parallel links.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}
struct Workspace(Handle<loomc_workspace_t>);
// A workspace moves between workers, but is exclusively borrowed by one lease.
unsafe impl Send for Workspace {}
struct Pool {
    state: Mutex<PoolState>,
    ready: Condvar,
    limit: usize,
}
struct PoolState {
    idle: Vec<Workspace>,
    count: usize,
}
struct Lease<'a> {
    owner: &'a Prepared,
    workspace: Option<Workspace>,
}
impl Lease<'_> {
    fn raw(&self) -> *mut loomc_workspace_t {
        self.workspace.as_ref().unwrap().0.raw
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut state = self
            .owner
            .pool
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.idle.push(self.workspace.take().unwrap());
        self.owner.pool.ready.notify_one();
    }
}
impl Prepared {
    pub(super) fn open(
        path: &Path,
        identity: &str,
        workers: usize,
        architecture: &crate::Target,
    ) -> Result<Self> {
        let api = library(path, identity)?;
        unsafe {
            let alloc = api.loomc_allocator_system();
            let environment = output(&api, api.loomc_target_environment_release, |out| {
                api.loomc_target_environment_create_amdgpu(alloc, out)
            })?;
            let target = loomc_context_target_options_t {
                type_: LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
                structure_size: size_of::<loomc_context_target_options_t>(),
                target_environment: environment.raw,
                ..Default::default()
            };
            let options = loomc_context_options_t {
                type_: LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
                structure_size: size_of::<loomc_context_options_t>(),
                next: (&target as *const loomc_context_target_options_t).cast(),
            };
            let context = output(&api, api.loomc_context_release, |out| {
                api.loomc_context_create(&options, alloc, out)
            })?;
            let profile_options = loomc_amdgpu_profile_options_t {
                type_: LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
                structure_size: size_of::<loomc_amdgpu_profile_options_t>(),
                identifier: view(architecture.as_str()),
                identity: loomc_amdgpu_target_identity_t {
                    target: view(architecture.as_str()),
                    ..Default::default()
                },
                ..Default::default()
            };
            let profile = output(&api, api.loomc_target_profile_release, |out| {
                api.loomc_target_profile_create_amdgpu(
                    environment.raw,
                    &profile_options,
                    alloc,
                    out,
                )
            })?;
            let linker = output(&api, api.loomc_linker_release, |out| {
                api.loomc_linker_create(context.raw, ptr::null(), alloc, out)
            })?;
            let compiler = output(&api, api.loomc_compiler_release, |out| {
                api.loomc_compiler_create(context.raw, ptr::null(), alloc, out)
            })?;
            let pipeline_options = loomc_target_pipeline_options_t {
                type_: LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
                structure_size: size_of::<loomc_target_pipeline_options_t>(),
                kind: LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
                control_flow_lowering: LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
                source_to_low_max_errors: 20,
                ..Default::default()
            };
            let (pipeline, _) =
                result_call(&api, api.loomc_pass_program_release, |out, result| {
                    api.loomc_pass_program_create_from_target_pipeline(
                        context.raw,
                        &pipeline_options,
                        alloc,
                        out,
                        result,
                    )
                })?;
            Ok(Self {
                compiler,
                pipeline,
                linker,
                profile,
                context,
                environment,
                api,
                pool: Pool {
                    state: Mutex::new(PoolState {
                        idle: Vec::new(),
                        count: 0,
                    }),
                    ready: Condvar::new(),
                    limit: workers,
                },
            })
        }
    }
    fn workspace(&self) -> Result<Lease<'_>> {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(workspace) = state.idle.pop() {
                return Ok(Lease {
                    owner: self,
                    workspace: Some(workspace),
                });
            }
            if state.count < self.pool.limit {
                let handle = unsafe {
                    output(&self.api, self.api.loomc_workspace_release, |out| {
                        self.api.loomc_workspace_create(
                            ptr::null(),
                            self.api.loomc_allocator_system(),
                            out,
                        )
                    })?
                };
                state.count += 1;
                return Ok(Lease {
                    owner: self,
                    workspace: Some(Workspace(handle)),
                });
            }
            state = self
                .pool
                .ready
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
    pub(super) fn trim(&self) {
        let state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        for workspace in &state.idle {
            unsafe { self.api.loomc_workspace_trim(workspace.0.raw) }
        }
    }
    pub(super) fn index(&self, source: &str) -> Result<(Index, Vec<Diagnostic>)> {
        let api = &self.api;
        unsafe {
            let options = loomc_source_options_t {
                type_: LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
                structure_size: size_of::<loomc_source_options_t>(),
                format: LOOMC_SOURCE_FORMAT_TEXT,
                identifier: view("kernel.loom"),
                contents: loomc_byte_span_t {
                    data: source.as_ptr(),
                    data_length: source.len(),
                },
                storage: LOOMC_SOURCE_STORAGE_COPY,
                ..Default::default()
            };
            let source = output(api, api.loomc_source_release, |out| {
                api.loomc_source_create(&options, api.loomc_allocator_system(), out)
            })?;
            let builder = output(api, api.loomc_link_index_builder_release, |out| {
                api.loomc_link_index_builder_create(
                    self.context.raw,
                    ptr::null(),
                    api.loomc_allocator_system(),
                    out,
                )
            })?;
            status(
                api,
                api.loomc_link_index_builder_add_source(
                    builder.raw,
                    source.raw,
                    ptr::null(),
                    ptr::null_mut(),
                ),
            )?;
            let (index, diagnostics) =
                result_call(api, api.loomc_link_index_release, |out, result| {
                    api.loomc_link_index_builder_finish(builder.raw, out, result)
                })?;
            Ok((Index(index), diagnostics))
        }
    }
    pub(super) fn compile(&self, index: &Index, spec: &Specialization) -> Result<super::Compiled> {
        let lease = self.workspace()?;
        let api = &self.api;
        unsafe {
            let alloc = api.loomc_allocator_system();
            let root = view(&spec.symbol);
            let config: Vec<_> = spec
                .config
                .iter()
                .map(|(k, v)| loomc_config_binding_t {
                    key: view(k),
                    value: view(v),
                })
                .collect();
            let options = loomc_link_options_t {
                type_: LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
                structure_size: size_of::<loomc_link_options_t>(),
                link_index: index.0.raw,
                module_name: view("kernel"),
                mode: LOOMC_LINK_MODE_LINK,
                flags: LOOMC_LINK_FLAG_ALLOW_UNRESOLVED_SYMBOLS,
                root_symbols: &root,
                root_symbol_count: 1,
                config: loomc_config_options_t {
                    bindings: config.as_ptr(),
                    binding_count: config.len(),
                    flags: LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
                    ..Default::default()
                },
                ..Default::default()
            };
            let (module, mut diagnostics) =
                result_call(api, api.loomc_module_release, |out, result| {
                    api.loomc_link_module(self.linker.raw, lease.raw(), &options, out, result)
                })
                .map_err(|e| e.context("linking export"))?;
            let specialization = loomc_target_specialization_t {
                function_symbol: root,
                target_profile: self.profile.raw,
            };
            let targets = loomc_target_specialization_options_t {
                type_: LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
                structure_size: size_of::<loomc_target_specialization_options_t>(),
                specializations: &specialization,
                specialization_count: 1,
                ..Default::default()
            };
            let options = loomc_compile_options_t {
                type_: LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
                structure_size: size_of::<loomc_compile_options_t>(),
                next: (&targets as *const loomc_target_specialization_options_t).cast(),
                module_name: view("kernel"),
                ..Default::default()
            };
            let result = output(api, api.loomc_result_release, |out| {
                api.loomc_compile_module(
                    self.compiler.raw,
                    lease.raw(),
                    self.pipeline.raw,
                    module.raw,
                    &options,
                    alloc,
                    out,
                )
            })?;
            diagnostics.extend(result.check().map_err(|e| e.context("lowering export"))?);
            drop(result);
            let manifest = loomc_artifact_manifest_options_t {
                type_: LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
                structure_size: size_of::<loomc_artifact_manifest_options_t>(),
                mode: LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS,
                ..Default::default()
            };
            let options = loomc_emit_options_t {
                type_: LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
                structure_size: size_of::<loomc_emit_options_t>(),
                next: if spec.report {
                    (&manifest as *const loomc_artifact_manifest_options_t).cast()
                } else {
                    ptr::null()
                },
                artifact_format: view("amdgpu-hsaco"),
                artifact_flags: LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
                ..Default::default()
            };
            let result = output(api, api.loomc_result_release, |out| {
                api.loomc_emit_module(
                    self.environment.raw,
                    lease.raw(),
                    module.raw,
                    &options,
                    alloc,
                    out,
                )
            })?;
            diagnostics.extend(result.check()?);
            let mut bytes = None;
            let mut report = None;
            for i in 0..api.loomc_result_artifact_count(result.raw) {
                let artifact = &*api.loomc_result_artifact_at(result.raw, i);
                let mut span = loomc_byte_span_t::default();
                status(
                    api,
                    api.loomc_byte_sequence_clone(artifact.contents, alloc, &mut span),
                )?;
                let data = if span.data_length == 0 {
                    Arc::<[u8]>::from([])
                } else {
                    Arc::<[u8]>::from(std::slice::from_raw_parts(span.data, span.data_length))
                };
                api.loomc_allocator_free(alloc, span.data.cast_mut().cast());
                if string(artifact.format) == "amdgpu-hsaco" {
                    bytes = Some(data);
                } else if artifact.kind == LOOMC_ARTIFACT_KIND_REPORT {
                    report = Some(serde_json::from_slice(&data)?);
                }
            }
            let bytes = bytes
                .filter(|v| !v.is_empty())
                .ok_or_else(|| Error::Message("Loom emitted no AMDGPU artifact".into()))?;
            Ok(super::Compiled {
                bytes,
                diagnostics,
                report,
            })
        }
    }
}
