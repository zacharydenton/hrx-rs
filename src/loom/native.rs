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
// The caller retains the native result until every borrowed field is copied.
unsafe fn copy_diagnostic(
    raw: *const loomc_diagnostic_t,
    source_identifier: impl Fn(*const loomc_source_t) -> String,
) -> Diagnostic {
    unsafe {
        let d = &*raw;
        let related_locations = (0..d.related_location_count)
            .map(|j| {
                let location = &*d.related_locations.add(j);
                let range = &location.range;
                super::RelatedLocation {
                    label: string(location.label),
                    source: if range.source.is_null() {
                        String::new()
                    } else {
                        source_identifier(range.source)
                    },
                    line: range.start_line,
                    column: range.start_column,
                    end_line: range.end_line,
                    end_column: range.end_column,
                }
            })
            .collect();
        Diagnostic {
            severity: d.severity.into(),
            code: string(d.code),
            message: string(d.message),
            formatted_text: string(d.formatted_text),
            parameters: (0..d.parameter_count)
                .map(|i| {
                    let parameter = &*d.parameters.add(i);
                    super::DiagnosticParameter {
                        name: string(parameter.name),
                        value: string(parameter.value),
                    }
                })
                .collect(),
            source: if d.range.source.is_null() {
                String::new()
            } else {
                source_identifier(d.range.source)
            },
            line: d.range.start_line,
            column: d.range.start_column,
            related_locations,
            related_location_omitted_count: d.related_location_omitted_count,
        }
    }
}

impl Handle<loomc_result_t> {
    fn check(&self) -> Result<Vec<Diagnostic>> {
        if self.raw.is_null() {
            return Err(Error::Message("Loom returned no result".into()));
        }
        unsafe {
            let diagnostics: Vec<_> = (0..self.api.loomc_result_diagnostic_count(self.raw))
                .map(|i| {
                    let raw = self.api.loomc_result_diagnostic_at(self.raw, i);
                    copy_diagnostic(raw, |source| {
                        string(self.api.loomc_source_identifier(source))
                    })
                })
                .collect();
            if !self.api.loomc_result_succeeded(self.raw) {
                return Err(Error::Compile {
                    message: super::summarize(&diagnostics),
                    diagnostics,
                    report: None,
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
    api: Arc<Loomc>,
    pool: Pool,
    artifact_format: &'static str,
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
        processor_mode: super::ProcessorMode,
        sanitizer: super::SanitizerOptions,
    ) -> Result<Self> {
        if architecture.is_xdna() && processor_mode != super::ProcessorMode::Default {
            return Err(Error::Message("CU/WGP mode applies only to AMDGPU".into()));
        }
        let api = library(path, identity)?;
        unsafe {
            let alloc = api.loomc_allocator_system();
            let environment = output(&api, api.loomc_target_environment_release, |out| {
                if architecture.is_xdna() {
                    api.loomc_target_environment_create_xdna(alloc, out)
                } else {
                    api.loomc_target_environment_create_amdgpu(alloc, out)
                }
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
                source_retention: LOOMC_SOURCE_RETENTION_METADATA_ONLY,
            };
            let context = output(&api, api.loomc_context_release, |out| {
                api.loomc_context_create(&options, alloc, out)
            })?;
            let execution = loomc_amdgpu_profile_execution_options_t {
                type_: LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_EXECUTION_OPTIONS,
                structure_size: size_of::<loomc_amdgpu_profile_execution_options_t>(),
                processor_mode: match processor_mode {
                    super::ProcessorMode::Default => LOOMC_AMDGPU_PROCESSOR_MODE_DEFAULT,
                    super::ProcessorMode::ComputeUnit => LOOMC_AMDGPU_PROCESSOR_MODE_CU,
                    super::ProcessorMode::WorkgroupProcessor => LOOMC_AMDGPU_PROCESSOR_MODE_WGP,
                },
                ..Default::default()
            };
            let profile_options = loomc_amdgpu_profile_options_t {
                type_: LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
                structure_size: size_of::<loomc_amdgpu_profile_options_t>(),
                next: if processor_mode == super::ProcessorMode::Default {
                    ptr::null()
                } else {
                    (&execution as *const loomc_amdgpu_profile_execution_options_t).cast()
                },
                identifier: view(architecture.as_str()),
                identity: loomc_amdgpu_target_identity_t {
                    target: view(architecture.as_str()),
                    ..Default::default()
                },
            };
            let profile = output(&api, api.loomc_target_profile_release, |out| {
                if architecture.is_xdna() {
                    api.loomc_target_profile_create_xdna(
                        environment.raw,
                        view(architecture.as_str()),
                        alloc,
                        out,
                    )
                } else {
                    api.loomc_target_profile_create_amdgpu(
                        environment.raw,
                        &profile_options,
                        alloc,
                        out,
                    )
                }
            })?;
            let linker = output(&api, api.loomc_linker_release, |out| {
                api.loomc_linker_create(context.raw, ptr::null(), alloc, out)
            })?;
            let compiler = output(&api, api.loomc_compiler_release, |out| {
                api.loomc_compiler_create(context.raw, ptr::null(), alloc, out)
            })?;
            let sanitizer_options = loomc_sanitizer_options_t {
                type_: LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                structure_size: size_of::<loomc_sanitizer_options_t>(),
                checks: sanitizer.checks.bits(),
                reporting_mode: match sanitizer.reporting {
                    super::SanitizerReporting::Default => LOOMC_SANITIZER_REPORTING_MODE_DEFAULT,
                    super::SanitizerReporting::Trap => LOOMC_SANITIZER_REPORTING_MODE_TRAP,
                    super::SanitizerReporting::ReportOnly => {
                        LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY
                    }
                },
                ..Default::default()
            };
            let pipeline_options = loomc_target_pipeline_options_t {
                next: (&sanitizer_options as *const loomc_sanitizer_options_t).cast(),
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
                api,
                artifact_format: architecture.artifact_format(),
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
    unsafe fn source(
        &self,
        identifier: &str,
        contents: &str,
        format: u32,
    ) -> Result<Handle<loomc_source_t>> {
        let options = loomc_source_options_t {
            type_: LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
            structure_size: size_of::<loomc_source_options_t>(),
            format,
            identifier: view(identifier),
            contents: loomc_byte_span_t {
                data: contents.as_ptr(),
                data_length: contents.len(),
            },
            storage: LOOMC_SOURCE_STORAGE_COPY,
            ..Default::default()
        };
        output(&self.api, self.api.loomc_source_release, |out| unsafe {
            self.api
                .loomc_source_create(&options, self.api.loomc_allocator_system(), out)
        })
    }
    pub(super) fn index(&self, sources: &[super::Source]) -> Result<(Index, Vec<Diagnostic>)> {
        let api = &self.api;
        unsafe {
            let builder = output(api, api.loomc_link_index_builder_release, |out| {
                api.loomc_link_index_builder_create(
                    self.context.raw,
                    ptr::null(),
                    api.loomc_allocator_system(),
                    out,
                )
            })?;
            let mut diagnostics = Vec::new();
            for source in sources {
                let source = match source {
                    super::Source::Loom {
                        identifier,
                        contents,
                    } => self.source(identifier, contents, LOOMC_SOURCE_FORMAT_TEXT)?,
                    super::Source::Cxx(unit) => {
                        let lease = self.workspace()?;
                        let input = self.source(
                            &unit.identifier,
                            &unit.contents,
                            LOOMC_SOURCE_FORMAT_UNKNOWN,
                        )?;
                        let mut provider = IncludeProvider {
                            api,
                            headers: &unit.headers,
                        };
                        let paths: Vec<_> = unit.include_paths.iter().map(|s| view(s)).collect();
                        let defines: Vec<_> = unit
                            .defines
                            .iter()
                            .map(|(name, value)| loomc_cxx_define_t {
                                name: view(name),
                                value: view(value),
                            })
                            .collect();
                        let roots: Vec<_> = unit.roots.iter().map(|s| view(s)).collect();
                        let options = loomc_cxx_import_options_t {
                            type_: LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS,
                            structure_size: size_of::<loomc_cxx_import_options_t>(),
                            standard: view(unit.standard.as_str()),
                            triple: view("x86_64-unknown-linux-gnu"),
                            data_model: LOOMC_CXX_DATA_MODEL_LP64,
                            flags: if unit.approximate_functions {
                                LOOMC_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS
                            } else {
                                0
                            },
                            source_provider: loomc_cxx_source_provider_t {
                                fn_: Some(include_source),
                                user_data: (&mut provider as *mut IncludeProvider<'_>).cast(),
                            },
                            include_paths: paths.as_ptr(),
                            include_path_count: paths.len(),
                            defines: defines.as_ptr(),
                            define_count: defines.len(),
                            roots: roots.as_ptr(),
                            root_count: roots.len(),
                            ..Default::default()
                        };
                        let (module, imported) =
                            result_call(api, api.loomc_module_release, |out, result| {
                                api.loomc_module_import_cxx(
                                    self.context.raw,
                                    lease.raw(),
                                    input.raw,
                                    &options,
                                    api.loomc_allocator_system(),
                                    out,
                                    result,
                                )
                            })
                            .map_err(|e| e.context(format!("importing {}", unit.identifier)))?;
                        diagnostics.extend(imported);
                        let options = loomc_module_serialize_options_t {
                            type_: LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
                            structure_size: size_of::<loomc_module_serialize_options_t>(),
                            format: LOOMC_SOURCE_FORMAT_BYTECODE,
                            identifier: view(&unit.identifier),
                            ..Default::default()
                        };
                        output(api, api.loomc_source_release, |out| {
                            api.loomc_module_serialize_to_source(
                                module.raw,
                                &options,
                                api.loomc_allocator_system(),
                                out,
                            )
                        })?
                    }
                };
                status(
                    api,
                    api.loomc_link_index_builder_add_source(
                        builder.raw,
                        source.raw,
                        ptr::null(),
                        ptr::null_mut(),
                    ),
                )?;
            }
            let (index, indexed) =
                result_call(api, api.loomc_link_index_release, |out, result| {
                    api.loomc_link_index_builder_finish(builder.raw, out, result)
                })?;
            diagnostics.extend(indexed);
            Ok((Index(index), diagnostics))
        }
    }
    pub(super) fn compile(
        &self,
        index: &Index,
        spec: &Specialization,
        trace: Option<(&super::TraceOptions, &mut dyn std::io::Write)>,
    ) -> Result<super::Compiled> {
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
            let manifest_options = loomc_artifact_manifest_options_t {
                type_: LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
                structure_size: size_of::<loomc_artifact_manifest_options_t>(),
                mode: if self.artifact_format == "amdgpu-hsaco" {
                    LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS
                } else {
                    LOOMC_ARTIFACT_MANIFEST_MODE_NONE
                },
                ..Default::default()
            };
            let report_options = loomc_compile_report_options_t {
                type_: LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
                structure_size: size_of::<loomc_compile_report_options_t>(),
                next: (&manifest_options as *const loomc_artifact_manifest_options_t).cast(),
                mode: match spec.report {
                    super::ReportMode::None => LOOMC_COMPILE_REPORT_MODE_NONE,
                    super::ReportMode::Summary => LOOMC_COMPILE_REPORT_MODE_SUMMARY,
                    super::ReportMode::Details => LOOMC_COMPILE_REPORT_MODE_DETAILS,
                },
                ..Default::default()
            };
            let emit_options = loomc_emit_options_t {
                type_: LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
                structure_size: size_of::<loomc_emit_options_t>(),
                next: (&report_options as *const loomc_compile_report_options_t).cast(),
                artifact_format: view(self.artifact_format),
                artifact_flags: LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
                ..Default::default()
            };
            let mut trace_state = trace.map(|(options, writer)| TraceSink {
                api,
                options,
                writer,
                written: 0,
                error: None,
            });
            let before: Vec<_> = trace_state
                .as_ref()
                .map(|state| state.options.before.iter().map(|s| view(s)).collect())
                .unwrap_or_default();
            let after: Vec<_> = trace_state
                .as_ref()
                .map(|state| state.options.after.iter().map(|s| view(s)).collect())
                .unwrap_or_default();
            let trace_options = loomc_pass_trace_options_t {
                type_: LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
                structure_size: size_of::<loomc_pass_trace_options_t>(),
                format: match trace_state.as_ref().map(|state| state.options.format) {
                    Some(super::TraceFormat::Text) => LOOMC_PASS_TRACE_FORMAT_TEXT,
                    _ => LOOMC_PASS_TRACE_FORMAT_JSONL,
                },
                flags: if before.is_empty() && after.is_empty() {
                    LOOMC_PASS_TRACE_FLAG_AFTER_ALL
                } else {
                    0
                },
                tool_name: view("hrx-rs"),
                input_identifier: root,
                before_filters: before.as_ptr(),
                before_filter_count: before.len(),
                after_filters: after.as_ptr(),
                after_filter_count: after.len(),
                sink: loomc_pass_trace_sink_t {
                    write: Some(write_trace),
                    user_data: trace_state.as_mut().map_or(ptr::null_mut(), |state| {
                        (state as *mut TraceSink<'_>).cast()
                    }),
                },
                ..Default::default()
            };
            let options = loomc_compile_artifact_options_t {
                type_: LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
                structure_size: size_of::<loomc_compile_artifact_options_t>(),
                next: if trace_state.is_some() {
                    (&trace_options as *const loomc_pass_trace_options_t).cast()
                } else {
                    ptr::null()
                },
                roots: &root,
                root_count: 1,
                target_profile: self.profile.raw,
                emit_options: &emit_options,
                artifact_flags: if self.artifact_format == "amdgpu-hsaco" {
                    LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG
                } else {
                    0
                },
                ..Default::default()
            };
            let result = output(api, api.loomc_result_release, |out| {
                api.loomc_compile_artifact(
                    self.compiler.raw,
                    lease.raw(),
                    self.pipeline.raw,
                    module.raw,
                    &options,
                    alloc,
                    out,
                )
            });
            if let Some(error) = trace_state.and_then(|state| state.error) {
                return Err(error.context("writing compiler pass trace"));
            }
            let result = result?;
            let checked = result.check();
            let mut bytes = None;
            let mut report = None;
            let mut manifest = None;
            let mut launch_bytes = None;
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
                if artifact.kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG {
                    launch_bytes = Some(data);
                } else if string(artifact.format) == "loom-artifact-manifest-json" {
                    manifest = Some(serde_json::from_slice(&data)?);
                } else if string(artifact.format) == self.artifact_format {
                    bytes = Some(data);
                } else if string(artifact.format) == "loom-compile-report-json" {
                    report = Some(serde_json::from_slice(&data)?);
                }
            }
            match checked {
                Ok(found) => diagnostics.extend(found),
                Err(Error::Compile {
                    message,
                    diagnostics: failed,
                    ..
                }) => {
                    diagnostics.extend(failed);
                    return Err(Error::Compile {
                        message,
                        diagnostics,
                        report: report.map(Box::new),
                    });
                }
                Err(error) => return Err(error),
            }
            if manifest_options.mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE && manifest.is_none() {
                return Err(Error::Message("Loom emitted no artifact manifest".into()));
            }
            let launch_bytes = launch_bytes.filter(|v| !v.is_empty());
            if options.artifact_flags != 0 && launch_bytes.is_none() {
                return Err(Error::Message("Loom emitted no launch program".into()));
            }
            let bytes = bytes.filter(|v| !v.is_empty()).ok_or_else(|| {
                Error::Message(format!("Loom emitted no {} artifact", self.artifact_format))
            })?;
            Ok(super::Compiled {
                bytes,
                diagnostics,
                report,
                manifest,
                launch_bytes,
            })
        }
    }
}

struct IncludeProvider<'a> {
    api: &'a Loomc,
    headers: &'a std::collections::BTreeMap<String, String>,
}
// The importer invokes this callback synchronously and retains no borrowed
// strings or callback state. A missing virtual header never consults the host.
unsafe extern "C" fn include_source(
    data: *mut std::ffi::c_void,
    path: loomc_string_view_t,
    out: *mut *mut loomc_source_t,
) -> loomc_status_t {
    unsafe {
        *out = ptr::null_mut();
        let provider = &*data.cast::<IncludeProvider<'_>>();
        let path_string = string(path);
        let Ok(key) = super::source::virtual_path(&path_string) else {
            return ptr::null_mut();
        };
        let Some(contents) = provider.headers.get(&key) else {
            return ptr::null_mut();
        };
        let options = loomc_source_options_t {
            type_: LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
            structure_size: size_of::<loomc_source_options_t>(),
            format: LOOMC_SOURCE_FORMAT_UNKNOWN,
            identifier: path,
            contents: loomc_byte_span_t {
                data: contents.as_ptr(),
                data_length: contents.len(),
            },
            storage: LOOMC_SOURCE_STORAGE_COPY,
            ..Default::default()
        };
        provider
            .api
            .loomc_source_create(&options, provider.api.loomc_allocator_system(), out)
    }
}

// Unlike compiler handles, invocation scratch is exclusive to this owner.
pub(super) struct LaunchProgram {
    program: Handle<loomc_launch_config_program_t>,
    function: loomc_launch_config_function_t,
}
unsafe impl Send for LaunchProgram {}
#[derive(Clone)]
pub(super) struct LaunchLoader(Arc<Loomc>);
impl std::fmt::Debug for LaunchLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LaunchLoader")
    }
}
impl Prepared {
    pub(super) fn launch_loader(&self) -> LaunchLoader {
        LaunchLoader(self.api.clone())
    }
}
impl LaunchLoader {
    pub(super) fn load(&self, bytes: &[u8], export: &str) -> Result<super::LaunchProgram> {
        let api = &self.0;
        unsafe {
            let alloc = api.loomc_allocator_system();
            let sequence = output(api, api.loomc_byte_sequence_release, |out| {
                api.loomc_byte_sequence_create_copy(
                    loomc_byte_span_t {
                        data: bytes.as_ptr(),
                        data_length: bytes.len(),
                    },
                    alloc,
                    out,
                )
            })?;
            let artifact = loomc_artifact_t {
                kind: LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                format: view("loombc"),
                contents: sequence.raw,
                ..Default::default()
            };
            let program = output(api, api.loomc_launch_config_program_release, |out| {
                api.loomc_launch_config_program_load(&artifact, alloc, out)
            })?;
            let mut function = loomc_launch_config_function_t::default();
            status(
                api,
                api.loomc_launch_config_program_lookup_function(
                    program.raw,
                    view(export),
                    &mut function,
                ),
            )?;
            Ok(super::LaunchProgram(LaunchProgram { program, function }))
        }
    }
}
impl LaunchProgram {
    pub(super) fn evaluate(&mut self, arguments: &[u64]) -> Result<super::LaunchConfig> {
        let mut config = loomc_launch_config_t {
            type_: LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
            structure_size: size_of::<loomc_launch_config_t>(),
            ..Default::default()
        };
        unsafe {
            status(
                &self.program.api,
                self.program.api.loomc_launch_config_program_invoke(
                    self.program.raw,
                    self.function,
                    arguments.as_ptr(),
                    arguments.len(),
                    &mut config,
                ),
            )?;
        }
        let dimensions = |d: loomc_dimension3_t| [d.x, d.y, d.z];
        Ok(super::LaunchConfig {
            workgroup_count: dimensions(config.workgroup_count),
            workgroup_size: dimensions(config.workgroup_size),
            workgroup_cluster_size: dimensions(config.workgroup_cluster_size),
            subgroup_size: config.subgroup_size,
            workgroup_storage_bytes: config.workgroup_storage_bytes,
        })
    }
}

struct TraceSink<'a> {
    api: &'a Loomc,
    options: &'a super::TraceOptions,
    writer: &'a mut dyn std::io::Write,
    written: usize,
    error: Option<Error>,
}
unsafe extern "C" fn write_trace(
    user_data: *mut std::ffi::c_void,
    fragment: loomc_string_view_t,
) -> loomc_status_t {
    let state = unsafe { &mut *user_data.cast::<TraceSink<'_>>() };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        let total = state
            .written
            .checked_add(fragment.size)
            .filter(|n| *n <= state.options.max_bytes)
            .ok_or_else(|| Error::Message("compiler trace byte limit exceeded".into()))?;
        if fragment.size != 0 {
            let bytes =
                unsafe { std::slice::from_raw_parts(fragment.data.cast::<u8>(), fragment.size) };
            state.writer.write_all(bytes)?;
        }
        state.written = total;
        Ok(())
    }));
    match outcome {
        Ok(Ok(())) => ptr::null_mut(),
        failure => {
            state.error = Some(match failure {
                Ok(Err(error)) => error,
                _ => Error::Message("compiler trace writer panicked".into()),
            });
            unsafe {
                state.api.loomc_status_allocate(
                    LOOMC_STATUS_ABORTED,
                    c"hrx-rs".as_ptr(),
                    0,
                    view("compiler trace sink failed"),
                )
            }
        }
    }
}

#[cfg(test)]
mod diagnostic_abi_tests {
    use super::*;

    #[test]
    fn diagnostic_locations_outlive_native_result_storage() {
        let copied = {
            let label = String::from("previous declaration");
            let locations = [loomc_diagnostic_related_location_t {
                label: view(&label),
                range: loomc_source_range_t {
                    source: std::ptr::dangling(),
                    start_line: 4,
                    start_column: 2,
                    end_line: 4,
                    end_column: 9,
                    ..Default::default()
                },
            }];
            let parameter_value = String::from("64");
            let parameters = [loomc_diagnostic_parameter_t {
                name: view("width"),
                value: view(&parameter_value),
            }];
            let diagnostic = loomc_diagnostic_t {
                severity: 2,
                parameters: parameters.as_ptr(),
                parameter_count: parameters.len(),
                code: view("duplicate"),
                message: view("duplicate symbol"),
                formatted_text: view("header.loom:7:3: duplicate symbol\n"),
                range: loomc_source_range_t {
                    source: std::ptr::dangling(),
                    start_line: 7,
                    start_column: 3,
                    ..Default::default()
                },
                related_locations: locations.as_ptr(),
                related_location_count: 1,
                related_location_omitted_count: 3,
            };
            unsafe { copy_diagnostic(&diagnostic, |_| String::from("header.loom")) }
        };
        assert_eq!(copied.formatted_text, "header.loom:7:3: duplicate symbol\n");
        assert_eq!(
            copied.parameters,
            vec![super::super::DiagnosticParameter {
                name: "width".into(),
                value: "64".into()
            }]
        );
        assert_eq!(copied.source, "header.loom");
        assert_eq!(copied.line, 7);
        assert_eq!(copied.column, 3);
        assert_eq!(copied.message, "duplicate symbol");
        assert_eq!(copied.related_locations[0].label, "previous declaration");
        assert_eq!(copied.related_locations[0].source, "header.loom");
        assert_eq!(copied.related_locations[0].end_column, 9);
        assert_eq!(copied.related_location_omitted_count, 3);
        assert!(
            super::super::summarize(&[copied]).contains("header.loom:4:2: previous declaration")
        );
    }
}
