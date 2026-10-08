use hrx::{
    Error, Result, Target,
    loom::{
        CompileReport, Compiler, CompilerOptions, CxxSource, CxxStandard, ReportMode,
        SanitizerOptions, SanitizerReporting, Specialization, TraceFormat, TraceOptions,
    },
};
use std::{fs, path::Path};

pub const HELP: &str = "hrx compile SOURCE SYMBOL [key=value ...] [--name=value ...]
  --target=gfx1151|amd.xdna.strix_halo.17f0_11
  --language=loom|c23|c++26
  --include=DIR  --define=NAME[=VALUE]  --header=VIRTUAL=FILE
  --report=none|summary|details  --report-output=FILE
  --manifest-output=FILE  --launch-output=FILE
  --trace-output=FILE  --trace-format=text|jsonl  --trace-max-bytes=N
  --trace-before=PATTERN  --trace-after=PATTERN (repeatable)
  --sanitizer=access,value,operation,race
  --sanitizer-reporting=default|trap|report-only
Workload launch programs and sidecar manifests are AMDGPU target products.
Tracing recompiles cache hits; output defaults to a 16 MiB limit.
Sanitizer compilation does not provision runtime globals or structured reporting.";

pub fn compile(args: &[String]) -> Result<()> {
    let path = Path::new(&args[0]);
    let contents = fs::read_to_string(path)?;
    let mut target = Target::default();
    let mut language = match path.extension().and_then(|s| s.to_str()) {
        Some("c") => "c23".to_owned(),
        Some("cpp" | "cc" | "cxx") => "c++26".to_owned(),
        _ => "loom".to_owned(),
    };
    let mut source = CxxSource::new(path.to_string_lossy(), contents.clone());
    let mut request = Specialization::new(&args[1]);
    let mut report_output = None;
    let mut manifest_output = None;
    let mut launch_output = None;
    let mut trace_output = None;
    let mut trace = TraceOptions::default();
    let mut sanitizer = SanitizerOptions::default();
    for arg in &args[2..] {
        let (key, value) = arg.split_once('=').ok_or_else(|| {
            Error::Message("compile options use --name=value; configuration uses key=value".into())
        })?;
        match key {
            "--target" => target = Target::new(value)?,
            "--language" => language = value.into(),
            "--report" => {
                request.set_report(match value {
                    "none" => ReportMode::None,
                    "summary" => ReportMode::Summary,
                    "details" => ReportMode::Details,
                    _ => {
                        return Err(Error::Message(
                            "report mode must be none, summary, or details".into(),
                        ));
                    }
                });
            }
            "--sanitizer" => {
                for check in value.split(',') {
                    match check {
                        "access" => sanitizer.checks.access = true,
                        "value" => sanitizer.checks.value = true,
                        "operation" => sanitizer.checks.operation = true,
                        "race" => sanitizer.checks.race = true,
                        _ => {
                            return Err(Error::Message(
                                "sanitizer checks: access,value,operation,race".into(),
                            ));
                        }
                    }
                }
            }
            "--sanitizer-reporting" => {
                sanitizer.reporting = match value {
                    "default" => SanitizerReporting::Default,
                    "trap" => SanitizerReporting::Trap,
                    "report-only" => SanitizerReporting::ReportOnly,
                    _ => {
                        return Err(Error::Message(
                            "sanitizer reporting: default,trap,report-only".into(),
                        ));
                    }
                }
            }
            "--report-output" => report_output = Some(value),
            "--manifest-output" => manifest_output = Some(value),
            "--launch-output" => launch_output = Some(value),
            "--trace-output" => trace_output = Some(value),
            "--trace-format" => {
                trace.format = match value {
                    "text" => TraceFormat::Text,
                    "jsonl" => TraceFormat::JsonLines,
                    _ => return Err(Error::Message("trace format must be text or jsonl".into())),
                }
            }
            "--trace-max-bytes" => {
                trace.max_bytes = value.parse().map_err(|_| {
                    Error::Message("trace byte limit must be a nonnegative integer".into())
                })?
            }
            "--trace-before" => trace.before.push(value.into()),
            "--trace-after" => trace.after.push(value.into()),
            "--include" => source.include_paths.push(value.into()),
            "--define" => {
                let (name, replacement) = value.split_once('=').unwrap_or((value, "1"));
                source.defines.push((name.into(), replacement.into()));
            }
            "--header" => {
                let (name, file) = value
                    .split_once('=')
                    .ok_or_else(|| Error::Message("--header=virtual/path=FILE".into()))?;
                source
                    .headers
                    .insert(name.into(), fs::read_to_string(file)?);
            }
            _ if key.starts_with('-') => {
                return Err(Error::Message(format!("unknown compile option {key}")));
            }
            _ => {
                request.set_config(key, value);
            }
        }
    }
    if report_output.is_some() && request.report_mode() == ReportMode::None {
        return Err(Error::Message(
            "--report-output requires --report=summary or details".into(),
        ));
    }
    let compiler = Compiler::shared(
        None,
        CompilerOptions {
            target,
            sanitizer,
            ..Default::default()
        },
    )?;
    let module = match language.as_str() {
        "loom" => compiler.sources(vec![hrx::loom::Source::loom(
            path.to_string_lossy(),
            contents,
        )])?,
        "c23" | "c++26" => {
            source.standard = if language == "c23" {
                CxxStandard::C23
            } else {
                CxxStandard::Cpp26
            };
            compiler.import_cxx(source)?
        }
        _ => {
            return Err(Error::Message(
                "language must be loom, c23, or c++26".into(),
            ));
        }
    };
    let artifact = if let Some(path) = trace_output {
        let mut output = fs::File::create(path)?;
        module.compile_traced(&request, &trace, &mut output)?
    } else {
        module.compile(&request)?
    };
    if let Some(path) = manifest_output {
        let manifest = artifact.manifest().ok_or_else(|| {
            Error::Unsupported(
                "this target embeds metadata in its executable and has no sidecar manifest".into(),
            )
        })?;
        fs::write(path, serde_json::to_vec_pretty(manifest)?)?;
    }
    if let Some(path) = launch_output {
        let program = artifact
            .launch_bytes()
            .ok_or_else(|| Error::Unsupported("this target has no host launch program".into()))?;
        fs::write(path, program)?;
    }
    if let Some(path) = report_output {
        let report = artifact
            .report()
            .ok_or_else(|| Error::Message("compiler emitted no requested report".into()))?;
        fs::write(path, serde_json::to_vec_pretty(report)?)?;
    }
    println!("{}", artifact.path().display());
    Ok(())
}

pub fn report(args: &[String]) -> Result<()> {
    let load = |path: &str| -> Result<CompileReport> {
        let report: CompileReport = serde_json::from_slice(&fs::read(path)?)?;
        report.validate()?;
        Ok(report)
    };
    match args {
        [command, path] if command == "show" => {
            let report = load(path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "compiler": report.compiler_identity(), "target": report.target(),
                    "processor_mode": report.processor_mode(), "entries": report.entries()?, "wait_reasons": report.wait_reasons()?,
                    "guidance": report.guidance()?, "expansions": report.expansions()?
                }))?
            );
        }
        [command, before, after] if command == "diff" => {
            let before = load(before)?;
            let after = load(after)?;
            #[derive(serde::Serialize)]
            struct Diff<'a> {
                compiler: &'a str,
                changes: Vec<hrx::loom::EntryChange>,
                wait_reason_changes: Option<Vec<hrx::loom::WaitReasonChange>>,
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&Diff {
                    compiler: before.compiler_identity(),
                    changes: before.changes(&after)?,
                    wait_reason_changes: before.wait_reason_changes(&after)?,
                })?
            );
        }
        _ => {
            return Err(Error::Message(
                "usage: hrx report show FILE | report diff BEFORE AFTER".into(),
            ));
        }
    }
    Ok(())
}
