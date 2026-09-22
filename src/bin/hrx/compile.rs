use hrx::{
    Error, Result, Target,
    loom::{CompileReport, Compiler, CxxSource, CxxStandard, ReportMode, Specialization},
};
use std::{fs, path::Path};

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
            "--report-output" => report_output = Some(value),
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
    let compiler = Compiler::for_target(None, &target)?;
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
    let artifact = module.compile(&request)?;
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
                    "processor_mode": report.processor_mode(), "entries": report.entries()?
                }))?
            );
        }
        [command, before, after] if command == "diff" => {
            let before = load(before)?;
            let after = load(after)?;
            before.ensure_comparable(&after)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "compiler": before.compiler_identity(),
                    "changes": before.changes(&after)?
                }))?
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
