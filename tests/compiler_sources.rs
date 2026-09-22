//! Native compiler qualification; no device or vendor compiler is required.
use hrx::loom::{Compiler, CxxSource, ReportMode, Specialization};

#[test]
#[ignore = "requires the native compiler with C++ import and XDNA enabled"]
fn cpp_headers_and_options_participate_in_identity() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let mut source = CxxSource::new(
        "kernel.cpp",
        r#"
#include <hip/hip_runtime.h>
#include "factor.h"
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fill(float* output) { output[threadIdx.x] = FACTOR; }
"#,
    );
    source
        .headers
        .insert("factor.h".into(), "#define FACTOR 3.0f\n".into());
    let module = compiler.import_cxx(source.clone())?;
    let request = Specialization::new("fill").with_report(ReportMode::Summary);
    let first = module.compile(&request)?;
    let report = first.report().expect("requested compile evidence");
    assert_eq!(report.json()["kind"], "loom.compile_report");
    assert_eq!(report.compiler_identity(), compiler.identity());
    assert!(
        report
            .entries()?
            .iter()
            .any(|entry| entry.code_bytes.is_some_and(|n| n > 0))
    );
    assert_eq!(
        first.report().unwrap().json(),
        module.compile(&request)?.report().unwrap().json()
    );
    source
        .headers
        .insert("factor.h".into(), "#define FACTOR 7.0f\n".into());
    let changed = compiler.import_cxx(source.clone())?;
    assert_ne!(module.identity(), changed.identity());
    assert_ne!(first.bytes(), changed.compile(&request)?.bytes());
    source.headers.clear();
    assert!(compiler.import_cxx(source)?.compile(&request).is_err());
    Ok(())
}

#[test]
#[ignore = "requires the native compiler with XDNA enabled"]
fn xdna_compiles_offline_with_an_exact_device_profile() -> hrx::Result<()> {
    let compiler = Compiler::for_target(None, &hrx::Target::xdna())?;
    let module = compiler.module(include_str!("kernels/mul_i32.xdna.loom"));
    let request = Specialization::new("mul_i32").with_report(ReportMode::Summary);
    let artifact = module.compile(&request)?;
    assert!(artifact.bytes().starts_with(b"\x7fELF"));
    assert_eq!(artifact.path().extension().unwrap(), "xdna");
    assert_eq!(artifact.target(), "amd.xdna.strix_halo.17f0_11");
    assert_eq!(artifact.bytes(), module.compile(&request)?.bytes());
    assert!(artifact.report().is_some());
    Ok(())
}

#[test]
#[ignore = "requires the native compiler with C++ import and XDNA enabled"]
fn report_modes_are_distinct_and_details_survive_cache_hits() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let module = compiler.module(include_str!("kernels/euler.loom"));
    let request = Specialization::new("krea2_euler")
        .with_config("krea2.euler.grid_x", "1")
        .with_config("krea2.euler.grid_y", "1");
    let plain = module.compile(&request)?;
    assert!(plain.report().is_none());
    let summary = module.compile(&request.clone().with_report(ReportMode::Summary))?;
    let detailed_request = request.with_report(ReportMode::Details);
    let details = module.compile(&detailed_request)?;
    assert_ne!(plain.path(), summary.path());
    assert_ne!(summary.path(), details.path());
    assert_eq!(details.report().unwrap().json()["mode"], "details");
    assert_eq!(
        details.report().unwrap().json(),
        module.compile(&detailed_request)?.report().unwrap().json()
    );
    assert!(
        summary
            .report()
            .unwrap()
            .ensure_comparable(details.report().unwrap())
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires native C/C++ importer"]
fn c23_imports_link_with_loom_and_concurrent_cpp_imports_are_isolated() -> hrx::Result<()> {
    use hrx::loom::{CxxStandard, Source};
    let compiler = Compiler::resolve(None)?;
    let mut source = CxxSource::new(
        "src/unit.c",
        r#"
#include "nested/../factor.h"
[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void c_fill(unsigned* output) { output[0] = FACTOR; }
"#,
    );
    source.standard = CxxStandard::C23;
    source
        .headers
        .insert("src/factor.h".into(), "#define FACTOR 13u\n".into());
    let module = compiler.sources(vec![
        Source::Cxx(source),
        Source::loom("euler.loom", include_str!("kernels/euler.loom")),
    ])?;
    assert!(
        module
            .compile(&Specialization::new("c_fill"))?
            .bytes()
            .starts_with(b"\x7fELF")
    );
    assert!(
        module
            .compile(
                &Specialization::new("krea2_euler")
                    .with_config("krea2.euler.grid_x", "1")
                    .with_config("krea2.euler.grid_y", "1")
            )?
            .bytes()
            .starts_with(b"\x7fELF")
    );
    std::thread::scope(|scope| -> hrx::Result<()> {
        let handles: Vec<_> = (0..4)
            .map(|index| {
                let compiler = &compiler;
                scope.spawn(move || -> hrx::Result<String> {
                    let mut source = CxxSource::new(
                        "isolated.cpp",
                        r#"
#include "factor.h"
[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void cpp_fill(unsigned* output) { output[0] = FACTOR; }
"#,
                    );
                    source.headers.insert(
                        "factor.h".into(),
                        format!("#define FACTOR {}u\n", index + 1),
                    );
                    let artifact = compiler
                        .import_cxx(source)?
                        .compile(&Specialization::new("cpp_fill"))?;
                    Ok(hrx::bundle::digest(artifact.bytes()))
                })
            })
            .collect();
        let identities = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<hrx::Result<std::collections::BTreeSet<_>>>()?;
        assert_eq!(identities.len(), 4);
        Ok(())
    })?;
    Ok(())
}

#[test]
#[ignore = "requires native C/C++ importer"]
fn missing_includes_report_the_original_translation_unit() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let source = CxxSource::new("explicit/unit.cpp", "#include \"not-supplied.h\"\n");
    let error = compiler
        .import_cxx(source)?
        .compile(&Specialization::new("missing"))
        .expect_err("unprovided include must fail");
    fn diagnostics(error: &hrx::Error) -> &[hrx::loom::Diagnostic] {
        match error {
            hrx::Error::Compile { diagnostics, .. } => diagnostics,
            hrx::Error::Context { source, .. } => diagnostics(source),
            error => panic!("expected source diagnostics, got {error}"),
        }
    }
    let diagnostics = diagnostics(&error);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.source == "explicit/unit.cpp" && d.line == 1)
    );
    Ok(())
}

#[test]
#[ignore = "requires native C++ importer and XDNA compiler"]
fn cpp_worker_links_into_a_native_xdna_pipeline() -> hrx::Result<()> {
    use hrx::loom::Source;
    let compiler = Compiler::for_target(None, &hrx::Target::xdna())?;
    let pipeline = include_str!("kernels/copy.xdna.loom").replace(
        "vector.store %value,", 
        "%transformed = func.call @transform(%value) : (vector<16xi32>) -> (vector<16xi32>)\n    vector.store %transformed,");
    let pipeline =
        format!("func.decl @transform(%input: vector<16xi32>) -> (vector<16xi32>)\n{pipeline}");
    let worker = CxxSource::new(
        "worker.cpp",
        r#"
typedef int i32x16 __attribute__((vector_size(64)));
extern "C" i32x16 transform(i32x16 input) { return input + 1; }
"#,
    );
    let module = compiler.sources(vec![
        Source::loom("pipeline.loom", pipeline),
        Source::Cxx(worker),
    ])?;
    let artifact = module.compile(&Specialization::new("copy").with_config("copy.packets", "1"))?;
    assert_eq!(artifact.target(), hrx::Target::xdna().as_str());
    assert!(artifact.bytes().starts_with(b"\x7fELF"));
    Ok(())
}
