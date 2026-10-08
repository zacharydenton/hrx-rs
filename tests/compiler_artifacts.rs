//! Owned compiler companion products and bounded diagnostic streaming.
use hrx::loom::{Compiler, CxxSource, ReportMode, Specialization, TraceOptions};

#[test]
#[ignore = "requires the native compiler and gfx1151"]
fn bounded_extrema_materialize_wide_kernel_arguments() -> hrx::Result<()> {
    // The unbounded comparison keeps count's ABI carrier at 64 bits. The
    // separate assumption proves both extrema fit the native signed-i32 rule.
    let artifact = Compiler::resolve(None)?
        .module(include_str!("kernels/bounded_extrema.loom"))
        .compile(&Specialization::new("bounded_extrema"))?;
    let mut stream = hrx::Stream::open()?;
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    for count in [1u32, 63, 64, 65, 129, 1025] {
        let expected: Vec<u32> = (0..count)
            .flat_map(|row| {
                if row % 64 == 0 {
                    [(row + 64).min(count), (row + 64).max(count)]
                } else {
                    [0xabababab; 2]
                }
            })
            .collect();
        let output = stream.allocate(expected.len() * 4)?;
        stream.fill(output.binding(), 0xab)?;
        let launch = kernel.launch_config(&[u64::from(count)])?;
        let constants = hrx::Constants::indices(&kernel, &[count])?;
        // Each workgroup writes only its first row; the rest are guards.
        unsafe {
            stream.dispatch(
                &kernel,
                launch.workgroup_count,
                launch.workgroup_size,
                &constants,
                &[output.binding()],
            )?;
        }
        let actual = stream.read(output.binding())?.wait(&mut stream)?;
        let expected: Vec<u8> = expected.into_iter().flat_map(u32::to_le_bytes).collect();
        assert_eq!(actual, expected, "count={count}");
    }
    Ok(())
}

fn source() -> CxxSource {
    CxxSource::new(
        "launch.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(3, 2, 1)]]
void fill(float* output) { output[threadIdx.x] = 3.0f; }
"#,
    )
}

#[test]
#[ignore = "requires the native compiler"]
fn companions_survive_cache_and_compiler_owners() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let module = compiler.import_cxx(source())?;
    let request = Specialization::new("fill").with_report(ReportMode::Summary);
    let artifact = module.compile(&request)?;
    let cached = module.compile(&request)?;
    assert_eq!(artifact.manifest(), cached.manifest());
    assert!(artifact.manifest().is_some());
    assert_eq!(artifact.launch_bytes(), cached.launch_bytes());
    assert!(!artifact.launch_bytes().unwrap().is_empty());
    assert!(artifact.launch_program("missing").is_err());
    let mut first = artifact.launch_program("fill")?;
    let mut second = cached.launch_program("fill")?;
    drop((module, artifact, cached, compiler));
    let a = std::thread::spawn(move || first.evaluate(&[]));
    let b = second.evaluate(&[])?;
    assert_eq!(a.join().unwrap()?, b);
    assert_eq!(b.workgroup_count, [3, 2, 1]);
    assert_eq!(b.workgroup_size, [64, 1, 1]);
    assert_eq!(b.workgroup_cluster_size, [1; 3]);
    assert!(second.evaluate(&[1]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires the native compiler"]
fn tracing_recompiles_cache_hits_and_failure_preserves_cache() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let module = compiler.import_cxx(source())?;
    let request = Specialization::new("fill");
    let baseline = module.compile(&request)?;
    let mut trace = Vec::new();
    let traced = module.compile_traced(&request, &TraceOptions::default(), &mut trace)?;
    assert_eq!(baseline.bytes(), traced.bytes());
    assert!(!trace.is_empty());
    for line in trace
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let _: serde_json::Value = serde_json::from_slice(line)?;
    }
    let limited = TraceOptions {
        max_bytes: 1,
        ..Default::default()
    };
    trace.clear();
    let error = module
        .compile_traced(&request, &limited, &mut trace)
        .unwrap_err();
    assert!(error.to_string().contains("byte limit"));
    assert!(trace.len() <= 1);
    assert_eq!(baseline.bytes(), module.compile(&request)?.bytes());
    struct Panics;
    impl std::io::Write for Panics {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            panic!("sink panic")
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    assert!(
        module
            .compile_traced(&request, &TraceOptions::default(), &mut Panics)
            .unwrap_err()
            .to_string()
            .contains("panicked")
    );
    assert_eq!(baseline.bytes(), module.compile(&request)?.bytes());
    Ok(())
}

#[test]
#[ignore = "requires the native compiler"]
fn corrupt_launch_companion_is_recompiled() -> hrx::Result<()> {
    let compiler = Compiler::resolve(None)?;
    let mut input = source();
    input.identifier = "cache-companion.cpp".into();
    let module = compiler.import_cxx(input)?;
    let request = Specialization::new("fill");
    let original = module.compile(&request)?;
    let companion = original.path().parent().unwrap().join("launch.loombc");
    std::fs::write(&companion, b"corrupted launch program")?;
    let repaired = module.compile(&request)?;
    assert_eq!(original.launch_bytes(), repaired.launch_bytes());
    let mut launch = repaired.launch_program("fill")?;
    assert_eq!(launch.evaluate(&[])?.workgroup_count, [3, 2, 1]);
    Ok(())
}

#[test]
#[ignore = "requires the native compiler"]
fn sanitizer_pipelines_and_reports_have_distinct_cache_identities() -> hrx::Result<()> {
    use hrx::loom::{CompilerOptions, SanitizerChecks, SanitizerOptions, SanitizerReporting};
    let source = r#"
kernel.def @checked() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%value: i32, %output: buffer) {
  sanitizer.assert.op %value [ne(%value, 0)] : i32
  %zero = index.constant 0 : offset
  %index = index.constant 0 : index
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero] : buffer -> view<1xi32>
  view.store %value, %view[%index] : i32, view<1xi32>
  kernel.return
}
"#;
    let ordinary = Compiler::resolve(None)?
        .module(source)
        .compile(&Specialization::new("checked"))?;
    let compile = |reporting| -> hrx::Result<_> {
        Compiler::shared(
            None,
            CompilerOptions {
                sanitizer: SanitizerOptions {
                    checks: SanitizerChecks {
                        operation: true,
                        ..Default::default()
                    },
                    reporting,
                },
                ..Default::default()
            },
        )?
        .module(source)
        .compile(&Specialization::new("checked"))
    };
    let trap = compile(SanitizerReporting::Trap)?;
    let report = compile(SanitizerReporting::ReportOnly)?;
    assert_ne!(ordinary.path(), trap.path());
    assert_ne!(trap.path(), report.path());
    assert_ne!(ordinary.bytes(), trap.bytes());
    assert_ne!(trap.bytes(), report.bytes());
    assert_eq!(trap.bytes(), compile(SanitizerReporting::Trap)?.bytes());
    assert!(report.manifest().is_some());
    Ok(())
}

#[test]
#[ignore = "requires the native compiler"]
fn dynamic_workloads_use_owned_compiler_geometry() -> hrx::Result<()> {
    let artifact = Compiler::resolve(None)?
        .module(
            r#"
kernel.def @dynamic(%count: index) {
  %n = index.assume %count [range(%count, 1, 1048576)] : index
  %one = index.constant 1 : index
  %width = index.constant 64 : index
  %extra = index.constant 63 : index
  %rounded = index.add %n, %extra : index
  %groups = index.div %rounded, %width : index
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%width, %one, %one) : index
} launch(%output: buffer) {
  %value = scalar.constant 7 : i32
  %zero = index.constant 0 : offset
  %lane = kernel.workitem.id<x> : index
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero] : buffer -> view<64xi32>
  view.store %value, %view[%lane] : i32, view<64xi32>
  kernel.return
}
"#,
        )
        .compile(&Specialization::new("dynamic"))?;
    // No compiler or module owner remains when loading the evaluator.
    let mut launch = artifact.launch_program("dynamic")?;
    for count in [1u64, 63, 64, 65, 1025, 1048576] {
        let config = launch.evaluate(&[count])?;
        assert_eq!(config.workgroup_count, [count.div_ceil(64) as u32, 1, 1]);
        assert_eq!(config.workgroup_size, [64, 1, 1]);
    }
    assert!(launch.evaluate(&[]).is_err());
    Ok(())
}
