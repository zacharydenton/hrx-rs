//! Native sanitizer admission, report ownership, and bounded feedback.
use hrx::{
    fabric::{Argument, Device, Engine, SanitizerCheck, SanitizerRuntimeOptions},
    loom::{
        Compiler, CompilerOptions, SanitizerChecks, SanitizerOptions, SanitizerReporting,
        Specialization,
    },
};
const SOURCE: &str = r#"
kernel.def @checked() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
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
fn artifact_for_wave(wave: u32) -> hrx::Result<hrx::loom::Artifact> {
    let source = format!(
        "amdgpu.target<gfx1151> @checked_target {{subgroup_size = {wave}}}\n{}",
        SOURCE.replace(
            "kernel.def @checked",
            "kernel.def target(@checked_target) @checked"
        )
    );
    Compiler::shared(
        None,
        CompilerOptions {
            sanitizer: SanitizerOptions {
                checks: SanitizerChecks {
                    operation: true,
                    ..Default::default()
                },
                reporting: SanitizerReporting::ReportOnly,
            },
            ..Default::default()
        },
    )?
    .module(&source)
    .compile(&Specialization::new("checked"))
}
fn artifact() -> hrx::Result<hrx::loom::Artifact> {
    artifact_for_wave(32)
}
#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn sanitizer_reports_failures_without_trapping_and_preserves_success() -> hrx::Result<()> {
    let artifact = artifact()?;
    let device = Device::open(Engine::Gpu, 0)?;
    assert!(matches!(
        unsafe { device.load(&artifact) },
        Err(hrx::Error::Unsupported(_))
    ));
    let kernel = unsafe { device.load_sanitized(&artifact, &SanitizerRuntimeOptions::default()) }?;
    let output = device.fabric().allocate(4, std::slice::from_ref(&device))?;
    let queue = device.aql_queue(0)?;
    for value in [17i32, 0, 23, 0] {
        output.write(0, &0x55555555u32.to_le_bytes())?;
        let command = unsafe {
            queue.prepare(
                &kernel,
                [1; 3],
                [1; 3],
                &[
                    Argument::Value(&value.to_le_bytes()),
                    Argument::Buffer(&output, 0),
                ],
            )
        }?;
        let done = unsafe { command.dispatch() }?;
        assert!(matches!(
            kernel.sanitizer_reports(),
            Err(hrx::Error::Busy(_))
        ));
        assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
        let reports = kernel.sanitizer_reports()?;
        assert_eq!(reports.dropped, 0);
        let mut actual = [0; 4];
        output.read(0, &mut actual)?;
        if value == 0 {
            assert_eq!(reports.reports.len(), 1);
            assert_eq!(reports.reports[0].check, SanitizerCheck::Assertion);
            assert!(reports.reports[0].site.is_some());
            assert_eq!(actual, 0x55555555u32.to_le_bytes());
        } else {
            assert!(reports.reports.is_empty());
            assert_eq!(actual, value.to_le_bytes());
        }
        assert!(kernel.sanitizer_reports()?.reports.is_empty());
    }
    Ok(())
}
#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn sanitizer_capacity_reports_drops_and_retains_budget_until_retirement() -> hrx::Result<()> {
    for wave in [32, 64] {
        let artifact = artifact_for_wave(wave)?;
        let device = Device::open(Engine::Gpu, 0)?;
        let manager = hrx::residency::ResidencyManager::new(192)?;
        let budget = manager.budget();
        {
            let kernel = unsafe {
                device.load_sanitized(
                    &artifact,
                    &SanitizerRuntimeOptions {
                        capacity_bytes: 128,
                        memory_budget: Some(budget.clone()),
                        ..Default::default()
                    },
                )
            }?;
            assert_eq!(budget.reserved_bytes(), 192);
            let output = device.fabric().allocate(4, std::slice::from_ref(&device))?;
            let queue = device.aql_queue(0)?;
            let command = unsafe {
                queue.prepare(
                    &kernel,
                    [1; 3],
                    [1; 3],
                    &[
                        Argument::Value(&0i32.to_le_bytes()),
                        Argument::Buffer(&output, 0),
                    ],
                )
            }?;
            for _ in 0..16 {
                let mut last = None;
                for _ in 0..4 {
                    last = Some(unsafe { command.dispatch() }?);
                }
                let done = last.unwrap();
                assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
                let reports = kernel.sanitizer_reports()?;
                assert_eq!(reports.reports.len(), 1);
                assert_eq!(reports.dropped, 3);
                assert_eq!(budget.reserved_bytes(), 192);
            }
            drop(kernel);
            assert_eq!(budget.reserved_bytes(), 192);
        }
        assert_eq!(budget.reserved_bytes(), 0);
    }
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn graph_collects_owned_sanitizer_reports_and_charges_runtime_budget() -> hrx::Result<()> {
    use hrx::execution::{
        Access, BindingContract, ComputeEngine, KernelContract, MemoryPlacement, Runtime,
        RuntimeOptions,
    };
    let artifact = artifact()?;
    let contract = KernelContract {
        bindings: vec![BindingContract {
            bytes: 4,
            alignment: 4,
            access: Access::Write,
            layout: "i32".into(),
        }],
        constants: 0i32.to_le_bytes().to_vec(),
    };
    let pm4 = Runtime::new()?;
    assert!(matches!(
        unsafe { pm4.load_sanitized_gpu_artifact(&artifact, &[], contract.clone(), 128) },
        Err(hrx::Error::Unsupported(_))
    ));
    let manager = hrx::residency::ResidencyManager::new(64 << 20)?;
    let budget = manager.budget();
    let reports;
    {
        let runtime = Runtime::with_options(RuntimeOptions {
            compute_engine: ComputeEngine::Aql {
                maximum_private_bytes: 0,
            },
            memory_budget: Some(budget.clone()),
            ..Default::default()
        })?;
        let before = budget.reserved_bytes();
        let kernel = unsafe { runtime.load_sanitized_gpu_artifact(&artifact, &[], contract, 128) }?;
        assert!(budget.reserved_bytes() >= before + 192);
        let output = runtime.allocate(4, MemoryPlacement::HostVisible)?;
        output.map_write()?.copy_from_slice(&17i32.to_le_bytes());
        let mut graph = runtime.graph();
        graph.gpu(&kernel, &[output.view()])?;
        let graph = graph.prepare()?;
        let prepared = budget.reserved_bytes();
        for _ in 0..8 {
            graph.submit()?.wait()?;
            let result = kernel.sanitizer_reports()?;
            assert_eq!(result.reports.len(), 1);
            assert_eq!(result.dropped, 0);
            assert_eq!(&*output.map_read()?, &17i32.to_le_bytes());
            assert_eq!(budget.reserved_bytes(), prepared);
        }
        graph.submit()?.wait()?;
        reports = kernel.sanitizer_reports()?;
    }
    assert_eq!(budget.reserved_bytes(), 0);
    assert_eq!(reports.reports[0].check, SanitizerCheck::Assertion);
    assert!(reports.reports[0].site.is_some());
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn address_instrumentation_is_rejected_without_a_bounded_shadow_runtime() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let checks = SanitizerChecks {
        access: true,
        ..Default::default()
    };
    let source = SOURCE.replace(
        "  view.store",
        "  sanitizer.assert.access<write> %view[0] : view<1xi32>\n  view.store",
    );
    let artifact = Compiler::shared(
        None,
        CompilerOptions {
            sanitizer: SanitizerOptions {
                checks,
                reporting: SanitizerReporting::ReportOnly,
            },
            ..Default::default()
        },
    )?
    .module(&source)
    .compile(&Specialization::new("checked"))?;
    for result in [unsafe { device.load(&artifact) }, unsafe {
        device.load_sanitized(&artifact, &SanitizerRuntimeOptions::default())
    }] {
        match result {
            Err(hrx::Error::Unsupported(message)) if message.contains("shadow storage") => (),
            Err(error) => panic!("{}: {error}", artifact.path().display()),
            Ok(_) => panic!(
                "{}: unexpectedly admitted shadow instrumentation",
                artifact.path().display()
            ),
        };
    }
    Ok(())
}

fn race_artifact(symbol: &str, wave: u32) -> hrx::Result<hrx::loom::Artifact> {
    let source = format!(
        "amdgpu.target<gfx1151> @race_target {{subgroup_size = {wave}}}\n{}",
        include_str!("kernels/workgroup_races.loom")
            .replace("kernel.def @", "kernel.def target(@race_target) @")
    );
    Compiler::shared(
        None,
        CompilerOptions {
            sanitizer: SanitizerOptions {
                checks: SanitizerChecks {
                    race: true,
                    ..Default::default()
                },
                reporting: SanitizerReporting::ReportOnly,
            },
            ..Default::default()
        },
    )?
    .module(&source)
    .compile(&Specialization::new(symbol))
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn workgroup_races_distinguish_overlap_and_barriers_for_both_wave_sizes() -> hrx::Result<()> {
    use hrx::fabric::SanitizerAccess;
    let device = Device::open(Engine::Gpu, 0)?;
    let queue = device.aql_queue(0)?;
    for wave in [32, 64] {
        for (symbol, expected) in [
            ("tsan_workgroup_write_race", Some((4, 0))),
            ("tsan_workgroup_wide_upper_half_write_race", Some((4, 4))),
            ("tsan_workgroup_wide_current_write_race", Some((8, 0))),
            ("tsan_workgroup_adjacent_wide_writes", None),
            ("tsan_workgroup_barrier_safe_write", None),
        ] {
            let artifact = race_artifact(symbol, wave)?;
            assert!(matches!(
                unsafe { device.load(&artifact) },
                Err(hrx::Error::Unsupported(_))
            ));
            let kernel =
                unsafe { device.load_sanitized(&artifact, &SanitizerRuntimeOptions::default()) }?;
            let output = device.fabric().allocate(8, std::slice::from_ref(&device))?;
            let command = unsafe {
                queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)])
            }?;
            for _ in 0..3 {
                output.write(0, &[0; 8])?;
                let done = unsafe { command.dispatch() }?;
                assert!(
                    done.wait_timeout(std::time::Duration::from_secs(10))?,
                    "{symbol}, wave{wave}"
                );
                let reports = kernel.sanitizer_reports()?;
                assert_eq!(reports.dropped, 0);
                assert_eq!(
                    reports.reports.len(),
                    usize::from(expected.is_some()),
                    "{symbol}, wave{wave}: {reports:?}"
                );
                if let Some((bytes, address)) = expected {
                    let report = &reports.reports[0];
                    assert_eq!(report.check, SanitizerCheck::DataRace);
                    assert!(report.site.is_some());
                    let race = report.race.as_ref().unwrap();
                    assert_eq!(race.memory_space, 2);
                    assert_eq!(race.current_access, SanitizerAccess::Write);
                    assert_eq!(race.prior_access, SanitizerAccess::Write);
                    assert_eq!(race.access_bytes, bytes);
                    assert_eq!(race.memory_address, address);
                    assert!(race.prior_site.is_some());
                    assert_eq!(race.current_workgroup, [0; 3]);
                    assert_eq!(race.prior_workgroup, [0; 3]);
                    assert_ne!(race.current_workitem, race.prior_workitem);
                }
                let mut actual = [0; 8];
                output.read(0, &mut actual)?;
                let value = if symbol == "tsan_workgroup_barrier_safe_write" {
                    2u32
                } else {
                    1
                };
                assert_eq!(&actual[..4], &value.to_le_bytes(), "{symbol}, wave{wave}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn race_replay_is_queue_scoped_bounded_and_retains_its_budget() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let artifact = race_artifact("tsan_workgroup_wide_current_write_race", 32)?;
    let manager = hrx::residency::ResidencyManager::new(16 << 20)?;
    let budget = manager.budget();
    {
        let kernel = unsafe {
            device.load_sanitized(
                &artifact,
                &SanitizerRuntimeOptions {
                    capacity_bytes: 256,
                    memory_budget: Some(budget.clone()),
                    ..Default::default()
                },
            )
        }?;
        let loaded = budget.reserved_bytes();
        let output = device.fabric().allocate(8, std::slice::from_ref(&device))?;
        // Each queue prepares its own code configuration and shadow state. Two
        // uses of the same feedback channel still collect as one kernel context.
        let mut commands = Vec::new();
        for _ in 0..2 {
            let queue = device.aql_queue(0)?;
            commands.push(unsafe {
                queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)])
            }?);
        }
        let prepared = budget.reserved_bytes();
        assert!(prepared > loaded);
        for batch in 0..140 {
            // Alternating queues are explicitly ordered because output is shared.
            let command = &commands[batch % 2];
            let mut last = None;
            for _ in 0..8 {
                last = Some(unsafe { command.dispatch() }?);
            }
            assert!(matches!(
                kernel.sanitizer_reports(),
                Err(hrx::Error::Busy(_))
            ));
            assert!(
                last.unwrap()
                    .wait_timeout(std::time::Duration::from_secs(10))?
            );
            let reports = kernel.sanitizer_reports()?;
            assert_eq!(reports.reports.len(), 1);
            assert_eq!(reports.dropped, 7);
            assert_eq!(budget.reserved_bytes(), prepared);
        }
        drop(kernel);
        assert_eq!(budget.reserved_bytes(), prepared);
    }
    assert_eq!(budget.reserved_bytes(), 0);
    // Geometry limits fail before allocating dispatch state; failed allocation
    // must release all partial reservations and leave the original kernel usable.
    let queue = device.aql_queue(0)?;
    let output = device.fabric().allocate(8, std::slice::from_ref(&device))?;
    let kernel = unsafe {
        device.load_sanitized(
            &artifact,
            &SanitizerRuntimeOptions {
                capacity_bytes: 256,
                maximum_shadow_bytes: 1,
                memory_budget: Some(budget.clone()),
            },
        )
    }?;
    let before = budget.reserved_bytes();
    assert!(
        unsafe { queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)]) }
            .is_err()
    );
    assert_eq!(budget.reserved_bytes(), before);
    drop(kernel);
    assert_eq!(budget.reserved_bytes(), 0);
    let small = hrx::residency::ResidencyManager::new(400)?;
    let small_budget = small.budget();
    let kernel = unsafe {
        device.load_sanitized(
            &artifact,
            &SanitizerRuntimeOptions {
                capacity_bytes: 256,
                memory_budget: Some(small_budget.clone()),
                ..Default::default()
            },
        )
    }?;
    let before = small_budget.reserved_bytes();
    assert!(
        unsafe { queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)]) }
            .is_err()
    );
    assert_eq!(small_budget.reserved_bytes(), before);
    drop(kernel);
    assert_eq!(small_budget.reserved_bytes(), 0);
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn race_graph_replays_without_new_reservations() -> hrx::Result<()> {
    use hrx::execution::{
        Access, BindingContract, ComputeEngine, KernelContract, MemoryPlacement, Runtime,
        RuntimeOptions,
    };
    let artifact = race_artifact("tsan_workgroup_wide_upper_half_write_race", 32)?;
    let manager = hrx::residency::ResidencyManager::new(64 << 20)?;
    let budget = manager.budget();
    {
        let runtime = Runtime::with_options(RuntimeOptions {
            compute_engine: ComputeEngine::Aql {
                maximum_private_bytes: 0,
            },
            memory_budget: Some(budget.clone()),
            ..Default::default()
        })?;
        let contract = KernelContract {
            bindings: vec![BindingContract {
                bytes: 8,
                alignment: 4,
                access: Access::Write,
                layout: "2xi32".into(),
            }],
            constants: vec![],
        };
        let kernel =
            unsafe { runtime.load_sanitized_gpu_artifact(&artifact, &[], contract, 1024) }?;
        let output = runtime.allocate(8, MemoryPlacement::HostVisible)?;
        let mut graph = runtime.graph();
        graph.gpu(&kernel, &[output.view()])?;
        let graph = graph.prepare()?;
        let before = budget.reserved_bytes();
        for _ in 0..16 {
            graph.submit()?.wait()?;
            let reports = kernel.sanitizer_reports()?;
            assert_eq!(reports.reports.len(), 1);
            assert_eq!(reports.dropped, 0);
            assert_eq!(reports.reports[0].race.as_ref().unwrap().memory_address, 4);
            assert_eq!(&*output.map_read()?, &[1, 0, 0, 0, 1, 0, 0, 0]);
            assert_eq!(budget.reserved_bytes(), before);
        }
    }
    assert_eq!(budget.reserved_bytes(), 0);
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn multidimensional_workgroups_have_independent_race_state() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let queue = device.aql_queue(0)?;
    // Give every workgroup its own output slice, independent of launch geometry.
    let source = include_str!("kernels/workgroup_races.loom")
        .replace(
            "%global = buffer.assume.memory_space<global> %output : buffer",
            r#"%gx = kernel.workgroup.id<x> : index
        %gy = kernel.workgroup.id<y> : index
        %gz = kernel.workgroup.id<z> : index
        %two = index.constant 2 : index
        %six = index.constant 6 : index
        %eight = index.constant 8 : index
        %row = index.mul %gy, %two : index
        %plane = index.mul %gz, %six : index
        %inplane = index.add %row, %gx : index
        %linear = index.add %inplane, %plane : index
        %byte_index = index.mul %linear, %eight : index
        %byte_offset = index.cast %byte_index : index to offset
        %global = buffer.assume.memory_space<global> %output : buffer"#,
        )
        .replace(
            "buffer.view %global[%byte0]",
            "buffer.view %global[%byte_offset]",
        );
    let compiler = Compiler::shared(
        None,
        CompilerOptions {
            sanitizer: SanitizerOptions {
                checks: SanitizerChecks {
                    race: true,
                    ..Default::default()
                },
                reporting: SanitizerReporting::ReportOnly,
            },
            ..Default::default()
        },
    )?;
    for symbol in [
        "tsan_workgroup_wide_current_write_race",
        "tsan_workgroup_adjacent_wide_writes",
    ] {
        let artifact = compiler
            .module(&source)
            .compile(&Specialization::new(symbol))?;
        let kernel =
            unsafe { device.load_sanitized(&artifact, &SanitizerRuntimeOptions::default()) }?;
        let output = device
            .fabric()
            .allocate(12 * 8, std::slice::from_ref(&device))?;
        let large = unsafe {
            queue.prepare(
                &kernel,
                [2, 3, 2],
                [2, 1, 1],
                &[Argument::Buffer(&output, 0)],
            )
        }?;
        let small =
            unsafe { queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)]) }?;
        for (command, groups) in [(&large, 12), (&small, 1), (&large, 12)] {
            let done = unsafe { command.dispatch() }?;
            assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
            let reports = kernel.sanitizer_reports()?;
            assert_eq!(reports.dropped, 0);
            if symbol == "tsan_workgroup_wide_current_write_race" {
                assert_eq!(reports.reports.len(), groups);
                let coordinates: std::collections::BTreeSet<_> = reports
                    .reports
                    .iter()
                    .map(|report| {
                        let race = report.race.as_ref().unwrap();
                        assert_eq!(race.current_workgroup, race.prior_workgroup);
                        race.current_workgroup
                    })
                    .collect();
                assert_eq!(coordinates.len(), groups);
                if groups == 12 {
                    assert!(coordinates.contains(&[1, 2, 1]));
                }
            } else {
                assert!(reports.reports.is_empty());
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires current native compiler/bridge and gfx1151"]
fn simultaneous_queues_share_reports_but_keep_private_race_state() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let artifact = race_artifact("tsan_workgroup_wide_upper_half_write_race", 64)?;
    let kernel = unsafe { device.load_sanitized(&artifact, &SanitizerRuntimeOptions::default()) }?;
    let mut commands = Vec::new();
    for _ in 0..2 {
        let output = device.fabric().allocate(8, std::slice::from_ref(&device))?;
        let queue = device.aql_queue(0)?;
        commands.push(unsafe {
            queue.prepare(&kernel, [1; 3], [2, 1, 1], &[Argument::Buffer(&output, 0)])
        }?);
    }
    for _ in 0..16 {
        let mut last = [None, None];
        for _ in 0..8 {
            for (slot, command) in last.iter_mut().zip(&commands) {
                *slot = Some(unsafe { command.dispatch() }?);
            }
        }
        assert!(
            last[0]
                .as_ref()
                .unwrap()
                .wait_timeout(std::time::Duration::from_secs(10))?
        );
        // The other queue's lease stays active until its completion is observed,
        // even if its device-side signal has already retired.
        assert!(matches!(
            kernel.sanitizer_reports(),
            Err(hrx::Error::Busy(_))
        ));
        assert!(
            last[1]
                .as_ref()
                .unwrap()
                .wait_timeout(std::time::Duration::from_secs(10))?
        );
        let reports = kernel.sanitizer_reports()?;
        assert_eq!(reports.dropped, 0);
        assert_eq!(reports.reports.len(), 16);
        for report in &reports.reports {
            assert_eq!(report.check, SanitizerCheck::DataRace);
            assert_eq!(report.race.as_ref().unwrap().memory_address, 4);
        }
    }
    // Dropping all public preparation handles must leave submitted native work
    // and its shadow alive through the owned completion handles.
    let first = unsafe { commands[0].dispatch() }?;
    let second = unsafe { commands[1].dispatch() }?;
    drop(commands);
    assert!(first.wait_timeout(std::time::Duration::from_secs(10))?);
    assert!(second.wait_timeout(std::time::Duration::from_secs(10))?);
    assert_eq!(kernel.sanitizer_reports()?.reports.len(), 2);
    Ok(())
}
