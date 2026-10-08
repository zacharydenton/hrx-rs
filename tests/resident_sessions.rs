//! Real single-invocation GPU–NPU causal exchange and pre-start rollback.
use hrx::{
    fabric::{
        Argument, Engine, Fabric, NativeLifetime, ResidentGpu, ResidentSession, ResidentStartup,
        XdnaBinding,
    },
    loom::{Compiler, Specialization},
};
#[path = "support/resident_routes.rs"]
mod routes;

#[test]
#[ignore = "requires current native bridge, gfx1151 and NPU5"]
fn resident_exchange_allocated() -> hrx::Result<()> {
    run_exchange(false, false)
}

#[test]
#[ignore = "requires current native bridge, gfx1151 and NPU5"]
fn resident_exchange_registered() -> hrx::Result<()> {
    run_exchange(true, false)
}

#[test]
#[ignore = "requires current native bridge, gfx1151 AQL and NPU5"]
fn resident_exchange_registered_aql() -> hrx::Result<()> {
    run_exchange(true, true)
}

fn run_exchange(registered: bool, aql: bool) -> hrx::Result<()> {
    let fabric = Fabric::resolve_with_lifetime(if registered {
        NativeLifetime::Process
    } else {
        NativeLifetime::Instance
    })?;
    let endpoints = fabric.endpoints()?;
    let open = |engine| {
        endpoints
            .iter()
            .find(|e| e.engine() == engine)
            .ok_or_else(|| hrx::Error::Unsupported(format!("{engine:?} unavailable")))?
            .open()
    };
    let gpu = open(Engine::Gpu)?;
    let npu = open(Engine::Xdna)?;
    let gpu_source = format!(
        "{}\n{}",
        include_str!("kernels/completed_tick.loom"),
        include_str!("kernels/resident_exchange.gpu.loom")
            .replace("template.decl @completed_tick() -> (i32)", "")
    );
    let gpu_artifact = Compiler::for_target(None, gpu.target())?
        .module(&gpu_source)
        .compile(&Specialization::new("resident_exchange"))?;
    let npu_artifact = Compiler::for_target(None, npu.target())?
        .module(include_str!("kernels/resident_exchange.xdna.loom"))
        .compile(&Specialization::new("resident_service_array"))?;
    let kernel = unsafe { gpu.load(&gpu_artifact) }?;
    let pm4_queue = if aql { None } else { Some(gpu.queue()?) };
    let aql_queue = if aql { Some(gpu.aql_queue(0)?) } else { None };
    for (rounds, words, credits, gpu_first, abort) in [
        (0u32, 1u32, 1u32, true, 0),
        (1, 17, 1, false, 0),
        (17, 17, 2, true, 0),
        (257, 1024, 2, false, 0),
        (258, 16, 2, true, 0),
        (257, 64, 1, false, 0),
        (17, 16, 2, true, 1),
        (17, 16, 2, false, 2),
    ] {
        eprintln!(
            "resident: rounds={rounds}, words={words}, credits={credits}, gpu_first={gpu_first}, abort={abort}"
        );
        let manager = hrx::residency::ResidencyManager::new(8 << 20)?;
        let budget = manager.budget();
        {
            let startup = ResidentStartup::new(&gpu, &npu, Some(&budget))?;
            let profile = gpu
                .fabric()
                .allocation_profiles(&[gpu.clone(), npu.clone()], true)?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    hrx::Error::Unsupported("coherent joint backing unavailable".into())
                })?;
            let allocate = |n| {
                if registered {
                    fabric.allocate_registered(n, &[gpu.clone(), npu.clone()], Some(&budget))
                } else {
                    profile.allocate(n, Some(&budget))
                }
            };
            let stride = (words * 4 + 64).div_ceil(64) * 64;
            let request = allocate((credits * stride) as usize)?;
            let response = allocate((credits * stride) as usize)?;
            let control = allocate(256)?;
            let terminal = allocate(64)?;
            let configuration = allocate(64)?;
            let records = allocate(((words + 4) * 4 * rounds).max(64) as usize)?;
            for buffer in [&request, &response, &control, &terminal, &records] {
                buffer.write(0, &vec![0; buffer.len()])?;
            }
            configuration.write(0, &vec![0; configuration.len()])?;
            configuration.write(
                0,
                &[rounds, words, credits]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            let program = unsafe {
                npu.prepare_xdna(
                    &npu_artifact,
                    1,
                    &[
                        XdnaBinding {
                            buffer: &configuration,
                            offset: 0,
                            length: 64,
                        },
                        XdnaBinding {
                            buffer: &terminal,
                            offset: 0,
                            length: 64,
                        },
                    ],
                )
            }?;
            let (prefix, prefix_count, suffix, suffix_count) = routes::records(
                startup.buffer().device_address(&npu)?,
                request.device_address(&npu)?,
                response.device_address(&npu)?,
                control.device_address(&npu)? + 192,
                words * 4,
                64,
                credits,
                stride,
            );
            let program = unsafe {
                program.wrap_transaction(
                    &prefix,
                    prefix_count,
                    &suffix,
                    suffix_count,
                    &[
                        startup.buffer().clone(),
                        request.clone(),
                        response.clone(),
                        control.clone(),
                    ],
                    Some(&budget),
                )
            }?;
            let seed = 0xffffff01u32;
            let scalars = [rounds, seed, words, 16, credits, stride].map(u32::to_le_bytes);
            let mut arguments = vec![
                Argument::Buffer(&request, 0),
                Argument::Buffer(&response, 0),
                Argument::Buffer(startup.buffer(), 0),
                Argument::Buffer(&control, 0),
                Argument::Buffer(&records, 0),
            ];
            arguments.extend(scalars.iter().map(|v| Argument::Value(v)));
            let command = unsafe {
                if let Some(queue) = &pm4_queue {
                    ResidentGpu::Pm4(queue.prepare(&kernel, [1; 3], [1; 3], &arguments)?)
                } else {
                    ResidentGpu::Aql(
                        aql_queue
                            .as_ref()
                            .unwrap()
                            .prepare(&kernel, [1; 3], [1; 3], &arguments)?,
                    )
                }
            };
            let mut session = unsafe { ResidentSession::new(startup, command, program) };
            assert!(session.start().is_err());
            assert!(session.wait_timeout(std::time::Duration::ZERO).is_err());
            if gpu_first {
                session.submit_gpu()?;
            } else {
                session.submit_npu()?;
            }
            assert!(!session.is_complete()?);
            assert!(matches!(
                request.read(0, &mut [0]),
                Err(hrx::Error::Busy(_))
            ));
            if abort != 0 {
                session.abort()?;
            } else {
                if gpu_first {
                    session.submit_npu()?;
                } else {
                    session.submit_gpu()?;
                }
                session.start()?;
                assert!(session.abort().is_err());
            }
            assert!(
                session.wait_timeout(std::time::Duration::from_secs(20))?,
                "rounds={rounds}, words={words}, credits={credits}, abort={abort}"
            );
            assert!(session.is_complete()?);
            assert!(session.submit_gpu().is_err());
            let mut transcript = vec![0; records.len()];
            records.read(0, &mut transcript)?;
            let get =
                |i: usize| u32::from_le_bytes(transcript[4 * i..4 * i + 4].try_into().unwrap());
            if abort == 0 {
                let mut causes = [seed, seed.wrapping_add(0x9e3779b9)];
                for generation in 1..=rounds {
                    let slot = ((generation - 1) % credits) as usize;
                    let cause = causes[slot];
                    let base = ((generation - 1) * (words + 4)) as usize;
                    assert_eq!(get(base), generation);
                    assert_eq!(get(base + 1), cause);
                    for word in 0..words {
                        let expected = cause
                            .wrapping_add(generation.wrapping_mul(257))
                            .wrapping_add(word * 17)
                            .wrapping_mul(3)
                            .wrapping_add(generation);
                        assert_eq!(
                            get(base + 4 + word as usize),
                            expected,
                            "rounds={rounds} generation={generation} word={word} cause={cause}"
                        );
                    }
                    causes[slot] = get(base + 4);
                }
            } else {
                assert!(transcript.iter().all(|v| *v == 0));
            }
            for buffer in [&request, &response] {
                let mut actual = vec![0; buffer.len()];
                buffer.read(0, &mut actual)?;
                for slot in 0..credits as usize {
                    let base = slot * stride as usize;
                    assert!(actual[base + 4..base + 64].iter().all(|v| *v == 0));
                    assert!(
                        actual[base + 64 + words as usize * 4..base + stride as usize]
                            .iter()
                            .all(|v| *v == 0)
                    );
                }
                assert!(
                    actual[(credits * stride) as usize..]
                        .iter()
                        .all(|v| *v == 0)
                );
                if abort != 0 {
                    assert!(actual.iter().all(|v| *v == 0));
                }
            }
            let mut final_control = vec![0; control.len()];
            control.read(0, &mut final_control)?;
            assert_eq!(
                u32::from_le_bytes(final_control[192..196].try_into().unwrap()),
                if abort == 0 {
                    1
                } else if abort == 1 {
                    2
                } else {
                    0
                }
            );
            assert!(
                final_control[..192]
                    .iter()
                    .chain(&final_control[196..])
                    .all(|v| *v == 0)
            );
            let mut final_npu = [0; 64];
            terminal.read(0, &mut final_npu)?;
            let terminal_words: Vec<_> = final_npu
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect();
            if abort == 1 {
                assert!(terminal_words.iter().all(|v| *v == 0));
            } else {
                assert_eq!(terminal_words[0], if abort == 0 { 1 } else { 2 });
                assert_eq!(terminal_words[1], if abort == 0 { rounds } else { 0 });
                assert_eq!(terminal_words[2], words);
                if rounds != 0 && abort == 0 {
                    let last = ((rounds - 1) * (words + 4) + 4) as usize;
                    assert_eq!(terminal_words[3], get(last));
                    assert_eq!(terminal_words[4], get(last + words as usize - 1));
                    assert_eq!(
                        terminal_words[5],
                        (0..words as usize).fold(0u32, |sum, i| sum.wrapping_add(get(last + i)))
                    );
                } else {
                    assert_eq!(&terminal_words[3..6], &[0; 3]);
                }
                assert!(terminal_words[6..].iter().all(|v| *v == 0));
            }
        }
        assert_eq!(budget.reserved_bytes(), 0);
    }
    Ok(())
}
