//! GPU-authored file requests, native completion/error handling and owned pages.
#![cfg(target_os = "linux")]
use hrx::{
    fabric::{Argument, Engine, Fabric, NativeLifetime, StorageOptions, StorageRing},
    loom::{Compiler, Specialization},
};
use std::os::{fd::AsRawFd, unix::fs::FileExt};

#[test]
#[ignore = "requires the native compiler"]
fn compiles_gpu_file_protocol_for_current_target() -> hrx::Result<()> {
    let artifact = Compiler::resolve(None)?
        .module(include_str!("kernels/file_exchange.gpu.loom"))
        .compile(&Specialization::new("file_exchange"))?;
    let launch = artifact.launch_program("file_exchange")?.evaluate(&[])?;
    assert_eq!(launch.workgroup_count, [1; 3]);
    assert_eq!(launch.workgroup_size, [1; 3]);
    Ok(())
}
#[test]
#[ignore = "requires gfx1151 and Linux caller-owned SQPOLL rings"]
fn gpu_file_roundtrip_short_read_error_and_budget_ownership() -> hrx::Result<()> {
    let fabric = Fabric::resolve_with_lifetime(NativeLifetime::Process)?;
    let device = fabric
        .endpoints()?
        .into_iter()
        .find(|e| e.engine() == Engine::Gpu)
        .ok_or_else(|| hrx::Error::Unsupported("GPU unavailable".into()))?
        .open()?;
    let artifact = Compiler::for_target(None, device.target())?
        .module(include_str!("kernels/file_exchange.gpu.loom"))
        .compile(&Specialization::new("file_exchange"))?;
    let kernel = unsafe { device.load(&artifact) }?;
    let queue = device.queue()?;
    let manager = hrx::residency::ResidencyManager::new(8 << 20)?;
    let budget = manager.budget();
    for case in ["roundtrip", "short_read", "read_only"] {
        let short = case == "short_read";
        {
            let file = tempfile::tempfile()?;
            let words = 16u32;
            let rounds = 17u32;
            let seed = 5u32;
            let blocks = 8u32;
            let mut file_words = (0..2 * blocks * words)
                .map(|i| i.wrapping_mul(73).wrapping_add(19))
                .collect::<Vec<_>>();
            let bytes = file_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>();
            file.write_all_at(if short { &bytes[..4] } else { &bytes }, 0)?;
            let payload = device.fabric().allocate_registered(
                3 * 4096,
                std::slice::from_ref(&device),
                Some(&budget),
            )?;
            payload.write(0, &vec![0xa5; payload.len()])?;
            let registered_file = if case == "read_only" {
                std::fs::OpenOptions::new()
                    .read(true)
                    .open(format!("/proc/self/fd/{}", file.as_raw_fd()))?
            } else {
                file.try_clone()?
            };
            let before_ring = budget.reserved_bytes();
            assert!(
                StorageRing::new(
                    &device,
                    std::slice::from_ref(&registered_file),
                    &payload,
                    StorageOptions {
                        entries: 3,
                        memory_budget: Some(budget.clone()),
                        ..Default::default()
                    }
                )
                .is_err()
            );
            assert_eq!(budget.reserved_bytes(), before_ring);
            let ring = StorageRing::new(
                &device,
                std::slice::from_ref(&registered_file),
                &payload,
                StorageOptions {
                    memory_budget: Some(budget.clone()),
                    ..Default::default()
                },
            )?;
            assert!(matches!(
                ring.memory().write(0, &[0]),
                Err(hrx::Error::Busy(_))
            ));
            let layout = ring.layout();
            let records = device.fabric().allocate_budgeted(
                16 + (rounds * (words + 3) * 4) as usize,
                std::slice::from_ref(&device),
                &budget,
            )?;
            records.write(0, &vec![0; records.len()])?;
            let scalars = [
                rounds,
                words,
                blocks - 1,
                if short { 0 } else { seed },
                4096,
                0,
            ]
            .map(u32::to_le_bytes);
            let host = layout.host_payload.to_le_bytes();
            let sq = layout.submission_mask.to_le_bytes();
            let cq = layout.completion_mask.to_le_bytes();
            let mut args = vec![
                Argument::Buffer(ring.memory(), layout.submission_entries),
                Argument::Buffer(ring.memory(), layout.submission_tail),
                Argument::Buffer(ring.memory(), layout.completion_entries),
                Argument::Buffer(ring.memory(), layout.completion_head),
                Argument::Buffer(ring.memory(), layout.completion_tail),
                Argument::Buffer(&payload, 0),
                Argument::Buffer(&records, 0),
                Argument::Value(&host),
                Argument::Value(&sq),
                Argument::Value(&cq),
            ];
            args.extend(scalars.iter().map(|s| Argument::Value(s)));
            let command = unsafe { queue.prepare(&kernel, [1; 3], [1; 3], &args) }?;
            let charge = budget.reserved_bytes();
            let mut execution = unsafe { ring.dispatch(command) }?;
            assert!(matches!(
                payload.read(0, &mut [0]),
                Err(hrx::Error::Busy(_))
            ));
            assert!(execution.wait_timeout(std::time::Duration::from_secs(20))?);
            assert_eq!(budget.reserved_bytes(), charge);
            let mut actual = vec![0; records.len()];
            records.read(0, &mut actual)?;
            let get = |i: usize| u32::from_le_bytes(actual[i * 4..i * 4 + 4].try_into().unwrap());
            if short {
                assert_eq!(get(0), 0);
                assert_eq!(get(1) as i32, -61);
                assert_eq!(get(2), 2);
            } else if case == "read_only" {
                assert_eq!(get(0), 0);
                assert_eq!(get(1) as i32, -libc::EBADF);
                assert_eq!(get(2), 2);
                let mut actual_file = vec![0; bytes.len()];
                file.read_exact_at(&mut actual_file, 0)?;
                assert_eq!(actual_file, bytes);
            } else {
                assert_eq!(get(0), rounds);
                assert_eq!(get(1), 0);
                assert_eq!(get(2), rounds * 3);
                let mut cause = seed;
                for round in 0..rounds {
                    let input = cause & (blocks - 1);
                    let output = (input * 5 + 1) & (blocks - 1);
                    let output = output + blocks;
                    let base = (4 + round * (words + 3)) as usize;
                    assert_eq!(get(base), input);
                    assert_eq!(get(base + 1), cause);
                    assert_eq!(get(base + 2), output);
                    for index in 0..words {
                        let expected = file_words[(input * words + index) as usize]
                            .wrapping_mul(3)
                            .wrapping_add(cause)
                            .wrapping_add(round);
                        assert_eq!(get(base + 3 + index as usize), expected);
                        file_words[(output * words + index) as usize] = expected;
                    }
                    cause = get(base + 3);
                }
                let mut actual_file = vec![0; bytes.len()];
                file.read_exact_at(&mut actual_file, 0)?;
                assert_eq!(
                    actual_file,
                    file_words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>()
                );
            }
            let mut guarded = vec![0; payload.len()];
            payload.read(0, &mut guarded)?;
            for start in [0, 4096, 8192] {
                assert!(
                    guarded[start + words as usize * 4..start + 4096]
                        .iter()
                        .all(|v| *v == 0xa5)
                );
            }
        }
        assert_eq!(budget.reserved_bytes(), 0);
    }
    Ok(())
}
