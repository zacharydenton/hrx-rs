//! Native storage ownership and byte-preservation tests.
#![cfg(target_os = "linux")]
use hrx::{
    fabric::NativeLifetime,
    storage::{StorageConfig, StorageMode, StorageProgress, StorageSession},
};
use std::{fs::File, io::Write, time::Duration};

#[test]
#[ignore = "requires native compiler"]
fn coordinator_compiles() -> hrx::Result<()> {
    let artifact = hrx::loom::Compiler::resolve(None)?
        .module(include_str!("../src/storage/batch.loom"))
        .compile(&hrx::loom::Specialization::new("storage_batch"))?;
    assert_eq!(
        artifact
            .launch_program("storage_batch")?
            .evaluate(&[])?
            .workgroup_size,
        [1; 3]
    );
    Ok(())
}
#[test]
#[ignore = "requires gfx1151 and native io_uring"]
fn bounded_storage_reads_copies_writes_and_releases() -> hrx::Result<()> {
    for mode in [StorageMode::Buffered, StorageMode::Direct] {
        for progress in [StorageProgress::Sqpoll, StorageProgress::Wait] {
            let mut file = tempfile::NamedTempFile::new_in(env!("CARGO_MANIFEST_DIR"))?;
            let data: Vec<u8> = (0..131089).map(|i| ((i * 73) ^ (i >> 8)) as u8).collect();
            file.write_all(&data)?;
            file.as_file().sync_all()?;
            let manager = hrx::residency::ResidencyManager::new(16 << 20)?;
            {
                let device = hrx::Device::open_with_lifetime(0, NativeLifetime::Process)?;
                let mut stream = device.stream_with_options(hrx::StreamOptions {
                    memory_budget: Some(manager.budget()),
                    ..Default::default()
                })?;
                let session = StorageSession::new(
                    &stream,
                    &[File::open(file.path())?],
                    StorageConfig {
                        mode,
                        progress,
                        slots: 4,
                        slot_bytes: 32768,
                        statistics: true,
                    },
                )?;
                let mut tickets = Vec::new();
                for offset in [7, 4096, 8195, 16384] {
                    tickets.push(session.read(0, offset, 4097)?);
                }
                assert!(matches!(
                    session.read(0, 30000, 1),
                    Err(hrx::Error::Busy(_))
                ));
                let duplicate = session.read(0, 7, 4097)?;
                assert_eq!(session.statistics().shared_reads, 1);
                let output = stream.allocate(4097 * 4)?;
                let mut copies = Vec::new();
                for (i, ticket) in tickets.iter().enumerate() {
                    let lease = ticket
                        .wait_timeout(Duration::from_secs(10))?
                        .expect("storage timeout");
                    copies.push(lease.copy_to(&mut stream, output.slice(i * 4097, 4097))?);
                }
                for copy in &mut copies {
                    copy.wait()?;
                }
                let mut actual = vec![0; 4097 * 4];
                stream.read_blocking(output.binding(), &mut actual)?;
                for (i, offset) in [7, 4096, 8195, 16384].into_iter().enumerate() {
                    assert_eq!(
                        actual[i * 4097..(i + 1) * 4097],
                        data[offset..offset + 4097]
                    );
                }
                drop(tickets);
                drop(duplicate);
                drop(copies);
                // Repeated batches wrap native SQ/CQ counters without resetting the ring.
                for round in 0..70 {
                    let ticket = session.read(0, (round * 257) as u64, 29)?;
                    let lease = ticket.wait()?;
                    let mut got = [0; 29];
                    lease.read(&mut got)?;
                    assert_eq!(got, data[round * 257..round * 257 + 29]);
                }
                let tail = session.read(0, 131080, 9)?;
                let lease = tail.wait()?;
                let mut actual = [0; 9];
                lease.read(&mut actual)?;
                assert_eq!(actual, data[131080..]);
                drop(lease);
                drop(tail);
                assert!(session.read(0, data.len() as u64, 1).is_err());
                assert!(session.write(0, 0, &[0; 4096]).is_err());
                assert!(session.statistics().physical_requests >= 75);
            }
            assert_eq!(manager.budget().reserved_bytes(), 0);
            {
                let device = hrx::Device::open_with_lifetime(0, NativeLifetime::Process)?;
                let stream = device.stream()?;
                let session = StorageSession::new(
                    &stream,
                    &[file.reopen()?],
                    StorageConfig {
                        mode,
                        progress,
                        slots: 2,
                        slot_bytes: 8192,
                        ..Default::default()
                    },
                )?;
                session.write(0, 4096, &[0x35; 4096])?.wait()?;
                let read = session.read(0, 4096, 4096)?;
                let lease = read.wait()?;
                let mut actual = [0; 4096];
                lease.read(&mut actual)?;
                assert_eq!(actual, [0x35; 4096]);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires native GPU and io_uring"]
fn storage_errors_and_retained_ownership() -> hrx::Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(env!("CARGO_MANIFEST_DIR"))?;
    file.write_all(&[0x29; 8192])?;
    let manager = hrx::residency::ResidencyManager::new(8 << 20)?;
    let device = hrx::Device::open_with_lifetime(0, NativeLifetime::Process)?;
    let instance = hrx::Device::open(0)?;
    let instance_stream = instance.stream()?;
    assert_ne!(instance_stream.device_id(), device.stream()?.device_id());
    assert!(
        StorageSession::new(
            &instance_stream,
            &[file.reopen()?],
            StorageConfig {
                slots: 1,
                slot_bytes: 4096,
                ..Default::default()
            }
        )
        .is_err()
    );
    drop(instance_stream);
    let mut stream = device.stream_with_options(hrx::StreamOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    let session = StorageSession::new(
        &stream,
        &[file.reopen()?, file.reopen()?],
        StorageConfig {
            slots: 2,
            slot_bytes: 4096,
            ..Default::default()
        },
    )?;
    let ticket = session.read(0, 0, 4096)?;
    let duplicate = session.read(1, 0, 4096)?;
    assert_eq!(session.statistics().shared_reads, 1);
    assert!(matches!(
        session.write(1, 0, &[0; 4096]),
        Err(hrx::Error::Busy(_))
    ));
    let lease = ticket.wait()?;
    drop(ticket);
    drop(duplicate);
    drop(session);
    assert!(manager.budget().reserved_bytes() >= 8192);
    let mut actual = [0; 4096];
    lease.read(&mut actual)?;
    assert_eq!(actual, [0x29; 4096]);
    drop(lease);
    assert_eq!(manager.budget().reserved_bytes(), 0);
    let consumers = StorageSession::new(
        &stream,
        &[file.reopen()?],
        StorageConfig {
            slots: 1,
            slot_bytes: 4096,
            ..Default::default()
        },
    )?;
    let output = stream.allocate(64)?;
    let lease = consumers.read(0, 0, 64)?.wait()?;
    // SAFETY: the callback enqueues only this source read, then injects a host error.
    let failed = unsafe {
        lease.enqueue(&mut stream, |stream, source| {
            stream.copy(output.binding(), source)?;
            Err(hrx::Error::Message("injected consumer error".into()))
        })
    };
    assert!(failed.is_err());
    let mut actual = [0; 64];
    stream.read_blocking(output.binding(), &mut actual)?;
    assert_eq!(actual, [0x29; 64]);
    // Successful drain reclaimed the slot despite the callback failure.
    consumers.read(0, 128, 64)?.wait()?;
    drop(consumers);
    drop(output);
    let extension = StorageSession::new(
        &stream,
        &[file.reopen()?, file.reopen()?],
        StorageConfig {
            slots: 1,
            slot_bytes: 4096,
            ..Default::default()
        },
    )?;
    extension.write(0, 8192, &[0x71; 4096])?.wait()?;
    let read = extension.read(1, 8192, 4096)?.wait()?;
    let mut bytes = [0; 4096];
    read.read(&mut bytes)?;
    assert_eq!(bytes, [0x71; 4096]);
    drop(read);
    drop(extension);
    let session = StorageSession::new(
        &stream,
        &[file.reopen()?],
        StorageConfig {
            slots: 2,
            slot_bytes: 4096,
            progress: StorageProgress::Wait,
            ..Default::default()
        },
    )?;
    // Registered size is only an admission bound; an external truncation still
    // produces a terminal short-read error and stops later issuance.
    file.as_file().set_len(1024)?;
    assert!(session.read(0, 0, 4096)?.wait().is_err());
    assert_eq!(session.statistics().physical_requests, 2);
    assert!(session.read(0, 0, 16).is_err());
    drop(session);
    drop(stream);
    assert_eq!(manager.budget().reserved_bytes(), 0);
    Ok(())
}
