//! Qualification of ABI-10 memory and extension-2 observation services.
use hrx::fabric::{
    AllocationOptions, Argument, Device, Engine, Fabric, MappingUpdate, ReservationAccess,
    native::*,
};
use hrx::loom::{Compiler, CxxSource, Specialization};

#[test]
#[ignore = "requires current compiler and GPU"]
fn cpp_workgroup_scans_and_reductions_execute_through_the_public_importer() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let artifact = Compiler::for_target(None, device.target())?
        .import_cxx(CxxSource::new(
            "scans.cpp",
            r#"
#include <loomcxx/kernel.h>
using loom::kernel::scan_mode;
using loom::kernel::scan_direction;
namespace wg = loom::kernel::workgroup;
[[loom::kernel, loom::workgroup_size(80, 1, 1), loom::workgroup_count(3, 1, 1)]]
void scans(const unsigned* input, unsigned* output) {
    unsigned item = wg::id.x * 80u + loom::kernel::workitem::id.x;
    unsigned value = input[item];
    output[item * 5] = wg::scan::addi(value);
    output[item * 5 + 1] = wg::scan::addi<scan_mode::exclusive>(value);
    output[item * 5 + 2] = wg::scan::addi<scan_mode::inclusive, scan_direction::reverse>(value);
    output[item * 5 + 3] = wg::scan::addi<scan_mode::exclusive, scan_direction::reverse>(value);
    output[item * 5 + 4] = wg::reduce::addi(value);
}
"#,
        ))?
        .compile(&Specialization::new("scans"))?;
    let input_values: Vec<u32> = (0..240).map(|i| i % 7).collect();
    let input_bytes: Vec<u8> = input_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = device
        .fabric()
        .allocate_from(&input_bytes, std::slice::from_ref(&device))?;
    let output = device
        .fabric()
        .allocate(240 * 5 * 4, std::slice::from_ref(&device))?;
    let kernel = unsafe { device.load(&artifact)? };
    let queue = device.queue()?;
    let done = unsafe {
        queue.dispatch(
            &kernel,
            [3, 1, 1],
            [80, 1, 1],
            &[Argument::Buffer(&input, 0), Argument::Buffer(&output, 0)],
        )?
    };
    assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
    let mut bytes = vec![0; output.len()];
    output.read(0, &mut bytes)?;
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    for i in 0..240 {
        let start = i / 80 * 80;
        let end = start + 80;
        let before: u32 = input_values[start..i].iter().sum();
        let after: u32 = input_values[i + 1..end].iter().sum();
        let value = input_values[i];
        assert_eq!(
            &words[i * 5..i * 5 + 5],
            &[
                before + value,
                before,
                after + value,
                after,
                before + value + after
            ]
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires libamdf and XDNA hardware"]
fn native_xdna_result_requirements_and_timer_rate_are_reachable() -> hrx::Result<()> {
    let device = Device::open(Engine::Xdna, 0)?;
    let fabric = device.fabric();
    let native = fabric.native_api();
    let xdna = native.xdna();
    let options = amdf_xdna_context_create_info_t {
        type_: HRX_AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
        structure_size: size_of::<amdf_xdna_context_create_info_t>() as u32,
        logical_column_count: 1,
        physical_column_origin: AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
        acceptable_scheduling_modes: AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
        ..Default::default()
    };
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { xdna.context_create.unwrap()(device.native_handle(), &options, &mut context) },
        0
    );
    let mut info = amdf_xdna_result_buffer_info_t {
        type_: HRX_AMDF_STRUCTURE_TYPE_XDNA_RESULT_BUFFER_INFO,
        structure_size: size_of::<amdf_xdna_result_buffer_info_t>() as u32,
        ..Default::default()
    };
    let status = unsafe { xdna.context_query_result_buffer_info.unwrap()(context, &mut info) };
    let rate = device.timer_clock_info();
    let destroyed = unsafe { xdna.context_destroy.unwrap()(context) };
    if destroyed != 0 {
        std::mem::forget(device);
        std::mem::forget(fabric);
    }
    assert_eq!(destroyed, 0);
    assert_eq!(status, 0);
    assert!(info.maximum_byte_length >= info.minimum_byte_length);
    assert!(info.byte_offset_alignment.is_power_of_two());
    assert!(rate?.frequency_hz > 0);
    Ok(())
}

#[test]
#[ignore = "requires libamdf; no device needed"]
fn host_clock_and_remapping_preserve_addresses_and_backing_leases() -> hrx::Result<()> {
    let fabric = Fabric::resolve()?;
    assert!(fabric.host_clock_info()?.frequency_hz > 0);
    let before = fabric.host_clock()?;
    assert!(fabric.host_clock()? >= before);
    let caps = fabric.host_reservation_capabilities()?;
    let unit = caps.mapping_granularity.max(caps.reservation_granularity);
    assert!(
        fabric
            .allocate_with_options(
                unit as usize,
                &[],
                AllocationOptions {
                    host_numa_node: Some(u32::MAX),
                    ..Default::default()
                },
                None
            )
            .is_err(),
        "invalid NUMA placement must not silently fall back"
    );
    let permissions = caps.required_access | AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    let mut reservation = fabric.reserve_virtual_memory(
        unit,
        unit,
        &[ReservationAccess {
            device: None,
            address_domain: 0,
            access: permissions,
            mapping_granularity: unit,
        }],
    )?;
    let address = reservation.access_info(0)?.address;
    let mut sources = Vec::new();
    for byte in [17, 39] {
        let source = fabric.allocate_with_options(
            unit as usize,
            &[],
            AllocationOptions {
                host_mapping_source: true,
                required_backing_flags: caps.required_source_flags,
                ..Default::default()
            },
            None,
        )?;
        source.write(0, &[byte; 16])?;
        sources.push(source);
    }
    for (index, source) in sources.iter().enumerate() {
        unsafe {
            reservation.update(
                0,
                &[MappingUpdate {
                    target_offset: 0,
                    bytes: unit,
                    buffer: Some(source),
                    source_access: 0,
                    access: permissions,
                    source_offset: 0,
                }],
            )?;
        }
        assert_eq!(reservation.access_info(0)?.address, address);
        let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, 16) };
        assert_eq!(bytes, &[if index == 0 { 17 } else { 39 }; 16]);
        assert!(matches!(source.write(0, &[0]), Err(hrx::Error::Busy(_))));
    }
    // A wholly invalid update is rejected before mutating the installed range.
    assert!(
        unsafe {
            reservation.update(
                0,
                &[MappingUpdate {
                    target_offset: unit,
                    bytes: unit,
                    buffer: Some(&sources[0]),
                    source_access: 0,
                    access: permissions,
                    source_offset: 0,
                }],
            )
        }
        .is_err()
    );
    assert_eq!(unsafe { *(address as *const u8) }, 39);
    unsafe {
        reservation.detach(0)?;
    }
    for source in sources {
        source.write(0, &[5])?;
    }
    unsafe {
        reservation.destroy()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires current compiler and GPU with AQL timestamps"]
fn aql_timing_survives_signal_reuse_and_native_clocks_are_explicit() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let family = device
        .endpoint()
        .queue_capabilities()?
        .into_iter()
        .find(|f| f.dispatch_timestamps)
        .expect("AQL timestamp capability");
    let clock = device.queue_clock_info(family.ordinal)?;
    assert!(clock.frequency_hz > 0);
    let sample = device.clock_sample(family.ordinal)?;
    assert!(sample.host[2] >= sample.host[0]);
    let queue = device.aql_queue_with_timestamps(0, None, true)?;
    let source = CxxSource::new(
        "timed.cpp",
        r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fill(unsigned* output) { output[threadIdx.x] = threadIdx.x + 7; }
"#,
    );
    let artifact = Compiler::for_target(None, device.target())?
        .import_cxx(source)?
        .compile(&Specialization::new("fill"))?;
    let kernel = unsafe { device.load(&artifact)? };
    let output = device
        .fabric()
        .allocate(256, std::slice::from_ref(&device))?;
    let prepared = unsafe {
        queue.prepare(
            &kernel,
            [1, 1, 1],
            [64, 1, 1],
            &[Argument::Buffer(&output, 0)],
        )?
    };
    let first = unsafe { prepared.dispatch()? };
    assert!(first.wait_timeout(std::time::Duration::from_secs(10))?);
    let timing = first.timing()?.unwrap();
    assert_eq!(timing.clock, clock);
    for _ in 0..3 {
        let done = unsafe { prepared.dispatch()? };
        assert!(done.wait_timeout(std::time::Duration::from_secs(10))?);
        assert!(done.timing()?.unwrap().start_tick > timing.end_tick);
        assert_eq!(first.timing()?.unwrap().start_tick, timing.start_tick);
    }
    let mut bytes = [0; 256];
    output.read(0, &mut bytes)?;
    for (i, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(u32::from_le_bytes(*word), i as u32 + 7);
    }
    Ok(())
}

#[test]
#[ignore = "requires current compiler and GPU"]
fn gpu_reads_replaced_backing_at_the_same_virtual_address() -> hrx::Result<()> {
    let device = Device::open(Engine::Gpu, 0)?;
    let fabric = device.fabric();
    let caps = device.reservation_capabilities(0)?;
    let unit = caps.mapping_granularity.max(caps.reservation_granularity);
    let permissions = caps.required_access | AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    let mut reservation = fabric.reserve_virtual_memory(
        unit,
        unit,
        &[ReservationAccess {
            device: Some(&device),
            address_domain: 0,
            access: permissions,
            mapping_granularity: unit,
        }],
    )?;
    let address = reservation.access_info(0)?.address;
    let artifact = Compiler::for_target(None, device.target())?
        .import_cxx(CxxSource::new(
            "remap.cpp",
            r#"
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void copy(const unsigned* input, unsigned* output) { output[threadIdx.x] = input[threadIdx.x]; }
"#,
        ))?
        .compile(&Specialization::new("copy"))?;
    let kernel = unsafe { device.load(&artifact)? };
    let output = fabric.allocate(256, std::slice::from_ref(&device))?;
    let queue = device.queue()?;
    let prepared = unsafe {
        queue.prepare(
            &kernel,
            [1, 1, 1],
            [64, 1, 1],
            &[Argument::Address(address), Argument::Buffer(&output, 0)],
        )?
    };
    for value in [17u32, 83] {
        let source = fabric.allocate_with_options(
            unit as usize,
            std::slice::from_ref(&device),
            AllocationOptions {
                mapping_source: true,
                required_backing_flags: caps.required_source_flags,
                ..Default::default()
            },
            None,
        )?;
        source.write(0, &value.to_le_bytes().repeat(64))?;
        unsafe {
            reservation.update(
                0,
                &[MappingUpdate {
                    target_offset: 0,
                    bytes: unit,
                    buffer: Some(&source),
                    source_access: 0,
                    access: permissions,
                    source_offset: 0,
                }],
            )?;
        }
        // Dropping this reference tests that the reservation retains the source.
        drop(source);
        let done = unsafe { prepared.dispatch()? };
        match done.wait_timeout(std::time::Duration::from_secs(10)) {
            Ok(true) => {}
            result => {
                // Raw pointer arguments are external to the queue's ownership
                // chain. A timeout/error cannot authorize freeing their backing.
                std::mem::forget(reservation);
                return Err(result.err().unwrap_or_else(|| {
                    hrx::Error::DeviceLost(
                        "reservation reader did not retire in ten seconds".into(),
                    )
                }));
            }
        }
        let mut bytes = [0; 256];
        output.read(0, &mut bytes)?;
        assert_eq!(bytes.as_slice(), value.to_le_bytes().repeat(64));
    }
    unsafe {
        reservation.detach(0)?;
    }
    Ok(())
}
