// Adapted from libamdf/cts/interop/gpu/xdna/recipes/resident_transaction.cc.
// Copyright 2026 The IREE Authors and hrx-rs contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// One-column scalar stream service; ordinary compiler command bytes stay intact.
#[allow(clippy::too_many_arguments)]
pub fn records(
    startup: u64,
    request: u64,
    response: u64,
    ack: u64,
    payload: u32,
    offset: u32,
    credits: u32,
    stride: u32,
) -> (Vec<u8>, u32, Vec<u8>, u32) {
    fn word(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn masked(bytes: &mut Vec<u8>, opcode: u8, address: u32, mask: u32, value: u32) {
        let start = bytes.len();
        bytes.resize(start + 32, 0);
        let r = &mut bytes[start..];
        r[0] = opcode;
        word(r, 8, address);
        word(r, 16, value);
        word(r, 20, mask);
        word(r, 24, 32);
    }
    fn write(bytes: &mut Vec<u8>, address: u32, value: u32) {
        let start = bytes.len();
        bytes.resize(start + 24, 0);
        let r = &mut bytes[start..];
        word(r, 8, address);
        word(r, 16, value);
        word(r, 20, 24);
    }
    fn descriptor(bytes: &mut Vec<u8>, index: u32, address: u64, length: u32, completion: u32) {
        assert!(address < 1u64 << 48 && address.is_multiple_of(4) && length.is_multiple_of(4));
        let start = bytes.len();
        bytes.resize(start + 48, 0);
        let r = &mut bytes[start..];
        r[0] = 1;
        word(r, 8, 0x1d000 + index * 32);
        word(r, 12, 48);
        for (i, v) in [
            length / 4,
            address as u32,
            (address >> 32) as u32,
            0,
            3 << 30,
            2 << 24,
            0,
            (1 << 25) | completion,
        ]
        .into_iter()
        .enumerate()
        {
            word(r, 16 + i * 4, v);
        }
    }
    assert!((1..=2).contains(&credits));
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    masked(&mut prefix, 3, 0x232000, 3, 2);
    masked(&mut prefix, 3, 0x1f000, 0xc000, 0x4000);
    masked(&mut prefix, 3, 0x1f004, 0xc0, 0x40);
    for (address, value) in [
        (0x3f034, 0x80000009),
        (0x3f124, 0x80000000),
        (0x1b0030, 0x80000008),
        (0x1b0120, 0x80000000),
        (0x23f000, 0x80000006),
        (0x23f118, 0x80000000),
        (0x23f018, 0x80000000),
        (0x23f100, 0x80000000),
        (0x1b0020, 0x8000000e),
        (0x1b0138, 0x80000000),
        (0x3f13c, 0xc0000000),
        (0x3f000, 0xc0000009),
        (0x3f014, 0xc0000091),
        (0x3f2f0, 0x071f0101),
        (0x3f2f4, 0x091f0111),
        (0x1d208, 0),
        (0x1d218, 0),
        (0x14000, 0),
    ] {
        write(&mut prefix, address, value);
    }
    descriptor(&mut prefix, 10, startup, 4, 0);
    for slot in 0..credits {
        let bd = 2 + 4 * slot;
        let origin = u64::from(slot * stride);
        descriptor(&mut prefix, bd, request + origin, 4, 0);
        descriptor(
            &mut prefix,
            bd + 1,
            request + origin + u64::from(offset),
            payload,
            0,
        );
        descriptor(
            &mut prefix,
            bd + 2,
            response + origin + u64::from(offset),
            payload,
            (1 << 18) | (1 << 26) | ((bd + 3) << 27),
        );
        descriptor(
            &mut prefix,
            bd + 3,
            response + origin,
            4,
            (1 << 12) | (0x7f << 5),
        );
    }
    descriptor(&mut prefix, 15, ack, 4, 0);
    masked(&mut suffix, 4, 0x1d224, 0x0078003c, 0);
    masked(&mut suffix, 4, 0x1d22c, 0x0078003c, 0);
    (prefix, 23 + 4 * credits, suffix, 2)
}
