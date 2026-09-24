#!/usr/bin/env python3
"""Check the optional bridge ABI without submitting GPU commands."""
import ctypes as c
import errno
import sys

library = c.CDLL(sys.argv[1])
marker = library.hrx_fabric_gpu_profile_marker
marker.argtypes = [c.c_uint64, c.POINTER(c.c_uint32), c.c_uint32, c.POINTER(c.c_uint32)]
marker.restype = c.c_int
clock = library.hrx_fabric_gpu_profile_clock
clock.argtypes = [c.c_uint32, c.c_uint32, c.POINTER(c.c_uint64)]
clock.restype = c.c_int
for address, capacity in [(0, 32), (7, 32), (4096, 5)]:
    words = (c.c_uint32 * 32)(*([0xDEADBEEF] * 32))
    count = c.c_uint32(0xBAD)
    assert marker(address, words, capacity, c.byref(count)) == errno.EINVAL
    assert count.value == 0xBAD and list(words) == [0xDEADBEEF] * 32
words = (c.c_uint32 * 32)(*([0xDEADBEEF] * 32))
count = c.c_uint32()
assert marker(4096, words, 32, c.byref(count)) == 0
assert count.value == 6 and list(words)[6:] == [0xDEADBEEF] * 26
assert words[4] == 4096 and words[5] == 0
hz = c.c_uint64(0xBAD)
assert clock(0xFFFFFFFF, 0xFFFFFFFF, c.byref(hz)) != 0
assert hz.value == 0xBAD
assert clock(0, 0, None) == errno.EINVAL
print('optional profile bridge ABI: passed')
