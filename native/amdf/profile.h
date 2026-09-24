// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#ifndef HRX_FABRIC_PROFILE_H
#define HRX_FABRIC_PROFILE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Optional diagnostics ABI. Nonzero results are errno values. Outputs are
// published only on success. No device resources survive either call.
int hrx_fabric_gpu_profile_clock(uint32_t major, uint32_t minor, uint64_t *hz);
// Emit a GPU clock write. The caller supplies aligned device-visible storage
// and completion ordering; this function does not submit or wait for work.
int hrx_fabric_gpu_profile_marker(uint64_t address, uint32_t *words,
                                  uint32_t capacity, uint32_t *count);
#ifdef __cplusplus
}
#endif
#endif
