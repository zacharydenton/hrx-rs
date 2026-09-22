// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#ifndef HRX_FABRIC_BRIDGE_H
#define HRX_FABRIC_BRIDGE_H
#include <stddef.h>
#include <stdint.h>

typedef struct hrx_fabric_gpu_image hrx_fabric_gpu_image;
typedef struct hrx_fabric_gpu_info {
  uint64_t storage_bytes;
  uint64_t descriptor_offset;
  uint32_t kernarg_bytes;
  uint32_t private_bytes;
  uint32_t wave_size;
  uint32_t local_bytes;
  uint32_t argument_count;
  uint32_t workgroup_size[3];
} hrx_fabric_gpu_info;
typedef struct hrx_fabric_gpu_argument {
  uint32_t kind;  // 1: inline bytes; 2: global buffer; 0: unsupported.
  uint32_t offset;
  uint32_t size;
  uint32_t reserved;
} hrx_fabric_gpu_argument;

// Nonzero status owns no error object; copy the thread-local error immediately.
const char* hrx_fabric_error(void);
int hrx_fabric_gpu_image_open(const uint8_t* data, size_t size,
                             const char* symbol, hrx_fabric_gpu_image** out,
                             hrx_fabric_gpu_info* info);
void hrx_fabric_gpu_image_close(hrx_fabric_gpu_image* image);
int hrx_fabric_gpu_argument_info(const hrx_fabric_gpu_image* image, uint32_t index,
                                hrx_fabric_gpu_argument* out);
int hrx_fabric_gpu_image_load(const hrx_fabric_gpu_image* image,
                             uint8_t* storage, size_t size, uint64_t address);
int hrx_fabric_gpu_dispatch(const hrx_fabric_gpu_image* image,
                           uint64_t image_address, const uint16_t block[3],
                           const uint32_t grid[3], uint64_t kernarg_address,
                           const uint8_t* kernarg, size_t kernarg_size,
                           uint64_t scratch_address, uint64_t scratch_length,
                           uint32_t scratch_waves, uint32_t shader_engines,
                           uint32_t* words, uint32_t capacity, uint32_t* count);

typedef struct hrx_fabric_xdna_program hrx_fabric_xdna_program;
typedef struct hrx_fabric_xdna_binding {
  void* memory;
  void* host_pointer;
  uint64_t allocation_length;
  uint64_t offset;
  uint64_t length;
  uint64_t address;
} hrx_fabric_xdna_binding;
// API tables, instance, endpoint and device are borrowed until close succeeds.
// A nonnull output on failure must still be closed or quarantined by the owner.
int hrx_fabric_xdna_open(const void* api, const void* xdna_api, void* instance,
    void* endpoint, void* device, const uint8_t* bytes, size_t length,
    const char* symbol, uint16_t columns, uint32_t binding_count,
    const hrx_fabric_xdna_binding* bindings, hrx_fabric_xdna_program** out);
int hrx_fabric_xdna_close(hrx_fabric_xdna_program* program);
uint64_t hrx_fabric_xdna_submit(hrx_fabric_xdna_program* program, uint64_t* submission);
uint64_t hrx_fabric_xdna_wait(hrx_fabric_xdna_program* program,
    uint64_t submission, uint64_t timeout_ns);
#endif
