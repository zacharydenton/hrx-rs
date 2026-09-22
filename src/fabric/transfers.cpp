// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
__global__ [[loom::workgroup_size(256, 1, 1), loom::workgroup_count_range(1, 16777216, 1, 1, 1, 1)]]
void hrx_copy(unsigned char* destination, const unsigned char* source, unsigned long count) {
  const unsigned long offset = (unsigned long)blockIdx.x * 256ul + threadIdx.x;
  if (offset < count) destination[offset] = source[offset];
}
__global__ [[loom::workgroup_size(256, 1, 1), loom::workgroup_count_range(1, 16777216, 1, 1, 1, 1)]]
void hrx_fill(unsigned char* destination, unsigned long count, unsigned value) {
  const unsigned long offset = (unsigned long)blockIdx.x * 256ul + threadIdx.x;
  if (offset < count) destination[offset] = (unsigned char)value;
}
__global__ [[loom::workgroup_size(256, 1, 1), loom::workgroup_count_range(1, 16777216, 1, 1, 1, 1)]]
void hrx_copy_words(unsigned long* destination, const unsigned long* source, unsigned long count) {
  const unsigned long offset = (unsigned long)blockIdx.x * 256ul + threadIdx.x;
  if (offset < count) destination[offset] = source[offset];
}
__global__ [[loom::workgroup_size(256, 1, 1), loom::workgroup_count_range(1, 16777216, 1, 1, 1, 1)]]
void hrx_fill_words(unsigned long* destination, unsigned long count, unsigned value) {
  const unsigned long offset = (unsigned long)blockIdx.x * 256ul + threadIdx.x;
  if (offset < count) destination[offset] = (unsigned long)(value & 255u) * 0x0101010101010101ul;
}
