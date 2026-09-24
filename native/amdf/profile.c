// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#include "profile.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include <errno.h>
#include <string.h>
#if defined(__linux__)
#include <drm/amdgpu_drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

int hrx_fabric_gpu_profile_clock(uint32_t device_major, uint32_t device_minor,
                                 uint64_t *hz) {
  if (!hz)
    return EINVAL;
#if defined(__linux__)
  char path[64];
  snprintf(path, sizeof(path), "/dev/char/%u:%u", device_major, device_minor);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return errno;
  struct stat info;
  int error = 0;
  if (fstat(fd, &info))
    error = errno;
  else if (!S_ISCHR(info.st_mode) || major(info.st_rdev) != device_major ||
           minor(info.st_rdev) != device_minor)
    error = ENODEV;
  struct drm_amdgpu_info_device device = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&device,
      .return_size = sizeof(device),
      .query = AMDGPU_INFO_DEV_INFO,
  };
  if (!error && ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &query))
    error = errno;
  close(fd);
  if (error)
    return error;
  if (!device.gpu_counter_freq)
    return ENOTSUP;
  // DRM reports the GPU counter frequency in kHz, independently of SCLK.
  *hz = (uint64_t)device.gpu_counter_freq * 1000;
  return 0;
#else
  return ENOTSUP;
#endif
}

int hrx_fabric_gpu_profile_marker(uint64_t address, uint32_t *words,
                                  uint32_t capacity, uint32_t *count) {
  if (!address || (address & 7) || !words || !count || capacity < 6)
    return EINVAL;
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(&slot, &builder);
  if (!iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
          &builder,
          IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM,
          (void *)(uintptr_t)address))
    return ENOTSUP;
  uint32_t emitted = iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder);
  if (emitted > capacity)
    return ENOSPC;
  memcpy(words, slot.dwords, emitted * sizeof(uint32_t));
  *count = emitted;
  return 0;
}
