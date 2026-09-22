// Portions adapted from experimental/xdna/iree-xdna-run.c in hrx-system.
// Copyright 2026 The IREE Authors and hrx-rs contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "bridge.h"
#include <stdlib.h>
#include <string.h>
#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "experimental/xdna/amdf_status.h"
#include "experimental/xdna/executable.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
extern int hrx_fabric_status(iree_status_t value);
struct hrx_fabric_xdna_program {
  iree_allocator_t host_allocator;
  const amdf_api_t* api;
  const amdf_xdna_api_t* xdna_api;
  amdf_instance_t* instance;
  amdf_endpoint_t* endpoint;
  amdf_device_t* device;
  amdf_memory_scope_t* memory_scope;
  amdf_xdna_context_t* context;
  struct {
    uint32_t count;
    iree_hal_amd_xdna_executable_storage_t* values;
    amdf_host_mapping_t** mappings;
  } storage;
  uint32_t queue_family_ordinal;
  iree_hal_amd_xdna_image_t* image;
  uint32_t entry_ordinal;
  amdf_kernel_queue_t* queue;
  amdf_xdna_kernel_command_t command;
};
typedef struct hrx_fabric_xdna_program iree_xdna_run_t;
static iree_status_t iree_xdna_run_select_memory_scope(iree_xdna_run_t* run) {
  uint32_t count = 0;
  const amdf_status_t count_status = run->api->instance_enumerate_memory_scopes(
      run->instance, 0, NULL, &count);
  if (count_status != amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        count_status, "instance_enumerate_memory_scopes(count)"));
  }
  iree_host_size_t scopes_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scopes_size, IREE_STRUCT_FIELD(count, amdf_memory_scope_t*, NULL)));
  amdf_memory_scope_t** scopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(run->host_allocator, scopes_size, (void**)&scopes));
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(run->api->instance_enumerate_memory_scopes(
                                        run->instance, count, scopes, &count),
                                    "instance_enumerate_memory_scopes");
  for (uint32_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->memory_scope_query_info(scopes[i], &info),
        "memory_scope_query_info");
    if (iree_status_is_ok(status) &&
        info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      run->memory_scope = scopes[i];
      break;
    }
  }
  iree_allocator_free(run->host_allocator, scopes);
  if (iree_status_is_ok(status) && run->memory_scope == NULL) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "no system memory scope");
  }
  return status;
}
static iree_status_t iree_xdna_run_allocate_storage(
    iree_xdna_run_t* run, uint32_t use,
    const iree_xdna_elf_allocation_record_t* requirement) {
  iree_hal_amd_xdna_executable_storage_t* storage = &run->storage.values[use];
  const bool is_command =
      requirement->domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
  amdf_memory_scope_t* scope = run->memory_scope;
  if (is_command) {
    uint32_t count = 0;
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->xdna_api->context_enumerate_memory_scopes(run->context, 1, &scope,
                                                       &count),
        "xdna.context_enumerate_memory_scopes"));
  }
  const amdf_memory_address_kind_t address_kind =
      is_command ? AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE
                 : AMDF_MEMORY_ADDRESS_XDNA_DMA;
  const amdf_memory_device_access_t access = {
      .device = run->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        (is_command ? AMDF_MEMORY_ACCESS_EXECUTE : 0),
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << address_kind,
          },
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
      .ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN,
  };
  for (uint32_t ordinal = 0;; ++ordinal) {
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t status = run->api->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, &profile, &capabilities);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "no host-mappable allocation profile for XDNA storage");
    }
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "memory_scope_query_device_profile"));
    if ((profile.roles & (AMDF_MEMORY_PROFILE_ROLE_CREATE |
                          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) ==
            (AMDF_MEMORY_PROFILE_ROLE_CREATE |
             AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
      break;
    }
  }
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  if (!granularity) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "zero allocation granularity");
  uint64_t rounded_length = 0;
  if (!iree_checked_add_u64(requirement->byte_length, granularity - 1,
                            &rounded_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA allocation size overflows");
  }
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = (rounded_length / granularity) * granularity,
      .minimum_alignment = profile.allocation.minimum_alignment,
      .accesses = &access,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_create(scope, &create_info, &storage->memory),
      "memory_create(storage)"));
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_query_address(storage->memory, 0, address_kind,
                                     &storage->device_address),
      "memory_query_address(storage)"));
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = requirement->byte_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->memory_map(storage->memory, &map_info,
                           &run->storage.mappings[use]),
      "memory_map(storage)"));
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_query_info(run->storage.mappings[use],
                                        &mapping_info),
      "host_mapping_query_info(storage)"));
  storage->mapping = iree_make_byte_span(
      mapping_info.pointer, (iree_host_size_t)requirement->byte_length);
  return iree_ok_status();
}
static iree_status_t iree_xdna_run_prepare_storage(
    iree_xdna_run_t* run, const iree_xdna_elf_entry_record_t* entry) {
  iree_host_size_t total_size = 0;
  iree_host_size_t mappings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD(entry->allocation_use_count,
                        iree_hal_amd_xdna_executable_storage_t, NULL),
      IREE_STRUCT_FIELD(entry->allocation_use_count, amdf_host_mapping_t*,
                        &mappings_offset)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(run->host_allocator, total_size,
                                             (void**)&run->storage.values));
  run->storage.mappings =
      (amdf_host_mapping_t**)((uint8_t*)run->storage.values + mappings_offset);
  run->storage.count = entry->allocation_use_count;
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(run->image);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < run->storage.count;
       ++i) {
    const uint32_t ordinal = iree_hal_amd_xdna_image_tables_allocation_use(
        tables, entry->first_allocation_use + i);
    const iree_xdna_elf_allocation_record_t allocation =
        iree_hal_amd_xdna_image_tables_allocation(tables, ordinal);
    status = iree_xdna_run_allocate_storage(run, i, &allocation);
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_load(
      run->image, run->entry_ordinal, run->storage.count, run->storage.values));
  return iree_ok_status();
}

static iree_status_t prepare(iree_xdna_run_t* run, const uint8_t* bytes,
    size_t length, const char* symbol, uint16_t columns, uint32_t binding_count,
    const hrx_fabric_xdna_binding* bindings) {
  IREE_RETURN_IF_ERROR(iree_xdna_run_select_memory_scope(run));
  amdf_xdna_endpoint_info_t endpoint = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO, .structure_size = sizeof(endpoint)};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
    run->xdna_api->endpoint_query_info(run->endpoint, &endpoint), "XDNA endpoint"));
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
    iree_make_cstring_view(endpoint.target_id), columns, &target));
  amdf_xdna_device_info_t device = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO, .structure_size = sizeof(device)};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
    run->xdna_api->device_query_info(run->device, &device), "XDNA device"));
  if (!device.instruction.maximum_byte_length)
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no native instruction interface");
  target.instruction_alignment = device.instruction.address_alignment;
  amdf_endpoint_info_t info = {
    .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO, .structure_size = sizeof(info)};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
    run->api->endpoint_query_info(run->endpoint, &info), "endpoint"));
  run->queue_family_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < info.queue_family_count; ++i) {
    amdf_queue_family_info_t family = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO, .structure_size = sizeof(family)};
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->endpoint_query_queue_family_info(run->endpoint, i, &family), "queue family"));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL)) {
      run->queue_family_ordinal = i;
      break;
    }
  }
  if (run->queue_family_ordinal == UINT32_MAX)
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no native XDNA queue family");
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_allocator_clone(run->host_allocator,
    iree_make_const_byte_span(bytes, length), (void**)&span.data));
  span.data_length = length;
  iree_byte_sequence_t* sequence = NULL;
  iree_status_t status = iree_byte_sequence_create_from_span_move(
    &span, run->host_allocator, &sequence);
  iree_allocator_free(run->host_allocator, span.data);
  if (!iree_status_is_ok(status)) return status;
  status = iree_hal_amd_xdna_image_create(sequence, &target, run->host_allocator, &run->image);
  iree_byte_sequence_release(sequence);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_image_find_entry(
    run->image, iree_make_cstring_view(symbol), &run->entry_ordinal));
  const iree_xdna_elf_entry_record_t entry = iree_hal_amd_xdna_image_tables_entry(
    iree_hal_amd_xdna_image_tables(run->image), run->entry_ordinal);
  if (entry.binding_count != binding_count)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding count mismatch");
  const amdf_xdna_context_create_info_t context = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO, .structure_size = sizeof(context),
    .logical_column_count = columns,
    .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
    .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
    run->xdna_api->context_create(run->device, &context, &run->context), "context create"));
  IREE_RETURN_IF_ERROR(iree_xdna_run_prepare_storage(run, &entry));
  iree_hal_amd_xdna_executable_binding_t* resolved = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(run->host_allocator,
    (iree_host_size_t)binding_count * sizeof(*resolved), (void**)&resolved));
  for (uint32_t i = 0; iree_status_is_ok(status) && i < binding_count; ++i) {
    const hrx_fabric_xdna_binding* binding = &bindings[i];
    if (!binding->host_pointer || !binding->memory || !binding->length ||
        binding->offset > binding->allocation_length ||
        binding->length > binding->allocation_length - binding->offset) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid binding range");
      break;
    }
    iree_hal_buffer_t* buffer = NULL;
    status = iree_hal_heap_buffer_wrap(iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, IREE_HAL_MEMORY_ACCESS_READ |
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_UNALIGNED,
      IREE_HAL_BUFFER_USAGE_STORAGE, binding->allocation_length,
      iree_make_byte_span(binding->host_pointer, binding->allocation_length),
      iree_hal_buffer_release_callback_null(), run->host_allocator, &buffer);
    if (iree_status_is_ok(status)) resolved[i] = (iree_hal_amd_xdna_executable_binding_t){
      .buffer_ref = iree_hal_make_buffer_ref(buffer, binding->offset, binding->length),
      .memory = binding->memory, .memory_byte_offset = binding->offset,
      .device_address = binding->address};
  }
  if (iree_status_is_ok(status)) status = iree_hal_amd_xdna_executable_bind(
    run->image, run->entry_ordinal, run->storage.count, run->storage.values,
    binding_count, resolved);
  for (uint32_t i = 0; i < binding_count; ++i)
    iree_hal_buffer_release(resolved[i].buffer_ref.buffer);
  iree_allocator_free(run->host_allocator, resolved);
  IREE_RETURN_IF_ERROR(status);
  for (uint32_t i = 0; i < run->storage.count; ++i) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->host_mapping_cache_control(run->storage.mappings[i],
        AMDF_HOST_CACHE_OPERATION_FLUSH, 0, run->storage.values[i].mapping.data_length),
      "storage flush"));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_executable_query_invocation(
    run->image, run->entry_ordinal, run->storage.count, run->storage.values, &run->command));
  const amdf_xdna_kernel_queue_create_info_t queue = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO, .structure_size = sizeof(queue),
    .queue_family_ordinal = run->queue_family_ordinal};
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
    run->xdna_api->kernel_queue_create(run->context, &queue, &run->queue), "queue create");
}

int hrx_fabric_xdna_open(const void* api, const void* xdna_api, void* instance,
    void* endpoint, void* device, const uint8_t* bytes, size_t length,
    const char* symbol, uint16_t columns, uint32_t binding_count,
    const hrx_fabric_xdna_binding* bindings, hrx_fabric_xdna_program** out) {
  *out = NULL;
  hrx_fabric_xdna_program* run = calloc(1, sizeof(*run));
  if (!run) return hrx_fabric_status(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED));
  run->host_allocator = iree_allocator_system();
  run->api = api; run->xdna_api = xdna_api;
  run->instance = instance; run->endpoint = endpoint; run->device = device;
  // Publish even partially constructed ownership to the caller for cleanup.
  *out = run;
  return hrx_fabric_status(prepare(run, bytes, length, symbol, columns, binding_count, bindings));
}

static iree_status_t close_program(hrx_fabric_xdna_program* run) {
  if (run->queue) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->api->kernel_queue_destroy(run->queue), "queue destroy"));
    run->queue = NULL;
  }
  for (uint32_t i = 0; i < run->storage.count; ++i) {
    if (run->storage.mappings[i]) {
      IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->host_mapping_destroy(run->storage.mappings[i]), "mapping destroy"));
      run->storage.mappings[i] = NULL;
    }
    if (run->storage.values[i].memory) {
      IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        run->api->memory_destroy(run->storage.values[i].memory), "storage destroy"));
      run->storage.values[i].memory = NULL;
    }
  }
  iree_allocator_free(run->host_allocator, run->storage.values);
  run->storage.values = NULL; run->storage.count = 0;
  if (run->context) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      run->xdna_api->context_destroy(run->context), "context destroy"));
    run->context = NULL;
  }
  iree_hal_amd_xdna_image_destroy(run->image);
  free(run);
  return iree_ok_status();
}
int hrx_fabric_xdna_close(hrx_fabric_xdna_program* program) {
  return hrx_fabric_status(close_program(program));
}
uint64_t hrx_fabric_xdna_submit(hrx_fabric_xdna_program* program, uint64_t* submission) {
  const amdf_xdna_kernel_queue_submission_info_t info = {
    .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
    .structure_size = sizeof(info), .command_count = 1, .commands = &program->command};
  return program->xdna_api->kernel_queue_submit(program->queue, &info, submission);
}
uint64_t hrx_fabric_xdna_wait(hrx_fabric_xdna_program* program,
    uint64_t submission, uint64_t timeout_ns) {
  return program->api->kernel_queue_wait(program->queue, submission, timeout_ns, 0);
}
