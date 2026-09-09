#!/usr/bin/env bash
# Development-only; consumers need neither bindgen nor compiler headers.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LOOM_SOURCE:?Set LOOM_SOURCE to the pinned hrx-system checkout}"
bindgen native/loomc.h \
  --allowlist-function 'loomc_(allocator_system|allocator_free|status_format|status_free|context_create|context_release|workspace_create|workspace_release|workspace_trim|source_create|source_release|module_release|link_index_builder_create|link_index_builder_add_source|link_index_builder_finish|link_index_builder_release|link_index_release|linker_create|linker_release|link_module|compiler_create|compiler_release|compile_module|pass_program_create_from_target_pipeline|pass_program_release|target_environment_create_amdgpu|target_environment_release|target_profile_create_amdgpu|target_profile_release|emit_module|result_release|result_succeeded|result_diagnostic_count|result_diagnostic_at|result_artifact_count|result_artifact_at|byte_sequence_length|byte_sequence_clone)' \
  --allowlist-type 'loomc_.*_flag_bits_e' --allowlist-type 'loomc_(context_target_options_t|target_specialization_options_t|target_specialization_t|artifact_manifest_options_t|amdgpu_emit_options_t)' --allowlist-var 'LOOMC_.*' --no-prepend-enum-name --with-derive-default \
  --dynamic-loading Loomc --dynamic-link-require-all \
  --output src/loom/ffi.rs -- -I"$LOOM_SOURCE/loom/binding/c/include"
