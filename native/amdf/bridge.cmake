# Included after the upstream project() call, before its targets are declared.
if(PROJECT_NAME STREQUAL "HRX" AND NOT TARGET hrx_fabric)
  # Reuse pure image/packet helpers directly: do not enable an HSA HAL driver
  # merely to get its conditionally declared utility targets.
  cmake_language(DEFER CALL iree_configure_rocm_hsa_runtime_headers)
  set(_hrx_gpu_helpers "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/drivers/amdgpu")
  add_library(hrx_fabric SHARED
    "${CMAKE_CURRENT_LIST_DIR}/gpu.c"
    "${CMAKE_CURRENT_LIST_DIR}/xdna.c"
    "${_hrx_gpu_helpers}/util/hsaco_metadata.c"
    "${_hrx_gpu_helpers}/util/pm4_dispatch.c"
    "${_hrx_gpu_helpers}/target/code_object.c"
    "${_hrx_gpu_helpers}/target/identity.c")
  target_include_directories(hrx_fabric PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
  target_link_libraries(hrx_fabric PRIVATE
    iree::base iree::hal iree::hal::utils::elf_format iree::third_party::hsa_runtime_headers
    iree::experimental::xdna::executable
    iree::experimental::xdna::amdf_status
    iree::hal::drivers::amd::xdna::image::aie2p::npu2)
  target_link_options(hrx_fabric PRIVATE "-Wl,-z,defs")
  set_target_properties(hrx_fabric PROPERTIES
    C_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
endif()
