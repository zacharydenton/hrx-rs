// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#include "bridge.h"
#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/target/code_object.h"
#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

static _Thread_local char last_error[2048];
const char* hrx_fabric_error(void) { return last_error; }
static int fail(const char* message) {
  snprintf(last_error, sizeof(last_error), "%s", message);
  return 1;
}
int hrx_fabric_status(iree_status_t value) {
  if (iree_status_is_ok(value)) return 0;
  iree_host_size_t length = 0;
  iree_status_format(value, sizeof(last_error), last_error, &length);
  last_error[sizeof(last_error) - 1] = 0;
  iree_status_ignore(value);
  return 1;
}
struct hrx_fabric_gpu_image {
  uint8_t* data;
  size_t size;
  Elf64_Ehdr elf;
  uint64_t low, high, descriptor;
  iree_hal_amdgpu_hsaco_metadata_t metadata;
  const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel;
  iree_hal_amdgpu_kernel_descriptor_t descriptor_data;
};
static int range(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size - offset;
}
static int add_signed(uint64_t base, int64_t addend, uint64_t* out) {
  const __int128 value = (__int128)base + addend;
  if (value < 0 || value > UINT64_MAX) return fail("ELF address arithmetic overflows");
  *out = (uint64_t)value;
  return 0;
}
static int section(const hrx_fabric_gpu_image* image, uint32_t index,
                   Elf64_Shdr* out) {
  if (index >= image->elf.e_shnum) return fail("ELF section index is out of bounds");
  memcpy(out, image->data + image->elf.e_shoff + index * sizeof(*out), sizeof(*out));
  if (out->sh_type != SHT_NOBITS && !range(out->sh_offset, out->sh_size, image->size))
    return fail("ELF section is out of bounds");
  return 0;
}
static int symbol_at(const hrx_fabric_gpu_image* image, const Elf64_Shdr* table,
                     uint32_t index, Elf64_Sym* out) {
  if ((table->sh_type != SHT_SYMTAB && table->sh_type != SHT_DYNSYM) ||
      !range(table->sh_offset, table->sh_size, image->size) ||
      table->sh_entsize != sizeof(*out) || table->sh_size % sizeof(*out) ||
      index >= table->sh_size / sizeof(*out))
    return fail("ELF symbol index is out of bounds");
  memcpy(out, image->data + table->sh_offset + (uint64_t)index * sizeof(*out), sizeof(*out));
  return 0;
}
static int find_symbol(const hrx_fabric_gpu_image* image, iree_string_view_t name,
                       uint64_t* address) {
  for (uint32_t i = 0; i < image->elf.e_shnum; ++i) {
    Elf64_Shdr table, strings;
    if (section(image, i, &table)) return 1;
    if (table.sh_type != SHT_DYNSYM && table.sh_type != SHT_SYMTAB) continue;
    if (table.sh_entsize != sizeof(Elf64_Sym) || table.sh_size % sizeof(Elf64_Sym))
      return fail("invalid ELF symbol table");
    if (section(image, table.sh_link, &strings) || strings.sh_type != SHT_STRTAB)
      return fail("invalid ELF symbol string table");
    for (uint64_t n = 0; n < table.sh_size / sizeof(Elf64_Sym); ++n) {
      Elf64_Sym symbol;
      if (n > UINT32_MAX || symbol_at(image, &table, (uint32_t)n, &symbol)) return 1;
      if (symbol.st_name >= strings.sh_size) return fail("invalid ELF symbol name");
      const char* text = (const char*)image->data + strings.sh_offset + symbol.st_name;
      const size_t available = strings.sh_size - symbol.st_name;
      if (name.size < available && !memcmp(text, name.data, name.size) && text[name.size] == 0) {
        if (symbol.st_shndx == SHN_UNDEF) return fail("unresolved kernel symbol");
        *address = symbol.st_value;
        return 0;
      }
    }
  }
  return fail("kernel descriptor symbol is absent");
}
void hrx_fabric_gpu_image_close(hrx_fabric_gpu_image* image) {
  if (!image) return;
  iree_hal_amdgpu_hsaco_metadata_deinitialize(&image->metadata);
  free(image->data);
  free(image);
}
int hrx_fabric_gpu_image_open(const uint8_t* data, size_t size, const char* symbol,
                             hrx_fabric_gpu_image** out, hrx_fabric_gpu_info* info) {
  if (!data || !symbol || !out || !info || size < sizeof(Elf64_Ehdr))
    return fail("incomplete GPU image arguments");
  *out = NULL;
  Elf64_Ehdr elf;
  memcpy(&elf, data, sizeof(elf));
  if (memcmp(elf.e_ident, ELFMAG, SELFMAG) || elf.e_ident[EI_CLASS] != ELFCLASS64 ||
      elf.e_ident[EI_DATA] != ELFDATA2LSB || elf.e_machine != EM_AMDGPU || elf.e_type != ET_DYN ||
      elf.e_phentsize != sizeof(Elf64_Phdr) || elf.e_shentsize != sizeof(Elf64_Shdr) ||
      !range(elf.e_phoff, (uint64_t)elf.e_phnum * sizeof(Elf64_Phdr), size) ||
      !range(elf.e_shoff, (uint64_t)elf.e_shnum * sizeof(Elf64_Shdr), size))
    return fail("expected a complete ELF64 little-endian AMDGPU shared code object");
  hrx_fabric_gpu_image* image = calloc(1, sizeof(*image));
  if (!image) return fail("GPU image allocation failed");
  image->elf = elf;
  image->size = size;
  image->data = malloc(size);
  if (!image->data) { free(image); return fail("GPU image allocation failed"); }
  memcpy(image->data, data, size);
  image->low = UINT64_MAX;
  for (uint32_t i = 0; i < elf.e_phnum; ++i) {
    Elf64_Phdr ph;
    memcpy(&ph, data + elf.e_phoff + i * sizeof(ph), sizeof(ph));
    if (ph.p_type != PT_LOAD) continue;
    if (!range(ph.p_offset, ph.p_filesz, size) || ph.p_filesz > ph.p_memsz ||
        ph.p_vaddr > UINT64_MAX - ph.p_memsz) {
      hrx_fabric_gpu_image_close(image); return fail("invalid ELF load segment");
    }
    if (ph.p_vaddr < image->low) image->low = ph.p_vaddr;
    if (ph.p_vaddr + ph.p_memsz > image->high) image->high = ph.p_vaddr + ph.p_memsz;
  }
  if (image->high <= image->low || image->high - image->low > (UINT64_C(1) << 30)) {
    hrx_fabric_gpu_image_close(image); return fail("invalid GPU image allocation size");
  }
  iree_const_byte_span_t bytes = iree_make_const_byte_span(image->data, size);
  iree_hal_amdgpu_target_identity_t identity;
  if (hrx_fabric_status(iree_hal_amdgpu_code_object_identity_from_elf(bytes, &identity)) ||
      identity.version.major != 11 || identity.version.minor != 5 || identity.version.stepping != 1) {
    hrx_fabric_gpu_image_close(image); return fail("GPU code object must target gfx1151");
  }
  if (hrx_fabric_status(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(bytes, iree_allocator_system(), &image->metadata))) {
    hrx_fabric_gpu_image_close(image); return 1;
  }
  for (size_t i = 0; i < image->metadata.kernel_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* candidate = &image->metadata.kernels[i];
    if (iree_string_view_equal(candidate->reflection_name, iree_make_cstring_view(symbol))) image->kernel = candidate;
  }
  if (!image->kernel || find_symbol(image, image->kernel->symbol_name, &image->descriptor) ||
      image->descriptor < image->low || !range(image->descriptor - image->low, 64, image->high - image->low)) {
    hrx_fabric_gpu_image_close(image); return fail("kernel descriptor is missing or out of range");
  }
  int found = 0;
  for (uint32_t i = 0; i < elf.e_phnum; ++i) {
    Elf64_Phdr ph;
    memcpy(&ph, data + elf.e_phoff + i * sizeof(ph), sizeof(ph));
    if (ph.p_type == PT_LOAD && image->descriptor >= ph.p_vaddr &&
        range(image->descriptor - ph.p_vaddr, 64, ph.p_filesz)) {
      memcpy(&image->descriptor_data, data + ph.p_offset + image->descriptor - ph.p_vaddr, 64);
      found = 1; break;
    }
  }
  if (!found || image->kernel->arg_count > UINT32_MAX) {
    hrx_fabric_gpu_image_close(image); return fail("kernel descriptor has no file storage");
  }
  uint64_t entry;
  if (add_signed(image->descriptor, image->descriptor_data.kernel_code_entry_byte_offset, &entry)) {
    hrx_fabric_gpu_image_close(image); return 1;
  }
  found = 0;
  for (uint32_t i = 0; i < elf.e_phnum; ++i) {
    Elf64_Phdr ph;
    memcpy(&ph, data + elf.e_phoff + i * sizeof(ph), sizeof(ph));
    if (ph.p_type == PT_LOAD && (ph.p_flags & PF_X) && entry >= ph.p_vaddr &&
        range(entry - ph.p_vaddr, 4, ph.p_filesz)) { found = 1; break; }
  }
  if (!found || (entry & 255)) {
    hrx_fabric_gpu_image_close(image); return fail("kernel entry is not aligned executable file storage");
  }
  *info = (hrx_fabric_gpu_info){ .storage_bytes = image->high - image->low,
      .descriptor_offset = image->descriptor - image->low,
      .kernarg_bytes = image->kernel->kernarg_segment_size,
      .private_bytes = image->descriptor_data.private_segment_fixed_size,
      .wave_size = (image->descriptor_data.kernel_code_properties & IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) ? 32 : 64,
      .local_bytes = image->kernel->group_segment_fixed_size,
      .argument_count = (uint32_t)image->kernel->arg_count };
  memcpy(info->workgroup_size, image->kernel->required_workgroup_size, sizeof(info->workgroup_size));
  *out = image;
  return 0;
}
int hrx_fabric_gpu_argument_info(const hrx_fabric_gpu_image* image, uint32_t index,
                                hrx_fabric_gpu_argument* out) {
  if (!image || !out || index >= image->kernel->arg_count) return fail("argument index is out of range");
  const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &image->kernel->args[index];
  if (!range(arg->offset, arg->size, image->kernel->kernarg_segment_size)) return fail("argument storage is out of range");
  *out = (hrx_fabric_gpu_argument){ .kind = arg->kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE ? 1 :
      arg->kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER ? 2 : 0,
      .offset = arg->offset, .size = arg->size };
  return 0;
}
int hrx_fabric_gpu_image_load(const hrx_fabric_gpu_image* image, uint8_t* storage,
                             size_t size, uint64_t address) {
  if (!image || !storage || size < image->high - image->low ||
      address > UINT64_MAX - size || address < image->low) return fail("GPU storage is invalid");
  memset(storage, 0, image->high - image->low);
  for (uint32_t i = 0; i < image->elf.e_phnum; ++i) {
    Elf64_Phdr ph;
    memcpy(&ph, image->data + image->elf.e_phoff + i * sizeof(ph), sizeof(ph));
    if (ph.p_type == PT_LOAD) memcpy(storage + ph.p_vaddr - image->low, image->data + ph.p_offset, ph.p_filesz);
  }
  uint64_t bias = address - image->low;
  for (uint32_t i = 0; i < image->elf.e_shnum; ++i) {
    Elf64_Shdr table;
    if (section(image, i, &table)) return 1;
    if (table.sh_type == SHT_REL) return fail("REL relocations are unsupported");
    if (table.sh_type != SHT_RELA || !(table.sh_flags & SHF_ALLOC)) continue;
    if (table.sh_entsize != sizeof(Elf64_Rela) || table.sh_size % sizeof(Elf64_Rela)) return fail("invalid relocation table");
    Elf64_Shdr symbols;
    if (section(image, table.sh_link, &symbols)) return 1;
    for (uint64_t n = 0; n < table.sh_size / sizeof(Elf64_Rela); ++n) {
      Elf64_Rela reloc;
      memcpy(&reloc, image->data + table.sh_offset + n * sizeof(reloc), sizeof(reloc));
      uint32_t type = ELF64_R_TYPE(reloc.r_info);
      if (type == 0) continue;
      uint64_t value;
      if (type == 13) { if (add_signed(bias, reloc.r_addend, &value)) return 1; }
      else {
        Elf64_Sym symbol;
        if (symbol_at(image, &symbols, ELF64_R_SYM(reloc.r_info), &symbol)) return 1;
        if (symbol.st_shndx == SHN_UNDEF) return fail("unresolved external GPU relocation");
        uint64_t base = symbol.st_shndx == SHN_ABS ? 0 : bias;
        if (symbol.st_value > UINT64_MAX - base) return fail("ELF symbol address overflows");
        if (add_signed(base + symbol.st_value, reloc.r_addend, &value)) return 1;
      }
      size_t width = (type == 3 || type == 13) ? 8 : 4;
      if (reloc.r_offset < image->low || !range(reloc.r_offset - image->low, width, size)) return fail("relocation destination is out of range");
      void* destination = storage + reloc.r_offset - image->low;
      if (type == 3 || type == 13) memcpy(destination, &value, 8);
      else if (type == 1 || type == 2 || type == 6) {
        if (type == 6 && value > UINT32_MAX) return fail("32-bit relocation overflows");
        uint32_t part = type == 2 ? (uint32_t)(value >> 32) : (uint32_t)value;
        memcpy(destination, &part, 4);
      } else return fail("unsupported AMDGPU relocation");
    }
  }
  return 0;
}
int hrx_fabric_gpu_dispatch(const hrx_fabric_gpu_image* image, uint64_t image_address,
    const uint16_t block[3], const uint32_t grid[3], uint64_t kernarg_address,
    const uint8_t* kernarg, size_t kernarg_size, uint64_t scratch_address,
    uint64_t scratch_length, uint32_t scratch_waves, uint32_t shader_engines, uint32_t* words, uint32_t capacity, uint32_t* count) {
  if (!image || !block || !grid || !words || !count || capacity < 80 ||
      kernarg_size < image->kernel->kernarg_segment_size || (!kernarg && kernarg_size)) return fail("incomplete dispatch arguments");
  for (int i = 0; i < 3; ++i) {
    if (!block[i] || !grid[i] || (image->kernel->has_required_workgroup_size && block[i] != image->kernel->required_workgroup_size[i]) ||
        grid[i] > UINT32_MAX / block[i]) return fail("invalid dispatch shape");
  }
  if ((uint64_t)block[0] * block[1] * block[2] > image->kernel->max_flat_workgroup_size) return fail("workgroup exceeds kernel limit");
  // The upstream helper validates all other descriptor requirements. This
  // gfx1151 extension supplies explicit per-invocation scratch below.
  iree_hal_amdgpu_kernel_descriptor_t descriptor = image->descriptor_data;
  const uint32_t private_bytes = descriptor.private_segment_fixed_size;
  const uint32_t scratch_enabled = descriptor.compute_pgm_rsrc2 &
      IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT;
  descriptor.private_segment_fixed_size = 0;
  descriptor.compute_pgm_rsrc2 &= ~IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT;
  iree_hal_amdgpu_pm4_dispatch_launch_state_t state;
  if (hrx_fabric_status(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize(
      (iree_hal_amdgpu_gfxip_version_t){11, 5, 1}, &descriptor,
      image_address + image->descriptor - image->low, block, 0, &state))) return 1;
  if (private_bytes || scratch_enabled) {
    const uint32_t lanes = (descriptor.kernel_code_properties &
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) ? 32 : 64;
    const uint64_t wave_units = ((uint64_t)private_bytes * lanes + 255) / 256;
    if (!private_bytes || !scratch_address || (scratch_address & 255) ||
        !shader_engines || !scratch_waves || scratch_waves % shader_engines ||
        scratch_waves / shader_engines > 0xfff || wave_units > 0x7fff ||
        scratch_length < wave_units * 256 * scratch_waves)
      return fail("invalid gfx1151 scratch backing or wave limit");
    state.resources[1] |= IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT;
    // COMPUTE_PGM_LO+4/+5 are DISPATCH_SCRATCH_BASE_LO/HI (address >> 8).
    // COMPUTE_TMPRING_SIZE uses waves per shader engine and 256-byte units.
    state.program[4] = (uint32_t)(scratch_address >> 8);
    state.program[5] = (uint32_t)(scratch_address >> 40);
    state.temporary_ring_size = (uint32_t)(wave_units << 12) | (scratch_waves / shader_engines);
  }
  uint32_t setup = 0, user = 0;
  if (hrx_fabric_status(iree_hal_amdgpu_pm4_dispatch_emit_setup(&state, capacity, words, &setup)) ||
      hrx_fabric_status(iree_hal_amdgpu_pm4_dispatch_emit_user_data(&state, kernarg_address, kernarg, capacity - setup, words + setup, &user))) return 1;
  uint32_t n = setup + user;
  if (capacity - n < 5) return fail("dispatch command storage exhausted");
  words[n++] = iree_hal_amdgpu_pm4_make_compute_header(IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT, 5);
  for (int i = 0; i < 3; ++i) words[n++] = grid[i] * block[i];
  words[n++] = state.dispatch_initiator;
  *count = n;
  return 0;
}
