// SPDX-License-Identifier: Apache-2.0
// Derived from dinov3-xdna2 xrt-shim; maintained with HRX.
// Minimal C ABI over XRT C++ for the dinov3-embed Rust engine.
// The ONLY C++ in the project; everything above this is Rust. Wraps the exact
// dispatch the python harness uses: device -> register xclbin -> hw_context ->
// kernel("MLIR_AIE") -> bo alloc/write/sync/run(3, insts, nbytes, A,B,C)/read.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct dv_ctx dv_ctx;   // device + xclbin + hw_context + kernel
typedef struct dv_bo dv_bo;     // one xrt::bo

// Create a context from an xclbin (registers it, builds hw_context + "MLIR_AIE"
// kernel). Returns NULL on failure (message on stderr).
dv_ctx *dv_ctx_create(int dev_idx, const char *xclbin_path);
void dv_ctx_free(dv_ctx *c);

// kernel.group_id(argidx) — the memory bank for a kernel argument.
int dv_kernel_group_id(dv_ctx *c, int argidx);

// kind: 0 = host_only, 1 = cacheable (instruction stream).
dv_bo *dv_bo_alloc(dv_ctx *c, size_t nbytes, int kind, int group_id);
dv_bo *dv_bo_import(dv_ctx *c, void *userptr, size_t nbytes, int kind, int group_id);
dv_bo *dv_bo_import_dmabuf(dv_ctx *c, int fd, size_t nbytes);
dv_bo *dv_bo_suballoc(dv_bo *parent, size_t size, size_t offset);
int dv_bo_write(dv_bo *b, const void *src, size_t nbytes); // write + sync TO_DEVICE
int dv_bo_read(dv_bo *b, void *dst, size_t nbytes);        // sync FROM_DEVICE + read
void dv_bo_free(dv_bo *b);

// Unified-memory zero-copy: map the BO to a host pointer (shared RAM on XDNA2),
// write/read it directly, and sync (cache flush, no copy). dir: 1=TO_DEVICE, 0=FROM.
void *dv_bo_map(dv_bo *b);
int dv_bo_sync(dv_bo *b, int to_device, size_t nbytes);

// Dispatch (opcode=3, insts, insts_nbytes, A, B, C) and wait. Returns the ERT
// command state (4 = COMPLETED), or -1 on exception/timeout.
int dv_run(dv_ctx *c, dv_bo *insts, uint32_t insts_nbytes, dv_bo *a, dv_bo *b,
           dv_bo *cbo, uint32_t timeout_ms);

// ASYNC dispatch (CPU/NPU overlap): submit and return an opaque run handle (non-blocking); do host work;
// then dv_run_wait(handle, timeout) blocks and frees it. A timeout is synchronously
// aborted before the handle is freed. Returns ERT state (4=COMPLETED), -1 on a
// contained XRT error, or -2 if abort failed and the handle had to be retained.
void *dv_run_start(dv_ctx *c, dv_bo *insts, uint32_t insts_nbytes, dv_bo *a, dv_bo *b, dv_bo *cbo);
int dv_run_wait(void *handle, uint32_t timeout_ms);
int dv_run_cancel(void *handle);

// HRX NPU ABI 1 extensions. All functions retain the current thread's last
// exception message until its next failing native operation.
uint32_t hrx_npu_abi_version(void);
const char *dv_last_error(void);
uint64_t dv_bo_address(dv_bo *b);
void *dv_run_start_args(dv_ctx *c, dv_bo *insts, uint32_t instruction_words,
                        dv_bo *const *args, size_t count);
void *dv_run_prepare(dv_ctx *c, dv_bo *insts, uint32_t instruction_words,
                     dv_bo *const *args, size_t count);
int dv_prepared_execute(void *run, uint32_t timeout_ms);
int dv_prepared_free(void *run);

#ifdef __cplusplus
}
#endif
