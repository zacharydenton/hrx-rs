// SPDX-License-Identifier: Apache-2.0
// Derived from dinov3-xdna2 xrt-shim; maintained with HRX.
#include "shim.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

struct dv_ctx {
  xrt::device dev;
  xrt::xclbin xclbin;
  xrt::hw_context ctx;
  xrt::kernel kern;
};

struct dv_bo {
  xrt::bo bo;
  size_t size;
};

namespace {
thread_local char last_error[1024] = {};
constexpr int DV_ERROR = -1;
constexpr int DV_RUN_RETAINED = -2;

void report_current_exception(const char *operation) noexcept {
  try {
    throw;
  } catch (const std::exception &e) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", operation, e.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s: unknown C++ exception", operation);
  }
}

// abort() is synchronous. Only release the native run after XRT confirms abort;
// retaining it on failure is preferable to freeing resources still used by DMA.
int abort_and_release(xrt::run *run, int result, const char *operation) noexcept {
  try {
    (void)run->abort();
    delete run;
    return result;
  } catch (...) {
    report_current_exception(operation);
    return DV_RUN_RETAINED;
  }
}
} // namespace

extern "C" dv_ctx *dv_ctx_create(int dev_idx, const char *xclbin_path) {
  try {
    if (!xclbin_path) return nullptr;
    auto c = std::make_unique<dv_ctx>();
    c->dev = xrt::device(dev_idx);
    c->xclbin = xrt::xclbin(std::string(xclbin_path));
    c->dev.register_xclbin(c->xclbin);
    c->ctx = xrt::hw_context(c->dev, c->xclbin.get_uuid());
    c->kern = xrt::kernel(c->ctx, "MLIR_AIE");
    return c.release();
  } catch (...) {
    report_current_exception("dv_ctx_create");
    return nullptr;
  }
}

extern "C" void dv_ctx_free(dv_ctx *c) { delete c; }

extern "C" int dv_kernel_group_id(dv_ctx *c, int argidx) {
  try {
    if (!c) return DV_ERROR;
    return static_cast<int>(c->kern.group_id(argidx));
  } catch (...) {
    report_current_exception("dv_kernel_group_id");
    return DV_ERROR;
  }
}

extern "C" dv_bo *dv_bo_alloc(dv_ctx *c, size_t nbytes, int kind, int group_id) {
  try {
    if (!c || nbytes == 0) return nullptr;
    auto flags = (kind == 1) ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
    auto result = std::make_unique<dv_bo>(dv_bo{xrt::bo(c->dev, nbytes, flags, group_id), nbytes});
    std::memset(result->bo.map<void *>(), 0, nbytes);
    return result.release();
  } catch (...) {
    report_current_exception("dv_bo_alloc");
    return nullptr;
  }
}

// Import host pages the caller owns as a BO, with no copy. `userptr` must be page-aligned.
// On an APU this is what lets one allocation back both the NPU (this BO) and the GPU (via
// the host-pointer import in the GPU runtime), so a handoff costs a cache flush, not a copy.
extern "C" dv_bo *dv_bo_import(dv_ctx *c, void *userptr, size_t nbytes, int kind,
                               int group_id) {
  try {
    if (!c || !userptr || nbytes == 0) return nullptr;
    auto flags = (kind == 1) ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
    return new dv_bo{xrt::bo(c->dev, userptr, nbytes, flags, group_id), nbytes};
  } catch (...) {
    report_current_exception("dv_bo_import");
    return nullptr;
  }
}

// Import a dma-buf exported by another driver (e.g. the GPU's HSA runtime) as a BO.
// This is the zero-copy bridge: the NPU addresses memory the GPU allocated, with no copy
// and no second physical allocation.
extern "C" dv_bo *dv_bo_import_dmabuf(dv_ctx *c, int fd, size_t nbytes) {
  try {
    if (!c || fd < 0 || nbytes == 0) return nullptr;
    auto bo = xrt::bo(c->dev, static_cast<xrt::bo::export_handle>(fd));
    if (bo.size() < nbytes) return nullptr;
    return new dv_bo{std::move(bo), nbytes};
  } catch (...) {
    report_current_exception("dv_bo_import_dmabuf");
    return nullptr;
  }
}

// Sub-buffer: an offset+size VIEW into a parent BO (no copy). Used for offset-dispatch: pack A once into a big
// resident BO, then dispatch the M-baked inst per 2048-row chunk via sub-BOs at chunk offsets (the 12.6-TFLOPS path).
extern "C" dv_bo *dv_bo_suballoc(dv_bo *parent, size_t size, size_t offset) {
  try {
    if (!parent || size == 0 || offset > parent->size || size > parent->size - offset)
      return nullptr;
    return new dv_bo{xrt::bo(parent->bo, size, offset), size};
  } catch (...) {
    report_current_exception("dv_bo_suballoc");
    return nullptr;
  }
}

extern "C" int dv_bo_write(dv_bo *b, const void *src, size_t nbytes) {
  try {
    if (!b || (!src && nbytes != 0) || nbytes > b->size) return DV_ERROR;
    b->bo.write(src, nbytes, 0);
    b->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0);
    return 0;
  } catch (...) {
    report_current_exception("dv_bo_write");
    return DV_ERROR;
  }
}

extern "C" int dv_bo_read(dv_bo *b, void *dst, size_t nbytes) {
  try {
    if (!b || (!dst && nbytes != 0) || nbytes > b->size) return DV_ERROR;
    b->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, nbytes, 0);
    b->bo.read(dst, nbytes, 0);
    return 0;
  } catch (...) {
    report_current_exception("dv_bo_read");
    return DV_ERROR;
  }
}

extern "C" void dv_bo_free(dv_bo *b) { delete b; }

extern "C" void *dv_bo_map(dv_bo *b) {
  try {
    return b ? b->bo.map() : nullptr;
  } catch (...) {
    report_current_exception("dv_bo_map");
    return nullptr;
  }
}

extern "C" int dv_bo_sync(dv_bo *b, int to_device, size_t nbytes) {
  try {
    if (!b || nbytes > b->size) return DV_ERROR;
    b->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE,
               nbytes, 0);
    return 0;
  } catch (...) {
    report_current_exception("dv_bo_sync");
    return DV_ERROR;
  }
}

extern "C" int dv_run(dv_ctx *c, dv_bo *insts, uint32_t insts_nbytes, dv_bo *a,
                      dv_bo *b, dv_bo *cbo, uint32_t timeout_ms) {
  auto *run = static_cast<xrt::run *>(
      dv_run_start(c, insts, insts_nbytes, a, b, cbo));
  if (!run) return DV_ERROR;
  const int state = dv_run_wait(run, timeout_ms);
  if (state == DV_RUN_RETAINED) {
    // This legacy synchronous entry point cannot retain the caller's BOs. Fail
    // the process rather than let an active DMA outlive them.
    std::fprintf(stderr, "dv_run: unable to abort run; terminating to preserve BO lifetime\n");
    std::terminate();
  }
  return state;
}

// ASYNC dispatch (Stage C — CPU/NPU overlap): submit the kernel and return immediately with an opaque
// run handle (heap xrt::run); the caller does host work, then dv_run_wait blocks on the handle. XRT's
// kernel(...) call returns an xrt::run that is already submitted/running — wait() is the only blocking part.
extern "C" void *dv_run_start(dv_ctx *c, dv_bo *insts, uint32_t insts_nbytes, dv_bo *a, dv_bo *b, dv_bo *cbo) {
  try {
    if (!c || !insts || !a || !b || !cbo || insts_nbytes == 0 || insts_nbytes % 4 != 0 || insts_nbytes > insts->size)
      return nullptr;
    return new xrt::run(c->kern(3, insts->bo, insts_nbytes / 4, a->bo, b->bo, cbo->bo));
  } catch (...) {
    report_current_exception("dv_run_start");
    return nullptr;
  }
}

extern "C" int dv_run_wait(void *handle, uint32_t timeout_ms) {
  if (!handle) return DV_ERROR;
  auto *r = static_cast<xrt::run *>(handle);
  try {
    const auto state = r->wait(std::chrono::milliseconds(timeout_ms));
    const int result = static_cast<int>(state);
    if (state == ERT_CMD_STATE_TIMEOUT)
      return abort_and_release(r, result, "dv_run_wait abort");
    delete r;
    return result;
  } catch (...) {
    report_current_exception("dv_run_wait");
    return abort_and_release(r, DV_ERROR, "dv_run_wait recovery abort");
  }
}

extern "C" int dv_run_cancel(void *handle) {
  if (!handle) return 0;
  return abort_and_release(static_cast<xrt::run *>(handle), 0, "dv_run_cancel");
}

extern "C" uint32_t hrx_npu_abi_version() { return 1; }

extern "C" void *dv_run_start_args(dv_ctx *c, dv_bo *insts, uint32_t words,
                                   dv_bo *const *args, size_t count) {
  try {
    if (!c || !insts || words == 0 || size_t(words) * 4 > insts->size || count > 64)
      return nullptr;
    auto run = std::make_unique<xrt::run>(c->kern);
    run->set_arg(0, uint32_t(3));
    run->set_arg(1, insts->bo);
    run->set_arg(2, words);
    for (size_t i = 0; i < count; ++i) {
      if (!args[i]) return nullptr;
      run->set_arg(int(i + 3), args[i]->bo);
    }
    run->start();
    return run.release();
  } catch (...) {
    report_current_exception("dv_run_start_args");
    return nullptr;
  }
}

struct dv_prepared_run { xrt::run run; bool running = false; };
extern "C" void *dv_run_prepare(dv_ctx *c, dv_bo *insts, uint32_t words,
                                dv_bo *const *args, size_t count) {
  try {
    if (!c || !insts || words == 0 || size_t(words) * 4 > insts->size || count > 64)
      return nullptr;
    auto prepared = std::make_unique<dv_prepared_run>(dv_prepared_run{xrt::run(c->kern)});
    prepared->run.set_arg(0, uint32_t(3));
    prepared->run.set_arg(1, insts->bo);
    prepared->run.set_arg(2, words);
    for (size_t i = 0; i < count; ++i) {
      if (!args[i]) return nullptr;
      prepared->run.set_arg(int(i + 3), args[i]->bo);
    }
    return prepared.release();
  } catch (...) { report_current_exception("dv_run_prepare"); return nullptr; }
}
extern "C" int dv_prepared_execute(void *handle, uint32_t timeout_ms) {
  last_error[0] = 0;
  auto p = static_cast<dv_prepared_run *>(handle);
  try {
    if (!p || p->running) return DV_ERROR;
    // Mark running before start: an exception may follow partial submission.
    p->running = true;
    p->run.start();
    auto state = p->run.wait(std::chrono::milliseconds(timeout_ms));
    if (state != ERT_CMD_STATE_COMPLETED) p->run.abort();
    p->running = false;
    return state == ERT_CMD_STATE_COMPLETED ? int(state) : DV_ERROR;
  } catch (...) {
    report_current_exception("dv_prepared_execute");
    try { if (p) { p->run.abort(); p->running = false; } }
    catch (...) { return DV_RUN_RETAINED; }
    return DV_ERROR;
  }
}
extern "C" int dv_prepared_free(void *handle) {
  auto p = static_cast<dv_prepared_run *>(handle);
  if (!p) return 0;
  if (p->running) {
    try { p->run.abort(); }
    catch (...) { report_current_exception("dv_prepared_free"); return DV_RUN_RETAINED; }
  }
  delete p;
  return 0;
}
extern "C" uint64_t dv_bo_address(dv_bo *bo) {
  try { return bo ? bo->bo.address() : 0; }
  catch (...) { report_current_exception("dv_bo_address"); return 0; }
}

extern "C" const char *dv_last_error() { return last_error; }
