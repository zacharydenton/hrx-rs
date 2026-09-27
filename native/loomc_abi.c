// Copyright 2026 hrx-rs contributors
// SPDX-License-Identifier: MIT
#include "loomc/diagnostic.h"

// Upstream extended this unversioned result view. Older bundles lack this
// optional probe: clients must read only the original prefix in that case.
LOOMC_API_EXPORT loomc_host_size_t loomc_hrx_diagnostic_size(void) {
  return sizeof(loomc_diagnostic_t);
}
