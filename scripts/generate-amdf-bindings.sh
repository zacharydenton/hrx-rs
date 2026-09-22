#!/usr/bin/env bash
# Maintainer-only; downstream Cargo builds need no native headers or bindgen.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HRX_SOURCE:?Set HRX_SOURCE to the pinned hrx-system checkout}"
bindgen native/amdf.h --allowlist-function amdf_query_api \
  --allowlist-type 'amdf_.*' --allowlist-var '(AMDF|HRX_AMDF)_.*' \
  --no-prepend-enum-name --with-derive-default \
  --dynamic-loading Amdf --dynamic-link-require-all \
  --output src/fabric/ffi.rs -- -I"$HRX_SOURCE/libamdf/include"
bindgen native/amdf/bridge.h --allowlist-function 'hrx_fabric_.*' \
  --allowlist-type 'hrx_fabric_.*' --with-derive-default \
  --dynamic-loading Bridge --dynamic-link-require-all \
  --output src/fabric/bridge_ffi.rs
