#!/usr/bin/env bash
# Release/development build only; Cargo never invokes a native compiler.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${1:-$repo_dir/artifacts/npu-runtime}"
mkdir -p "$out_dir"
"${CXX:-c++}" -std=c++17 -O2 -fPIC -shared \
  -I"${XRT_INCLUDE:-/usr/include}" "$repo_dir/native/npu/shim.cpp" \
  -L"${XRT_LIB:-/usr/lib}" -lxrt_coreutil \
  -Wl,-soname,libhrx_npu.so.1 -Wl,-rpath,'$ORIGIN' \
  -o "$out_dir/libhrx_npu.so.1"
