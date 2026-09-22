#!/usr/bin/env bash
# Build the 0.8 native boundary from a pinned, patched hrx-system source tree.
# Use this inside the pinned Ubuntu image for distributable release artifacts.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="$(realpath "${1:?usage: build-amdf.sh SOURCE BUILD}")"
build_dir="$(realpath -m "${2:?usage: build-amdf.sh SOURCE BUILD}")"
for native_patch in "$repo_dir/native/amdf/build.patch" "$repo_dir/native/amdf/queue-ring.patch" "$repo_dir/native/amdf/cache-policy.patch"; do
  if patch -d "$source_dir" -p1 --dry-run --batch --forward < "$native_patch" >/dev/null 2>&1; then
    patch -d "$source_dir" -p1 --batch --forward < "$native_patch"
  elif ! patch -d "$source_dir" -p1 --dry-run --batch --reverse < "$native_patch" >/dev/null 2>&1; then
    echo "Source does not match the reviewed patch: $native_patch" >&2
    exit 1
  fi
done

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PROJECT_INCLUDE="$repo_dir/native/amdf/bridge.cmake" \
  -DLIBHRX_BUILD=OFF -DAMDF_BUILD=ON -DAMDF_FAMILY_CDNA=OFF \
  -DAMDF_FAMILY_RDNA=ON -DAMDF_FAMILY_XDNA=ON \
  -DIREE_HAL_DRIVER_AMDGPU=OFF -DIREE_HAL_DRIVER_TASK=OFF \
  -DIREE_ENABLE_LIBBACKTRACE=OFF -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF -DHRX_INSTALL_TESTS=OFF \
  -DLOOM_TARGET_DEFAULTS=OFF -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_X86=OFF -DLOOM_TARGET_SPIRV=OFF \
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151 -DLOOM_TARGET_XDNA=ON \
  -DLOOM_EMIT_XDNA=ON -DLOOM_IMPORT_CXX=ON \
  -DLOOM_EXECUTE_DEFAULTS=OFF -DLOOM_EXECUTE_IREE_HAL=OFF
cmake --build "$build_dir" --target amdf loomc_shared hrx_fabric -j "${HRX_BUILD_JOBS:-8}"
