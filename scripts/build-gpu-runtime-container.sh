#!/usr/bin/env bash
# Maintainer-only Ubuntu 26.04 build. Source is already pinned and patched.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends clang cmake ninja-build git python3 ca-certificates
cmake -S /work/hrx-source -B /work/hrx-clang -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DLOOM_TARGET_AMDGPU=ON \
  -DLOOM_TARGET_AMDGPU_TARGETS=gfx1151 -DIREE_ENABLE_LIBBACKTRACE=OFF \
  -DIREE_HAL_DRIVER_AMDGPU=ON
cmake --build /work/hrx-clang --target loomc_shared libhrx_src_libhrx_hrx -j 16
clang --version > /work/hrx-clang/compiler-version.txt
dpkg-query -W > /work/hrx-clang/build-packages.txt
