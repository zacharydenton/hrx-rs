#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential cmake ninja-build git pkg-config libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-system-dev uuid-dev libdrm-dev ocl-icd-opencl-dev libncurses-dev libssl-dev libelf-dev rapidjson-dev python3 python3-yaml systemtap-sdt-dev patchelf
cmake -S /work/xdna-driver -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/hrx-xrt -DCMAKE_INSTALL_LIBDIR=lib -DSKIP_KMOD=ON -DBUILD_VXDNA=OFF -DXRT_ENABLE_WERROR=OFF
cmake --build /work/build --target xrt_coreutil xrt_core xrt_driver_xdna -j 12
mkdir -p /work/runtime/lib
for lib in xrt_coreutil xrt_core xrt_driver_xdna; do
  file=$(find /work/build -name "lib${lib}.so.2" -type l | head -1)
  cp -L "$file" "/work/runtime/lib/lib${lib}.so.2"
  patchelf --set-rpath '$ORIGIN' "/work/runtime/lib/lib${lib}.so.2"
done
cp -L /usr/lib/x86_64-linux-gnu/libuuid.so.1 /work/runtime/lib/
boost_headers_package="$(dpkg-query -S /usr/include/boost/version.hpp | cut -d: -f1)"
cp "/usr/share/doc/$boost_headers_package/copyright" /work/runtime/LICENSE-Boost.txt
cp /usr/share/doc/libuuid1/copyright /work/runtime/LICENSE-libuuid.txt
cp /usr/share/common-licenses/GPL-2 /work/runtime/LICENSE-GPL-2.0.txt
cp /usr/share/common-licenses/GPL-3 /work/runtime/LICENSE-GPL-3.0.txt
cp /usr/share/common-licenses/LGPL-2.1 /work/runtime/LICENSE-LGPL-2.1.txt
g++ -std=c++17 -O2 -fPIC -shared -I/work/xdna-driver/xrt/src/runtime_src/core/include -I/work/build/xrt/src/gen /repo/native/npu/shim.cpp -L/work/runtime/lib -l:libxrt_coreutil.so.2 -Wl,-soname,libhrx_npu.so.1 -Wl,-rpath,'$ORIGIN/lib' -o /work/runtime/libhrx_npu.so.1
dpkg-query -W > /work/build-packages.txt
