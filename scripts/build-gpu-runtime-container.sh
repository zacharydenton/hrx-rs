#!/usr/bin/env bash
# Maintainer-only unified native build in the pinned Ubuntu 26.04 image.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends clang cmake ninja-build git python3 ca-certificates patch
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash /repo/scripts/build-amdf.sh /work/hrx-source /work/hrx-clang
clang --version > /work/hrx-clang/compiler-version.txt
dpkg-query -W > /work/hrx-clang/build-packages.txt
