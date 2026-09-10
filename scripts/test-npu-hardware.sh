#!/usr/bin/env bash
# Build, provision, compile, and validate the NPU path on a configured host.
# Inputs are an existing pinned HRX build and an inventoried IRON toolchain.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
: "${HRX_NATIVE_BUILD:?set HRX_NATIVE_BUILD to the matching prepared native build}"
: "${HRX_NPU_TOOLCHAIN:?set HRX_NPU_TOOLCHAIN to a pinned toolchain.json}"
work_dir="${HRX_NPU_WORK:-$repo_dir/artifacts/npu-validation}"
mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
cd "$repo_dir"
base_dir="$(cargo run --quiet --features runner --bin hrx -- prepare)"
python3 scripts/build-interop-overlay.py "$HRX_NATIVE_BUILD" "$work_dir/runtime"
python3 - "$base_dir" "$work_dir/runtime" <<'PY'
from pathlib import Path
import sys
base, output = map(Path, sys.argv[1:])
for path in base.iterdir():
    target = output / path.name
    if '.so' in path.name and not path.name.startswith('libhrx') and not target.exists():
        target.symlink_to(path.resolve())
PY
bash scripts/build-npu-shim.sh "$work_dir/npu-runtime"
export HRX_RUNTIME_DIR="$work_dir/runtime"
export HRX_NPU_RUNTIME_DIR="$work_dir/npu-runtime"
HRX_TEST_NPU_DIR="$(cargo run --quiet --release --features npu-compile --example compile_npu -- "$HRX_NPU_TOOLCHAIN" 262144)"
export HRX_TEST_NPU_DIR
cargo run --quiet --features runner,npu --bin hrx -- doctor
cargo test --release --all-features -- --ignored --test-threads=1

if [[ "${HRX_QUALIFY_PERFORMANCE:-0}" == 1 ]]; then
  python3 scripts/qualify-npu-performance.py
fi
