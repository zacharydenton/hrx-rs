#!/usr/bin/env bash
# Qualify the selected unified native bundle on gfx1151 + Strix Halo NPU5.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --locked --features npu --bin hrx -- prepare
cargo run --locked --features npu --bin hrx -- doctor
cargo test --locked --all-features -- --ignored --test-threads=1
cargo run --locked --release --features npu --example gemm_pipeline
