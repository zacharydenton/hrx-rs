#!/usr/bin/env bash
# GPU/compiler/CLI are unconditional; NPU is the only optional capability.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo check --locked --all-targets
cargo check --locked --all-targets --features npu
