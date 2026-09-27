#!/usr/bin/env bash
# Development-only; consumers need neither protoc nor protobuf-codegen.
# Regenerates src/artifacts/onnx_proto.rs from src/artifacts/onnx.proto.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --quiet --manifest-path scripts/onnx-codegen/Cargo.toml
