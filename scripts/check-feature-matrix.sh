#!/usr/bin/env bash
# Check every subset, including runner without an explicit loom feature.
set -euo pipefail
cd "$(dirname "$0")/.."
features=(download loom runner)
for ((mask=0; mask<(1 << ${#features[@]}); mask++)); do
  selected=()
  for ((bit=0; bit<${#features[@]}; bit++)); do
    if ((mask & (1 << bit))); then
      selected+=("${features[bit]}")
    fi
  done
  if ((${#selected[@]})); then
    feature_list=$(IFS=,; echo "${selected[*]}")
    cargo check --locked --all-targets --no-default-features --features "$feature_list"
  else
    cargo check --locked --all-targets --no-default-features
  fi
done
