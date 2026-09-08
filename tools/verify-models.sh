#!/usr/bin/env bash
# Standalone reviewable validation, with sibling model repositories by default.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
krea=${KREA2_REPO:-$root/../krea2-loom}
h3=${H3_REPO:-$root/../minimax-h3-loom}
cargo test --manifest-path "$root/Cargo.toml" --all-features
cargo test --manifest-path "$root/Cargo.toml" --all-features --test gpu -- --ignored --test-threads=1
cargo build --manifest-path "$krea/Cargo.toml" --release -p krea2-abi
cargo build --manifest-path "$h3/Cargo.toml" --release -p h3
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cc -Wall -Wextra -Werror -O2 -I"$h3/include" -I"$krea/build/include" \
  "$root/tests/c/models.c" -ldl -pthread -o "$scratch/models"
"$scratch/models" "$h3/target/release/libh3.so" "$krea/target/release/libkrea2.so"
