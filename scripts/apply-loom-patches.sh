#!/usr/bin/env bash
# Development-only: apply HRX's compiler fixes to the pinned public source.
set -euo pipefail
patch_dir="$(cd "$(dirname "$0")/../patches/loom" && pwd)"
: "${LOOM_SOURCE:?Set LOOM_SOURCE to a clean hrx-system checkout}"
expected_revision="$(cat "$patch_dir/base-revision")"
actual_revision="$(git -C "$LOOM_SOURCE" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "Expected hrx-system $expected_revision; found $actual_revision" >&2
  exit 1
fi
if [[ -n "$(git -C "$LOOM_SOURCE" status --porcelain --untracked-files=no)" ]]; then
  echo "hrx-system has tracked changes; use a clean checkout before applying patches" >&2
  exit 1
fi
git -C "$LOOM_SOURCE" apply --check "$patch_dir"/*.patch
git -C "$LOOM_SOURCE" apply "$patch_dir"/*.patch
