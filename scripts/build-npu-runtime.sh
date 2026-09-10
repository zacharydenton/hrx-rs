#!/usr/bin/env bash
# Maintainer build only. Cargo never invokes this or a native compiler.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="${1:-$repo_dir/artifacts/npu-release-work}"
mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
python3 - "$repo_dir" "$work_dir" <<'PY'
import hashlib, json, pathlib, subprocess, sys
repo, work = map(pathlib.Path, sys.argv[1:])
inputs = json.loads((repo / 'native/npu-release-inputs.json').read_text())
root = work / 'xdna-driver'
if root.exists():
    raise SystemExit('Use a fresh work directory for a pinned NPU build')
for path, source in inputs['sources'].items():
    dest = root / path
    subprocess.run(['git', 'clone', '--no-checkout', source['repository'], str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', source['revision']], check=True)
subprocess.run(['git', '-C', str(root), 'submodule', 'init', 'xrt'], check=True)
subprocess.run(['git', '-C', str(root), 'submodule', 'absorbgitdirs', 'xrt'], check=True)
for patch, path in inputs['patches'].items():
    with (repo / patch).open('rb') as stream:
        subprocess.run(['patch', '-d', str(root / path), '-p1', '--batch', '--forward'], stdin=stream, check=True)
PY
build_image="$(python3 - "$repo_dir" <<'PY'
import json, pathlib, sys
print(json.loads((pathlib.Path(sys.argv[1]) / 'native/npu-release-inputs.json').read_text())['build_image'])
PY
)"
podman run --rm -v "$work_dir:/work" -v "$repo_dir:/repo:ro" \
  "$build_image" bash /repo/scripts/build-npu-runtime-container.sh
