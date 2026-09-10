#!/usr/bin/env bash
# Build the pinned HRX/Loom source with the reviewed compiler fixes.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="${1:-$repo_dir/artifacts/release-work}"
mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
source_dir="$work_dir/hrx-source"
build_dir="$work_dir/hrx-clang"
revision="$(cat "$repo_dir/patches/loom/base-revision")"
archive="$work_dir/sources/hrx-system-$revision.tar.gz"
if [[ -e "$source_dir" ]]; then
  echo "Use a fresh work directory; $source_dir already exists" >&2
  exit 1
fi
python3 - "$repo_dir" "$archive" <<'PY'
import hashlib, json, pathlib, sys
repo, archive = map(pathlib.Path, sys.argv[1:])
spec = json.loads((repo / 'native/release-inputs.json').read_text())['downloads']['hrx-system']
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == spec['sha256']
PY
mkdir -p "$source_dir"
tar -xzf "$archive" -C "$source_dir" --strip-components=1
for patch_file in "$repo_dir"/patches/loom/*.patch; do
  patch -d "$source_dir" -p1 --batch --forward < "$patch_file"
done
build_image="$(python3 - "$repo_dir" <<'PYCODE'
import json, pathlib, sys
print(json.loads((pathlib.Path(sys.argv[1]) / 'native/release-inputs.json').read_text())['build_image'])
PYCODE
)"
podman run --rm -v "$work_dir:/work" -v "$repo_dir:/repo:ro" \
  "$build_image" bash /repo/scripts/build-gpu-runtime-container.sh
