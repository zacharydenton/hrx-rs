#!/usr/bin/env python3
"""Stage a hashed NPU runtime component and manifest; does not publish."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runtime', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--url', required=True)
parser.add_argument('--revision', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=True)
files = {'libhrx_npu.so.1': args.runtime / 'libhrx_npu.so.1', 'NOTICE': repo / 'native/npu/NOTICE', 'LICENSE-Apache-2.0.txt': repo / 'native/licenses/LICENSE-HRX.txt', 'LICENSE-LLVM.txt': repo / 'native/npu/LICENSE-LLVM.txt'}
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
archive = args.output / 'hrx-npu-linux-x86_64.tar.gz'
with archive.open('wb') as raw:
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(mode='w', fileobj=zipped) as tar:
            for name, path in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o755 if '.so' in name else 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
manifest = {'schema': 1, 'component': 'npu-runtime', 'revision': args.revision, 'url': args.url, 'archive_sha256': digest(archive), 'files': {name: digest(path) for name, path in files.items()}}
(args.output / 'npu-bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(args.output / 'npu-bundle.json')
