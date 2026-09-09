#!/usr/bin/env python3
"""Fetch and unpack the reviewed inputs; does not build or publish anything."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=REPO / 'artifacts/release-work')
    args = parser.parse_args()
    work = args.work.resolve()
    (work / 'sources').mkdir(parents=True, exist_ok=True)
    inputs = json.loads((REPO / 'native/release-inputs.json').read_text())

    def fetch(item):
        name, spec = item
        parent = work if name.endswith(('.gz', '.zst', '.json')) else work / 'sources'
        path = parent / spec['archive']
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with urllib.request.urlopen(spec['url'], timeout=120) as response:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    output.flush()
                    with temporary.open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                            raise ValueError(f'Digest mismatch: {name}')
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError(f'Digest mismatch: {name}')
        return name, path

    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = dict(pool.map(fetch, inputs['downloads'].items()))
    for name in ['elfutils', 'numactl', 'libdrm', 'bzip2', 'zlib', 'zstd', 'liblzma', 'fmt', 'glog']:
        dest = work / 'sources' / name
        if not dest.exists():
            dest.mkdir()
            with tarfile.open(downloaded[name]) as archive:
                archive.extractall(dest, filter='data')
    for name, dest_name in [('therock.tar.gz', 'therock'), ('rocm-systems.tar.gz', 'rocm-systems')]:
        dest = work / dest_name
        if not dest.exists():
            dest.mkdir()
            # Verified GitHub source archives; omit the archive's enclosing directory.
            command = ['tar', '-xzf', str(downloaded[name]), '-C', str(dest), '--strip-components=1']
            if dest_name == 'rocm-systems':
                command += ['--wildcards', '*/projects/rocr-runtime/*', '*/projects/rocprofiler-register/*']
            subprocess.run(command, check=True)
    dest = work / 'upstream-deps'
    if not dest.exists():
        dest.mkdir()
        subprocess.run(['tar', '--zstd', '-xf', str(downloaded['upstream-deps.tar.zst']), '-C', str(dest)], check=True)
    print(f'Verified {len(downloaded)} inputs in {work}')


if __name__ == '__main__':
    main()
