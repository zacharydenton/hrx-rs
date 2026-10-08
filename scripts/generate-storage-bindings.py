#!/usr/bin/env python3
"""Regenerate io_uring layouts from verified release inputs (requires bindgen)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, help='directory populated by fetch-native-inputs.py')
    args = parser.parse_args()
    inputs = json.loads((ROOT / 'native/release-inputs.json').read_text())['downloads']
    with tempfile.TemporaryDirectory(prefix='hrx-storage-bindings-') as directory:
        include = Path(directory)
        for name, relative in [('linux_io_uring_header', 'linux/io_uring.h'),
                               ('linux_io_uring_zcrx_header', 'linux/io_uring/zcrx.h')]:
            spec = inputs[name]
            source = args.sources / spec['archive']
            if hashlib.sha256(source.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError(f'Corrupt pinned input: {source}')
            destination = include / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        subprocess.run([
            'bindgen', str(include / 'linux/io_uring.h'),
            '--allowlist-type', 'io_uring_params|io_uring_restriction',
            '--allowlist-var', 'IORING_SETUP_(NO_MMAP|NO_SQARRAY|R_DISABLED|SQPOLL)|IORING_ENTER_SQ_WAKEUP|IORING_SQ_NEED_WAKEUP',
            '--with-derive-default', '--raw-line',
            '// Generated from pinned Linux UAPI; see native/release-inputs.json.',
            '--output', str(ROOT / 'src/fabric/storage_ffi.rs'), '--', f'-I{include}',
        ], check=True)

if __name__ == '__main__':
    main()
