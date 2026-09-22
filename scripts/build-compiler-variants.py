#!/usr/bin/env python3
"""Build isolated optional-patch variants, restoring the input tree afterward.

SOURCE must already contain patches 0002, 0004 and 0007. BUILD is a configured
native-only compiler build. Do not run another build against SOURCE concurrently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('build', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source, build, output = (p.resolve() for p in (args.source, args.build, args.output))
    output.mkdir(parents=True, exist_ok=True)
    patches = {key: next((repo / 'patches').rglob(f'{key}-*.patch'))
               for key in ['0002', '0004', '0007']}
    applied = set(patches)
    records = {}

    def patch(key, enable):
        command = ['patch', '-p1', '--batch', '--forward' if enable else '--reverse',
                   '--input', str(patches[key])]
        subprocess.run(command + ['--dry-run'], cwd=source, check=True)
        subprocess.run(command, cwd=source, check=True)
        (applied.add if enable else applied.remove)(key)

    def snapshot(name):
        subprocess.run(['cmake', '--build', str(build), '--target', 'loomc_shared',
                        '-j', str(args.jobs)], check=True)
        destination = output / f'libloomc-{name}.so'
        shutil.copyfile(build / 'loom/binding/c/libloomc.so', destination)
        with destination.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        records[name] = {'library': str(destination), 'sha256': sha,
                         'optional_patches': sorted(applied)}
        (output / 'variants.json').write_text(json.dumps(records, indent=2) + '\n')

    try:
        snapshot('all')
        for key in reversed(patches):
            patch(key, False)
        snapshot('base')
        for key in patches:
            patch(key, True)
            snapshot(key)
            patch(key, False)
    finally:
        for key in patches:
            if key not in applied:
                patch(key, True)
        # Restore the library as well as the sources if every snapshot succeeded.
        if len(records) == 5:
            subprocess.run(['cmake', '--build', str(build), '--target', 'loomc_shared',
                            '-j', str(args.jobs)], check=True)


if __name__ == '__main__':
    main()
