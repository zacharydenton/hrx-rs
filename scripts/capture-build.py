#!/usr/bin/env python3
"""Build and bind an executable to its source tree and pinned native bundle.

Run from the application's checkout. Ignored build artifacts are not source;
tracked files and non-ignored untracked files are hashed before and after build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_tree(root):
    paths = subprocess.check_output(['git', 'ls-files', '-z', '-c', '-o', '--exclude-standard'], cwd=root).split(b'\0')
    files = {}
    for raw in sorted(set(paths) - {b''}):
        relative = os.fsdecode(raw)
        path = root / relative
        if path.is_symlink():
            files[relative] = {'symlink': os.readlink(path)}
        elif path.is_file():
            files[relative] = {'sha256': digest(path), 'executable': bool(path.stat().st_mode & 0o111)}
        elif not path.exists():
            files[relative] = {'deleted': True}
        else:
            raise ValueError(f'unsupported source entry: {relative}')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command: p.error('supply the build command after --')
    if a.output.exists(): p.error('output exists; build evidence is immutable')
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    source_hash, files = source_tree(root)
    native = a.native.resolve()
    libraries = {p.name: digest(p) for p in sorted(native.glob('*.so'))}
    if not {'libamdf.so', 'libhrx_fabric.so', 'libloomc.so'} <= libraries.keys():
        p.error('native directory must contain all three pinned libraries')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    start = time.monotonic()
    result = subprocess.run(command, cwd=root)
    if result.returncode: raise SystemExit(result.returncode)
    if source_tree(root) != (source_hash, files):
        raise SystemExit('source changed during build; rerun after dependencies/generation settle')
    if {p.name: digest(p) for p in sorted(native.glob('*.so'))} != libraries:
        raise SystemExit('native bundle changed during build')
    if not a.binary.is_file() or not os.access(a.binary, os.X_OK):
        raise SystemExit('build did not leave the requested executable')
    record = dict(source_revision=revision, source_tree_sha256=source_hash,
                  source_files=files, binary=str(a.binary.resolve()), binary_sha256=digest(a.binary),
                  native_directory=str(native), native_hashes=libraries,
                  compiler_sha256=libraries['libloomc.so'], build_command=command,
                  build_wall_seconds=time.monotonic()-start,
                  rustc_identity=subprocess.check_output(['rustc', '-vV'], text=True).strip(),
                  build_environment={k: v for k, v in os.environ.items()
                                     if k.startswith('CARGO_PROFILE_') or k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTUP_TOOLCHAIN', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS')})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as destination:
        json.dump(record, destination, indent=2)
        destination.write('\n')


if __name__ == '__main__':
    main()
