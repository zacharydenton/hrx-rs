#!/usr/bin/env python3
"""Record an installed IRON/AIE toolchain; never install or mutate its files.
Run in the environment used to compile projects. Chess must already be installed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--python', type=Path, required=True)
parser.add_argument('--aiecc', type=Path, required=True)
parser.add_argument('--backend', choices=['Peano', 'Chess'], required=True)
parser.add_argument('--identity-root', type=Path, action='append', default=[])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
python = args.python.absolute()  # Keep the virtualenv entry point, not its symlink target.
aiecc = args.aiecc.absolute()
roots = [python.parent.parent, *[p.absolute() for p in args.identity_root]]
if args.backend == 'Chess':
    root = os.environ.get('AIETOOLS_ROOT')
    if not root:
        parser.error('source the installed Chess environment (AIETOOLS_ROOT is missing)')
    roots.append(Path(root))
paths = {python, aiecc}
for root in roots:
    for path in root.rglob('*'):
        if path.is_file() and not any(part in {'.git', '__pycache__', '.cache'} for part in path.parts) and path.suffix != '.pyc':
            paths.add(path)
files = {}
for path in sorted(paths):
    with path.open('rb') as stream:
        files[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
environment = {key: value for key, value in os.environ.items() if key in {
    'HOME', 'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'AIETOOLS_ROOT', 'CHESS_LICENSE_FILE',
    'XILINXD_LICENSE_FILE', 'LM_LICENSE_FILE', 'PEANO_INSTALL_DIR', 'MLIR_AIE_INSTALL_DIR',
    'AIE_INSTALL_DIR', 'AIE_OPT_DIR', 'LLVM_AIE_INSTALL_DIR', 'XILINX_VITIS',
}}
environment['PATH'] = str(python.parent) + os.pathsep + environment.get('PATH', '/usr/bin:/bin')
environment['PYTHONDONTWRITEBYTECODE'] = '1'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'python': str(python), 'aiecc': str(aiecc), 'backend': args.backend, 'files': files, 'environment': environment}, indent=2) + '\n')
print(f'{args.output}: pinned {len(files)} toolchain files')
