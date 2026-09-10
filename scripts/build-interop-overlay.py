#!/usr/bin/env python3
"""Rebuild the interop object and relink a prepared HRX build without modifying it.
For local validation only; releases apply patch 0008 and rebuild all pinned sources.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('build', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
build = args.build.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
entries = json.loads((build / 'compile_commands.json').read_text())
entry = next(e for e in entries if e['file'].endswith('/libhrx/buffer.c') and 'hrx.objects' in e['command'])
source = out / 'buffer.c'
source.write_text(Path(entry['file']).read_text() + '\n' + (repo / 'native/interop.c').read_text())
command = shlex.split(entry['command'])
old_object = command[command.index('-o') + 1]
command[command.index('-o') + 1] = str(out / 'buffer.o')
command[command.index(entry['file'])] = str(source)
subprocess.run(command, cwd=entry['directory'], check=True)
commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'libhrx_src_libhrx_hrx'], text=True)
link = next(line for line in commands.splitlines() if '-o libhrx/src/libhrx/libhrx.so' in line)
command = shlex.split(link.removeprefix(': && ').removesuffix(' && :'))
command[command.index('-o') + 1] = str(out / 'libhrx.so')
command = [str(out / 'buffer.o') if arg == old_object else arg for arg in command]
command = [arg.replace('--dependency-file=libhrx/src/libhrx/CMakeFiles/libhrx_src_libhrx_hrx.dir/link.d', '--dependency-file=' + str(out / 'link.d')) for arg in command]
subprocess.run(command, cwd=build, check=True)
