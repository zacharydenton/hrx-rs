#!/usr/bin/env python3
"""Stage the complete NPU runtime, provenance and source archives; never publish."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]
HOST = {'libc.so.6', 'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1',
        'libpthread.so.0', 'libdl.so.2', 'librt.so.1', 'ld-linux-x86-64.so.2'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack(archive, files):
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(mode='w', fileobj=zipped) as tar:
                for name, path in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mode = 0o755 if '.so' in name or name.endswith('.sh') else 0o644
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--work', type=Path, required=True, help='pinned NPU build work directory')
    parser.add_argument('--url', required=True, help='future public URL of the binary archive')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output / 'stage'
    stage.mkdir()
    inputs = json.loads((REPO / 'native/npu-release-inputs.json').read_text())
    source = args.work / 'xdna-driver'
    source_files = {}
    # Preserve the exact pristine source of every pinned repository, including
    # nested modules; each git archive excludes untracked/build files and .git.
    for number, (path, spec) in enumerate(inputs['sources'].items()):
        directory = source / path
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec['revision']:
            raise ValueError(f'Wrong source revision: {path}')
        archive = args.output / f'upstream-{number:02d}.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', '-C', str(directory), 'archive', spec['revision']], stdout=stream, check=True)
        source_files[f'upstream/{archive.name}'] = archive
        spec['source_archive'] = f'upstream/{archive.name}'
        spec['archive_sha256'] = digest(archive)
    source_pins = stage / 'source-inputs.json'
    source_pins.write_text(json.dumps(inputs, indent=2) + '\n')
    source_files['native/npu-release-inputs.json'] = source_pins
    for name in ('native/NPU-RELEASE.md', 'scripts/build-npu-runtime.sh',
                 'scripts/build-npu-runtime-container.sh', 'scripts/stage-npu-component.py',
                 'native/npu/shim.cpp', 'native/npu/shim.h', 'native/npu/NOTICE',
                 'native/licenses/LICENSE-HRX.txt', 'native/npu/LICENSE-LLVM.txt'):
        source_files[name] = REPO / name
    for patch in inputs['patches']:
        source_files[patch] = REPO / patch
    # Runtime-only packaging: no diagnostics, VTD blobs, Python or compiler tools.
    required = ['libhrx_npu.so.1', 'lib/libxrt_coreutil.so.2', 'lib/libxrt_core.so.2',
                'lib/libxrt_driver_xdna.so.2', 'lib/libuuid.so.1']
    files = {name: args.runtime / name for name in required}
    for path in args.runtime.glob('LICENSE-*.txt'):
        files[path.name] = path
    licenses = {
        'LICENSE-Apache-2.0.txt': REPO / 'native/licenses/LICENSE-HRX.txt',
        'LICENSE-LLVM.txt': REPO / 'native/npu/LICENSE-LLVM.txt',
        'LICENSE-XRT.txt': source / 'xrt/LICENSE',
        'LICENSE-AIEBU.txt': source / 'xrt/src/runtime_src/core/common/aiebu/LICENSE',
        'LICENSE-AIE-RT.txt': source / 'xrt/src/runtime_src/core/common/aiebu/src/cpp/aie-rt/license.txt',
        'LICENSE-ELFIO.txt': source / 'xrt/src/runtime_src/core/common/elf/LICENSE.txt',
        'LICENSE-GSL.txt': source / 'xrt/src/runtime_src/core/common/gsl/LICENSE',
        'LICENSE-cxxopts.txt': source / 'xrt/src/runtime_src/core/common/aiebu/src/cpp/cxxopts/LICENSE',
        'LICENSE-zstd.txt': source / 'xrt/src/runtime_src/core/common/aiebu/src/cpp/zstd/LICENSE',
    }
    files.update(licenses)
    # Keep per-file copyright/SPDX declarations for runtime sources and embedded
    # headers; the complete source archive preserves notices elsewhere in files.
    headers = []
    for directory in (source / 'src/shim', source / 'xrt/src/runtime_src/core/common',
                      source / 'xrt/src/runtime_src/core/pcie/linux'):
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ('.h', '.cpp', '.c', '.hpp'):
                continue
            lines = path.read_text(errors='replace').splitlines()
            comment = []
            in_block = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('/*'):
                    in_block = True
                if in_block or stripped.startswith('//') or not stripped:
                    comment.append(line)
                    if '*/' in stripped:
                        in_block = False
                else:
                    break
            if comment:
                headers.append(f'Source: {path.relative_to(source)}\n' + '\n'.join(comment))
    attribution = stage / 'LICENSE-source-attributions.txt'
    attribution.write_text('\n\n'.join(headers) + '\n')
    files[attribution.name] = attribution
    # nlohmann's amalgamated headers declare MIT but omit its terms.
    mit = (REPO / 'LICENSE').read_text()
    mit = mit[mit.index('Permission is hereby granted'):]
    json_license = stage / 'LICENSE-nlohmann-json.txt'
    json_license.write_text('Copyright (c) 2013-2025 Niels Lohmann\n\n' + mit)
    files[json_license.name] = json_license
    source_files.update({f'licenses/{name}': path for name, path in files.items() if name.startswith('LICENSE-')})
    source_files['build-packages.txt'] = args.work / 'build-packages.txt'
    sources = args.output / 'hrx-npu-sources.tar.gz'
    pack(sources, source_files)
    source_record = {'url': args.url.rsplit('/', 1)[0] + '/' + sources.name, 'sha256': digest(sources)}
    components = {}
    for name in required:
        elf = subprocess.check_output(['readelf', '-d', str(files[name])], text=True)
        needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', elf)
        missing = set(needed) - {Path(n).name for n in required} - HOST
        if missing:
            raise ValueError(f'{name} has unbundled dependencies: {missing}')
        versions = subprocess.check_output(['readelf', '-V', str(files[name])], text=True)
        components[name] = {'sha256': digest(files[name]), 'needed': needed,
                            'host_symbol_versions': sorted(set(re.findall(r'\b(?:GLIBC|GLIBCXX|CXXABI)_[\d.]+', versions)))}
    inventory = {'schema': 1, 'component': 'npu-runtime', 'sources': inputs,
                 'components': components, 'corresponding_source': source_record,
                 'system_dependencies_not_shipped': sorted(HOST),
                 'build_packages_sha256': digest(args.work / 'build-packages.txt'),
                 'license_files': {name: digest(path) for name, path in files.items() if name.startswith('LICENSE-')},
                 'changes': 'Runtime-only XDNA packaging; use pinned AIEBU ISA headers; set DSO RUNPATH to $ORIGIN (shim: $ORIGIN/lib).',
                 'scope': 'Fresh Ubuntu 24.04 build with pinned upstream sources; package versions recorded, not a bit-reproducibility claim.'}
    inventory_path = stage / 'THIRD-PARTY.json'
    inventory_path.write_text(json.dumps(inventory, indent=2) + '\n')
    shutil.copyfile(inventory_path, REPO / 'native/NPU-THIRD-PARTY.json')
    notice = stage / 'NOTICE'
    notice.write_text((REPO / 'native/npu/NOTICE').read_text() + '\n'
        'Bundled XRT and the open-source XDNA plugin: Apache-2.0.\n'
        'Embedded AIEBU, AIE-RT, ELFIO, GSL, cxxopts and nlohmann JSON: MIT.\n'
        'Embedded Zstandard/xxHash: BSD licenses; Boost headers: BSL-1.0.\n'
        'Ubuntu libuuid: BSD-3-Clause; see LICENSE-libuuid.txt for its full notices.\n'
        'Full license terms and copyright attributions accompany this archive.\n'
        'VTD diagnostic blobs, firmware, kernel modules and compiler tools are not included.\n'
        f'Corresponding source: {source_record["url"]}\nSHA-256: {source_record["sha256"]}\n')
    files.update({'THIRD-PARTY.json': inventory_path, 'NOTICE': notice,
                  'source-inputs.json': source_pins, 'build-packages.txt': args.work / 'build-packages.txt'})
    archive = args.output / 'hrx-npu-linux-x86_64.tar.gz'
    pack(archive, files)
    manifest = {'schema': 1, 'component': 'npu-runtime',
                'revision': 'HRX NPU ABI 1; XRT c826a0efc56d; XDNA 8dfda66f67a8; Ubuntu 24.04',
                'url': args.url, 'archive_sha256': digest(archive),
                'files': {name: digest(path) for name, path in sorted(files.items())}}
    (args.output / 'npu-bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output / 'npu-bundle.json')


if __name__ == '__main__':
    main()
