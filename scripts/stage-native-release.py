#!/usr/bin/env python3
"""Stage the reviewed native distribution from pinned sources and a fresh HRX build.

Development/release tool only. Requires Python 3.12+, readelf, tar with zstd support,
and the extracted source trees described in native/RELEASE.md. Does not publish.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def first_comment(path):
    text = path.read_text()
    if text.startswith('//'):
        return '\n'.join(line for line in text.split('\n\n', 1)[0].splitlines()) + '\n'
    match = re.search(r'/\*.*?\*/', text, re.S)
    if not match:
        raise ValueError(f'No license comment in {path}')
    return match.group() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=REPO / 'artifacts/release-work')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--release-tag', required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    build = (args.build or work / 'hrx-clang').resolve()
    inputs = json.loads((REPO / 'native/release-inputs.json').read_text())
    downloaded = {}
    for name, spec in inputs['downloads'].items():
        candidates = [work / spec['archive'], work / 'sources' / spec['archive']]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None or digest(path) != spec['sha256']:
            raise ValueError(f'Missing or corrupt pinned input: {name}')
        downloaded[name] = path
    for name, sha in inputs['hrx']['patches'].items():
        if digest(REPO / 'patches/loom' / name) != sha:
            raise ValueError(f'Patch changed: {name}')

    stage = work / 'stage'
    stage.mkdir(exist_ok=True)
    if any(stage.iterdir()):
        raise ValueError(f'Stage must be empty: {stage}')
    license_dir = REPO / 'native/licenses'
    license_dir.mkdir(exist_ok=True)
    license_evidence = {}

    def license_file(name, path=None, text=None, evidence=None):
        dest = license_dir / name
        if path is not None:
            shutil.copyfile(path, dest)
            evidence = str(path.relative_to(work))
        else:
            dest.write_text(text)
        shutil.copyfile(dest, stage / name)
        license_evidence[name] = {'source_file': evidence, 'sha256': digest(dest)}
        return name

    hrx = work / 'hrx-source'
    deps = build / '_deps'
    rocr = work / 'rocm-systems/projects/rocr-runtime'
    register = work / 'rocm-systems/projects/rocprofiler-register'
    roots = {name: next(p for p in (work / 'sources' / name).iterdir() if p.is_dir())
             for name in ['elfutils', 'numactl', 'libdrm', 'bzip2', 'zlib', 'zstd', 'liblzma', 'fmt', 'glog']}
    for name, path in {
        'LICENSE-HRX.txt': hrx / 'LICENSE',
        'LICENSE-HSA-headers.txt': deps / 'hsa_runtime_headers-src/LICENSE.txt',
        'LICENSE-SPIRV-Headers.txt': deps / 'spirv_headers-src/LICENSE',
        'LICENSE-Vulkan-Headers.txt': deps / 'vulkan_headers-src/LICENSES/MIT.txt',
        'LICENSE-ROCR.txt': rocr / 'LICENSE.txt',
        'LICENSE-ROCT.txt': rocr / 'libhsakmt/LICENSE.md',
        'LICENSE-rocprofiler-register.txt': register / 'LICENSE.md',
        'LICENSE-fmt.txt': roots['fmt'] / 'LICENSE',
        'LICENSE-glog.txt': roots['glog'] / 'COPYING',
        'LICENSE-elfutils-LGPL-3.0.txt': roots['elfutils'] / 'COPYING-LGPLV3',
        'LICENSE-elfutils-GPL-3.0.txt': roots['elfutils'] / 'COPYING',
        'LICENSE-numactl-LGPL-2.1.txt': roots['numactl'] / 'LICENSE.LGPL2.1',
        'LICENSE-bzip2.txt': roots['bzip2'] / 'LICENSE',
        'LICENSE-zstd.txt': roots['zstd'] / 'LICENSE',
        'LICENSE-xz-0BSD.txt': roots['liblzma'] / 'COPYING.0BSD',
        'LICENSE-TheRock.txt': work / 'therock/LICENSE',
    }.items():
        license_file(name, path=path)
    for name, path in {
        'LICENSE-CORE-MATH.txt': hrx / 'runtime/src/iree/math/trigonometry.c',
        'LICENSE-ROCT-rbtree.txt': rocr / 'libhsakmt/src/rbtree.c',
        'LICENSE-zlib.txt': roots['zlib'] / 'zlib.h',
    }.items():
        license_file(name, text=first_comment(path), evidence=str(path.relative_to(work)))
    # The AMD XML explicitly declares MIT and its copyright holder. Supply
    # the MIT terms with that attribution (the ZIP itself has no license file).
    xml_copyright = 'Copyright (c) 2026 Advanced Micro Devices, Inc., or its affiliates.'
    mit_text = (deps / 'vulkan_headers-src/LICENSES/MIT.txt').read_text()
    mit_text = re.sub(r'Copyright[^\n]+', xml_copyright, mit_text, count=1)
    license_file('LICENSE-AMDGPU-ISA.txt', text=mit_text,
                 evidence='AMD_GPU_MR_ISA_XML_2026_03_05.zip: Document/Copyright and Document/License (MIT); standard MIT terms')
    # libdrm distributes its license and attribution in source headers, not a
    # project-wide COPYING file. Preserve the complete notices for the built
    # core/AMDGPU sources and their public/internal headers.
    drm = roots['libdrm']
    drm_paths = sorted(set(list(drm.glob('*.[ch]')) + list((drm / 'amdgpu').glob('*.[ch]')) +
        [drm / 'include/drm' / name for name in ['drm.h', 'drm_mode.h', 'drm_fourcc.h', 'amdgpu_drm.h']]))
    license_file('LICENSE-libdrm.txt', text='\n'.join(
        f'Source: {p.relative_to(drm)}\n{first_comment(p)}' for p in drm_paths),
        evidence='sources/libdrm/libdrm-libdrm-2.4.127: core/AMDGPU source and header notices')
    # Preserve per-file attributions in addition to the projects' license texts.
    notice_headers = []
    for name, paths in {
        'elfutils': list((roots['elfutils'] / 'libelf').glob('*.[ch]')) + list((roots['elfutils'] / 'lib').glob('*.[ch]')),
        'numactl': list(roots['numactl'].glob('*.[ch]')),
        'ROCR': list((rocr / 'runtime/hsa-runtime').rglob('*.cpp')) + list((rocr / 'runtime/hsa-runtime').rglob('*.h')),
        'ROCT': list((rocr / 'libhsakmt/src').glob('*.[ch]')),
    }.items():
        for path in sorted(paths):
            text = path.read_text(errors='replace')
            if 'Copyright' in text[:4000] or 'copyright' in text[:4000]:
                notice_headers.append(f'Source: {path.relative_to(work)}\n{first_comment(path)}')
    license_file('LICENSE-source-attributions.txt', text='\n'.join(notice_headers),
                 evidence='Pinned elfutils, numactl, ROCR and ROCT source headers')

    amd = work / 'upstream-deps/lib'
    runtime_names = {
        'libhsa-runtime64.so.1': ('ROCR-Runtime', '1.21.0', 'NCSA AND MIT AND BSD-2-Clause',
            ['LICENSE-ROCR.txt', 'LICENSE-ROCT.txt', 'LICENSE-ROCT-rbtree.txt', 'LICENSE-libdrm.txt'], ['rocm-systems.tar.gz']),
        'librocprofiler-register.so.0': ('rocprofiler-register', '0.6.0', 'MIT AND BSD-3-Clause',
            ['LICENSE-rocprofiler-register.txt', 'LICENSE-fmt.txt', 'LICENSE-glog.txt'], ['rocm-systems.tar.gz', 'fmt', 'glog']),
        'librocm_sysdeps_elf.so.1': ('elfutils/libelf', '0.192', 'LGPL-3.0-or-later',
            ['LICENSE-elfutils-LGPL-3.0.txt', 'LICENSE-elfutils-GPL-3.0.txt'], ['elfutils']),
        'librocm_sysdeps_numa.so.1': ('numactl/libnuma', '2.0.19', 'LGPL-2.1-only',
            ['LICENSE-numactl-LGPL-2.1.txt'], ['numactl']),
        'librocm_sysdeps_drm.so.2': ('libdrm', '2.4.127', 'MIT', ['LICENSE-libdrm.txt'], ['libdrm']),
        'librocm_sysdeps_drm_amdgpu.so.1': ('libdrm/AMDGPU', '2.4.127', 'MIT', ['LICENSE-libdrm.txt'], ['libdrm']),
        'librocm_sysdeps_bz2.so': ('bzip2', '1.0.8', 'bzip2-1.0.6', ['LICENSE-bzip2.txt'], ['bzip2']),
        'librocm_sysdeps_z.so.1': ('zlib', '1.3.2', 'Zlib', ['LICENSE-zlib.txt'], ['zlib']),
        'librocm_sysdeps_zstd.so.1': ('Zstandard', '1.5.7', 'BSD-3-Clause', ['LICENSE-zstd.txt'], ['zstd']),
        'librocm_sysdeps_liblzma.so.5': ('XZ Utils/liblzma', '5.8.1', '0BSD', ['LICENSE-xz-0BSD.txt'], ['liblzma']),
    }
    components = {}

    def component(name, source, project, version, spdx, texts, source_keys, origin):
        shutil.copyfile(source, stage / name)
        elf = subprocess.check_output(['readelf', '-d', str(stage / name)], text=True)
        components[name] = {
            'project': project, 'version': version, 'origin': origin,
            'sha256': digest(stage / name),
            'needed': re.findall(r'\(NEEDED\).*?\[(.*?)\]', elf),
            'sources': {key: inputs['downloads'][key] for key in source_keys},
            'license': {'spdx': spdx, 'status': 'confirmed', 'files': texts,
                        'basis': 'Exact upstream source texts in license_files; see native/RELEASE.md for scope and modifications.'},
        }

    for name, (project, version, spdx, texts, source_keys) in runtime_names.items():
        source = amd / name if (amd / name).exists() else amd / 'rocm_sysdeps/lib' / name
        component(name, source, project, version, spdx, texts + ['LICENSE-source-attributions.txt'],
                  source_keys + ['therock.tar.gz'], 'Unmodified bytes from ROCm/hrx-system v0.3.0 public-deps; TheRock run 26672984641')
    for name, relative in {'libhrx.so': 'libhrx/src/libhrx/libhrx.so',
                           'libhrx.so.0': 'libhrx/src/libhrx/libhrx.so',
                           'libloomc.so': 'loom/binding/c/libloomc.so'}.items():
        component(name, build / relative, 'HRX / Loom', inputs['hrx']['revision'],
                  'Apache-2.0 WITH LLVM-exception AND MIT AND NCSA',
                  ['LICENSE-HRX.txt', 'LICENSE-CORE-MATH.txt', 'LICENSE-HSA-headers.txt',
                   'LICENSE-SPIRV-Headers.txt', 'LICENSE-Vulkan-Headers.txt', 'LICENSE-AMDGPU-ISA.txt'],
                  ['hrx-system', 'hsa_runtime_headers', 'spirv_headers', 'vulkan_headers', 'amdgpu_isa_xml'], 'Fresh Release build of the pinned source plus the eight patches in patches/loom')
    components['librocprofiler-register.so.0']['statically_linked'] = {
        'fmt': {'version': '11.1.4', 'revision': inputs['downloads']['fmt']['revision'], 'license': 'MIT'},
        'glog': {'version': '0.7.1', 'revision': inputs['downloads']['glog']['revision'], 'license': 'BSD-3-Clause'},
    }
    components['libhsa-runtime64.so.1']['statically_linked'] = {
        'ROCT-Thunk': {'revision': 'cb6561243e0a80215f5566a0feeb19eb44702aa4', 'license': 'MIT AND BSD-2-Clause',
                       'note': 'Includes the Nginx-derived rbtree; its BSD-2-Clause notice is preserved.'},
    }
    for name in ['libhrx.so', 'libhrx.so.0', 'libloomc.so']:
        components[name]['statically_linked'] = {
            'IREE': {'revision': inputs['hrx']['revision'], 'license': 'Apache-2.0 WITH LLVM-exception'},
            'CORE-MATH adaptation': {'revision': 'e0a3599597503f2af90311c82e2ca92dc06c789a', 'license': 'MIT'},
        }
    for name in ['libhrx.so', 'libhrx.so.0']:
        components[name]['statically_linked'].pop('CORE-MATH adaptation')
    components['libloomc.so']['generated_code_inputs'] = {
        'AMD GPU ISA XML': {'version': '2026-03-05', 'license': 'MIT'},
        'SPIRV-Headers': {'version': 'vulkan-sdk-1.4.357.0', 'license': 'MIT'},
        'Vulkan-Headers': {'version': 'vulkan-sdk-1.4.357.0', 'license': 'MIT'},
    }
    # These remain host OS dependencies; nothing else may escape the bundle.
    host = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1',
            'libstdc++.so.6', 'libgcc_s.so.1', 'libatomic.so.1', 'ld-linux-x86-64.so.2'}
    for name, entry in components.items():
        missing = set(entry['needed']) - components.keys() - host
        if missing:
            raise ValueError(f'{name} has unbundled dependencies: {missing}')

    source_archive = work / 'hrx-native-sources.tar.gz'
    members = {f'upstream/{p.name}': p for p in downloaded.values() if not p.name.endswith('.tar.zst')}
    members['native/release-inputs.json'] = REPO / 'native/release-inputs.json'
    members['RELEASE.md'] = REPO / 'native/RELEASE.md'
    members['scripts/stage-native-release.py'] = Path(__file__).resolve()
    members['scripts/rebuild-hrx.sh'] = REPO / 'scripts/rebuild-hrx.sh'
    members['scripts/build-gpu-runtime-container.sh'] = REPO / 'scripts/build-gpu-runtime-container.sh'
    members['build-packages.txt'] = build / 'build-packages.txt'
    members['scripts/fetch-native-inputs.py'] = REPO / 'scripts/fetch-native-inputs.py'
    members['native/RELEASE.md'] = REPO / 'native/RELEASE.md'
    members['patches/loom/base-revision'] = REPO / 'patches/loom/base-revision'
    for path in sorted((REPO / 'patches/loom').glob('*.patch')):
        members[f'patches/loom/{path.name}'] = path
    for path in sorted(license_dir.glob('*.txt')):
        members[f'native/licenses/{path.name}'] = path
    with source_archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tar:
            for name, path in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    source_url = f'https://github.com/zacharydenton/hrx-rs/releases/download/{args.release_tag}/{source_archive.name}'
    source_record = {'url': source_url, 'sha256': digest(source_archive)}
    inventory = {'schema': 1, 'status': 'complete', 'components': components,
                 'license_files': license_evidence, 'corresponding_source': source_record,
                 'system_dependencies_not_shipped': sorted(host),
                 'review_scope': 'License identifiers confirmed against the pinned source files. AMD build identity is established by the public artifact manifest and TheRock build records; this is not a bit-reproducibility claim.'}
    write_json(REPO / 'THIRD-PARTY.json', inventory)
    shutil.copyfile(REPO / 'THIRD-PARTY.json', stage / 'THIRD-PARTY.json')
    provenance = {'schema': 1, 'hrx': inputs['hrx'], 'amd_build': inputs['therock'],
                  'upstream_artifacts': {key: spec for key, spec in inputs['downloads'].items() if key.endswith('.zst')},
                  'build_recipe': 'scripts/rebuild-hrx.sh', 'corresponding_source': source_record,
                  'compiler': (build / 'compiler-version.txt').read_text().strip(),
                  'cmake_cache_sha256': digest(build / 'CMakeCache.txt'),
                  'built_components': ['libhrx.so', 'libhrx.so.0', 'libloomc.so'],
                  'binary_sha256': {name: entry['sha256'] for name, entry in components.items()}}
    write_json(stage / 'provenance.json', provenance)
    notice = ('HRX native distribution\n\n'
              'This distribution contains software from the projects below. Their original\n'
              'copyright and permission notices are reproduced in the LICENSE-*.txt files.\n'
              'The MIT license for hrx-rs does not replace these component licenses.\n\n')
    for name, entry in components.items():
        notice += f"{name}: {entry['project']} {entry['version']}\n  {entry['license']['spdx']}\n  " + ', '.join(entry['license']['files']) + '\n'
    notice += ('\nrocprofiler-register embeds fmt 11.1.4 and glog 0.7.1. HSA embeds the\n'
               'ROCT thunk and its Nginx-derived rbtree. HRX/Loom include IREE and the\n'
               'CORE-MATH adaptation. Zstandard includes xxHash by Yann Collet / Meta.\n'
               'Compiler tables use AMD GPU ISA XML (MIT); compiler/runtime builds also\n'
               'use HSA, SPIR-V and Vulkan headers. Their source licenses are included.\n\n'
               'HRX/Loom are modified by the eight patches distributed with the source.\n'
               'AMD modifies sysdeps library names and symbol versions; its complete build\n'
               'recipes and patch scripts are in the accompanying TheRock source archive.\n'
               'No further changes are made to the AMD library bytes.\n\n'
               'libelf is distributed under LGPL-3.0-or-later; libnuma under LGPL-2.1-only.\n'
               'Corresponding source, modification/build scripts, and all license texts are\n'
               'available alongside the binary archive at no charge:\n'
               f'{source_url}\nSHA-256: {source_record["sha256"]}\n\n'
               'You may modify or replace these shared libraries. For a local replacement,\n'
               'copy the runtime directory, replace the library while preserving its ABI,\n'
               'and set HRX_RUNTIME_DIR to that directory. This bypasses the cache integrity\n'
               'checks. Reverse engineering for debugging modifications to these LGPL\n'
               'libraries is permitted. See RELEASE.md in the source archive for build\n'
               'instructions and LICENSE-elfutils-* / LICENSE-numactl-* for full terms.\n')
    (REPO / 'NOTICE').write_text(notice)
    shutil.copyfile(REPO / 'NOTICE', stage / 'NOTICE')
    print(f'Staged {len(components)} libraries and {len(license_evidence)} license/attribution files in {stage}')
    print(f'Corresponding source: {source_archive} ({source_record["sha256"]})')


if __name__ == '__main__':
    main()
