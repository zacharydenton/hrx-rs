#!/usr/bin/env python3
"""Stage the unified native runtime, licenses, provenance and rebuild sources."""
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--release-tag', required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    build = (args.build or work / 'hrx-clang').resolve()
    source = work / 'hrx-source'
    inputs = json.loads((REPO / 'native/release-inputs.json').read_text())
    downloaded = {}
    for name, spec in inputs['downloads'].items():
        path = next((p for p in [work / spec['archive'], work / 'sources' / spec['archive']] if p.is_file()), None)
        if path is None or digest(path) != spec['sha256']:
            raise ValueError(f'Missing or corrupt pinned input: {name}')
        downloaded[name] = path
    patches = {f'patches/loom/{name}': sha for name, sha in inputs['hrx']['patches'].items()}
    patches.update(inputs['native_patches'])
    for name, sha in patches.items():
        if digest(REPO / name) != sha:
            raise ValueError(f'Patch differs from pin: {name}')
        subprocess.run(['patch', '-p1', '--dry-run', '--batch', '--reverse', '--input', str(REPO / name)],
                       cwd=source, check=True, stdout=subprocess.DEVNULL)
    stage = work / 'stage-amdf'
    stage.mkdir(exist_ok=True)
    if any(stage.iterdir()):
        raise ValueError(f'Stage must be empty: {stage}')
    deps = build / '_deps'
    licenses = REPO / 'native/licenses'
    evidence = {}
    def license_file(name, path=None, text=None, origin=None):
        destination = licenses / name
        if path is not None:
            shutil.copyfile(path, destination)
            origin = str(path.relative_to(work)) if path.is_relative_to(work) else str(path.relative_to(REPO))
        else:
            destination.write_text(text)
        shutil.copyfile(destination, stage / name)
        evidence[name] = {'source_file': origin, 'sha256': digest(destination)}
    for name, path in {
        'LICENSE-HRX.txt': source / 'LICENSE',
        'LICENSE-HRX-RS.txt': REPO / 'LICENSE',
        'LICENSE-CXX.txt': deps / 'cxx-src/LICENSE',
        'LICENSE-HSA-headers.txt': deps / 'hsa_runtime_headers-src/LICENSE.txt',
    }.items():
        license_file(name, path=path)
    # These pinned source archives need not be unpacked by a native-only build.
    for name, dependency, relative in [
        ('LICENSE-SPIRV-Headers.txt', 'spirv_headers', 'LICENSE'),
        ('LICENSE-Vulkan-Headers.txt', 'vulkan_headers', 'LICENSES/MIT.txt'),
    ]:
        with tarfile.open(downloaded[dependency]) as archive:
            matches = [member for member in archive.getmembers()
                       if member.isfile() and '/'.join(Path(member.name).parts[1:]) == relative]
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous license in {dependency}: {relative}')
            with archive.extractfile(matches[0]) as stream:
                license_file(name, text=stream.read().decode('utf-8'),
                             origin=f'{downloaded[dependency].name}:{matches[0].name}')
    for name in ['LICENSE-CORE-MATH.txt', 'LICENSE-AMDGPU-ISA.txt']:
        shutil.copyfile(licenses / name, stage / name)
        evidence[name] = {'source_file': 'pinned HRX math and AMD ISA XML source notices', 'sha256': digest(licenses / name)}
    headers = sorted((deps / 'libdrm_headers-src/include/drm').glob('*.h'))
    # Preserve the complete source headers, including every embedded notice.
    license_file('LICENSE-linux-uapi.txt', text='\n'.join(
        f'Source: {p.relative_to(work)}\n{p.read_text()}' for p in headers +
        [p for name in inputs['downloads'] if name.startswith('linux_')
         for p in (deps / f'{name}-src/file').rglob('*.h')]),
        origin='pinned libdrm and Linux/amdxdna syscall headers; GPL-2.0 WITH Linux-syscall-note and permissive per-file notices')
    for name in ['linux-gpl2-license', 'linux-syscall-note']:
        license_file(f'LICENSE-{name}.txt', path=downloaded[name])
    host = {'libc.so.6','libm.so.6','libpthread.so.0','libdl.so.2','librt.so.1',
            'libstdc++.so.6','libgcc_s.so.1','libatomic.so.1','ld-linux-x86-64.so.2'}
    components = {}
    for name, relative in {'libamdf.so':'libamdf/libamdf.so',
                           'libloomc.so':'loom/binding/c/libloomc.so',
                           'libhrx_fabric.so':'libhrx_fabric.so'}.items():
        shutil.copyfile(build / relative, stage / name)
        elf = subprocess.check_output(['readelf','-d',str(stage/name)],text=True)
        needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', elf)
        if set(needed) - host:
            raise ValueError(f'{name}: unexpected dependency {needed}')
        components[name] = {'project':'HRX / libamdf / Loom / hrx-rs native bridge',
            'version': inputs['hrx']['revision'], 'sha256':digest(stage/name), 'needed':needed,
            'sources':inputs['downloads'], 'origin':'Fresh native-only Ubuntu 26.04 build',
            'license':{'spdx':'Apache-2.0 WITH LLVM-exception AND MIT AND NCSA',
                       'status':'confirmed','files':sorted(evidence),
                       'basis':'Pinned project licenses and per-file notices; syscall-only headers retain the Linux syscall exception.'}}
    members = {f'upstream/{p.name}':p for p in downloaded.values()}
    for directory in ['native/amdf','patches/loom','native/licenses']:
        for path in sorted((REPO/directory).rglob('*')):
            if path.is_file() and (directory != 'native/licenses' or path.name in evidence):
                members[str(path.relative_to(REPO))] = path
    for name in ['native/release-inputs.json','native/RELEASE.md','scripts/stage-native-release.py',
                 'scripts/rebuild-hrx.sh','scripts/build-amdf.sh','scripts/build-gpu-runtime-container.sh',
                 'scripts/fetch-native-inputs.py','scripts/seed-native-file-cache.py']:
        members[name] = REPO/name
    members['build-packages.txt'] = build/'build-packages.txt'
    archive = work/'hrx-native-sources.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=1) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for name,path in sorted(members.items()):
                info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644
                with path.open('rb') as stream:tar.addfile(info,stream)
    source_record={'url':f'https://github.com/zacharydenton/hrx-rs/releases/download/{args.release_tag}/{archive.name}', 'sha256':digest(archive)}
    inventory={'schema':1,'status':'complete','components':components,'license_files':evidence,
               'corresponding_source':source_record,'system_dependencies_not_shipped':sorted(host),
               'review_scope':'Native libamdf, Loom with the MIT C++ importer, and the hrx-rs bridge. No ROCr, XRT, IRON or vendor runtime libraries are shipped. Rebuild provenance is recorded; bit reproducibility is not claimed.'}
    write_json(REPO/'THIRD-PARTY.json',inventory)
    shutil.copyfile(REPO/'THIRD-PARTY.json',stage/'THIRD-PARTY.json')
    bridge_inputs={str(p.relative_to(REPO)):digest(p) for p in sorted((REPO/'native/amdf').glob('*')) if p.is_file()}
    write_json(stage/'provenance.json',{'schema':1,'inputs':inputs,'bridge_inputs':bridge_inputs,
        'build_recipe':'scripts/rebuild-hrx.sh','corresponding_source':source_record,
        'compiler':(build/'compiler-version.txt').read_text().strip(),
        'cmake_cache_sha256':digest(build/'CMakeCache.txt'),
        'binary_sha256':{n:e['sha256'] for n,e in components.items()}})
    notice='HRX 0.8 native distribution\n\nlibamdf and Loom derive from HRX/IREE under Apache-2.0 WITH LLVM-exception.\nThe C++ importer includes Roberto Raggi\'s MIT-licensed cplusplus parser.\nCORE-MATH and AMD ISA tables retain their MIT notices; HSA headers retain NCSA.\nLinux syscall headers retain their original notices and syscall exception.\nAll component license texts and header notices accompany the libraries.\nThe hrx-rs bridge is MIT licensed.\n\nModified source, patches and build recipes:\n'+source_record['url']+'\nSHA-256: '+source_record['sha256']+'\n'
    (stage/'NOTICE').write_text(notice)
    (REPO/'NOTICE').write_text(notice+'\nBenchmark fixtures in native/qualification derive from arcface-hrx and krea2-hrx (MIT),\nand h3-hrx (Apache-2.0); exact revisions and license texts accompany those files.\n')
    print(stage)
if __name__=='__main__':main()
