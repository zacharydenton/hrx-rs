#!/usr/bin/env python3
"""Release-time only: package an already staged native runtime; never builds hrx-system.
Usage: bundle.py RUNTIME OUTPUT_DIRECTORY URL [REVISION]
All symlinks are materialized so archive extraction requires no link traversal.
"""
import gzip, hashlib, io, json, pathlib, sys, tarfile
source, output, url = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
output.mkdir(parents=True, exist_ok=True)
files = {p.name: p.read_bytes() for p in sorted(source.iterdir()) if p.is_file() and p.name != 'manifest.json'}
for name in ('loom-compile', 'libhrx.so', 'libhsa-runtime64.so.1'):
    assert name in files, f'missing {name}'
archive = output / 'hrx-linux-x86_64-gfx1151.tar.gz'
with archive.open('wb') as raw:
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o755, 0
                tar.addfile(info, io.BytesIO(data))
manifest = dict(schema=1, target='x86_64-unknown-linux-gnu-gfx1151', revision=sys.argv[4] if len(sys.argv)>4 else 'local', url=url,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                files={n: hashlib.sha256(d).hexdigest() for n,d in files.items()})
(output/'bundle.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(archive)
