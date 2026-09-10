#!/usr/bin/env python3
"""Fetch the hash-pinned OpenCV Lena sample into a local benchmark directory."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/vision-bench-lena'))
    args = parser.parse_args()
    sources = json.loads(Path(__file__).with_name('sample.json').read_text())
    for name, source in sources.items():
        destination = args.output / name
        if destination.exists():
            data = destination.read_bytes()
        else:
            with urlopen(source['url'], timeout=30) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise RuntimeError(f'{destination}: unexpected SHA-256; refusing to replace or use it')
        args.output.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open('xb') as stream:
                stream.write(data)
        print(f'{destination}: SHA-256 verified')
    (args.output / 'SOURCES.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
