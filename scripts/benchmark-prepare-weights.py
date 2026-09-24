#!/usr/bin/env python3
"""Screen mapped-page preparation with fresh-process host reads or uploads.

This cannot qualify complete model loading. DONTNEED is advice, not proof
of cold storage. It applies only to the explicitly named tensor's file range.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--file', type=Path, required=True)
    p.add_argument('--tensor', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--upload', action='store_true', help='Measure a completed device upload and verify all bytes')
    a = p.parse_args()
    if a.output.exists(): p.error('output exists; choose a fresh record')
    with a.file.open('rb') as checkpoint:
        size = struct.unpack('<Q', checkpoint.read(8))[0]
        if size > 100_000_000: p.error('checkpoint header is too large')
        header = json.loads(checkpoint.read(size))
    lo, hi = header[a.tensor]['data_offsets']
    start, length = 8 + size + lo, hi - lo
    if length <= 0: p.error('tensor must be nonempty')
    with a.binary.open('rb') as binary:
        identity = hashlib.file_digest(binary, 'sha256').hexdigest()
    report = dict(binary=str(a.binary.resolve()), binary_sha256=identity,
                  file=str(a.file.resolve()), tensor=a.tensor, tensor_header=header[a.tensor],
                  timed_scope='preparation plus completed device upload' if a.upload else 'preparation plus complete host read', promotion_qualified=False,
                  runs=[], comparisons={})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for state in ['warm', 'DONTNEED-advised']:
            for pair in range(5):
                modes = ['none', 'advice', 'populate'] if pair % 2 == 0 else ['populate', 'advice', 'none']
                for mode in modes:
                    with a.file.open('rb') as checkpoint:
                        if state == 'DONTNEED-advised':
                            os.posix_fadvise(checkpoint.fileno(), start, length, os.POSIX_FADV_DONTNEED)
                        else:
                            checkpoint.seek(start)
                            remaining = length
                            while remaining:
                                block = checkpoint.read(min(16 << 20, remaining))
                                if not block: raise ValueError('truncated tensor')
                                remaining -= len(block)
                    command = [str(a.binary.resolve()), str(a.file.resolve()), a.tensor, mode]
                    if a.upload: command.append('--upload')
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    run = dict(cache_state=state, pair=pair, mode=mode, exit_code=result.returncode)
                    report['runs'].append(run)
                    if result.returncode:
                        run['stderr'] = result.stderr
                        raise RuntimeError('page preparation failed')
                    run.update(json.loads(result.stdout))
                    # The example has no cache-control authority. Preserve the
                    # collector's explicitly established/advised condition.
                    run['cache_state'] = state
            rows = [r for r in report['runs'] if r['cache_state'] == state]
            if len({r['checksum'] for r in rows}) != 1:
                raise RuntimeError('content checksum changed between modes')
            for candidate in ['advice', 'populate']:
                ratios = []
                for pair in range(5):
                    modes = {r['mode']: r for r in rows if r['pair'] == pair}
                    ratios.append(modes[candidate]['total_ms'] / modes['none']['total_ms'])
                report['comparisons'][f'{state}:{candidate}'] = dict(paired_time_ratios=ratios, median_time_ratio=statistics.median(ratios))
    finally:
        with a.output.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
    print(json.dumps(report['comparisons'], indent=2))


if __name__ == '__main__':
    main()
