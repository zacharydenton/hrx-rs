#!/usr/bin/env python3
"""Compare prepared heterogeneous replay with direct native execution in fresh processes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--max-ratio', type=float, default=1.05, help='reference cost ratio; enforced only with --strict')
parser.add_argument('--strict', action='store_true', help='fail when the reference ratio is exceeded')
args = parser.parse_args()
if args.runs < 5 or args.max_ratio <= 0:
    parser.error('qualification requires at least five process pairs and a positive ratio')
root = Path(__file__).resolve().parents[1]
results = []
for _ in range(args.runs):
    result = subprocess.run([
        'cargo', 'test', '--release', '--all-features', '--lib',
        'heterogeneous_latency_against_direct_backend', '--',
        '--ignored', '--nocapture', '--test-threads=1',
    ], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.stdout)
    match = re.search(r'paired_direct_p50_us=([\d.]+) scheduled_p50_us=([\d.]+) ratio=([\d.]+) scheduled_p95_us=([\d.]+)', result.stdout)
    if not match:
        raise SystemExit('benchmark did not produce measurements:\n' + result.stdout)
    results.append(dict(zip(('direct_p50_us', 'scheduled_p50_us', 'ratio', 'scheduled_p95_us'), map(float, match.groups()))))
    print(json.dumps(results[-1]), flush=True)
passed = all(result['ratio'] <= args.max_ratio for result in results)
print(json.dumps({'within_target': passed, 'target': args.max_ratio, 'strict': args.strict, 'process_pairs': len(results)}))
raise SystemExit(1 if args.strict and not passed else 0)
