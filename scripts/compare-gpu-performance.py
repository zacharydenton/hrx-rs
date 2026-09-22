#!/usr/bin/env python3
"""Compare two stream_bench executables, alternating process order with explicit native runtime directories."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--start-pair', type=int, default=0)
parser.add_argument('--baseline-runtime', type=Path)
parser.add_argument('--candidate-runtime', type=Path)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--samples', type=int, default=31)
parser.add_argument('--max-ratio', type=float, default=1.05, help='reference cost ratio; enforced only with --strict')
parser.add_argument('--strict', action='store_true', help='fail when the reference ratio is exceeded')
args = parser.parse_args()
if args.runs < 5 or args.samples < 9 or args.max_ratio <= 0:
    parser.error('use at least five processes, nine samples, and a positive ratio')
environment = os.environ.copy()
environment['HRX_BENCH_SAMPLES'] = str(args.samples)
executables = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
measurements = {arm: [] for arm in executables}
for pair in range(args.start_pair, args.start_pair + args.runs):
    for arm in (['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']):
        arm_environment = environment.copy()
        runtime = getattr(args, arm + '_runtime')
        if runtime:
            arm_environment['HRX_RUNTIME_DIR'] = str(runtime.resolve())
            for key in ['HRX_AMDF_LIBRARY', 'HRX_FABRIC_LIBRARY', 'HRX_LOOM_LIBRARY']:
                arm_environment.pop(key, None)
        result = subprocess.run([str(executables[arm])], env=arm_environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
        record = json.loads(result.stdout)
        measurements[arm].append(record)
        print(json.dumps({'pair': pair, 'arm': arm, 'metrics': record}), flush=True)
summary = {}
for metric in measurements['baseline'][0]:
    if metric == 'graph_scheduling_difference_ns_per_kernel':
        continue  # A difference can be near zero; its ratio is not a regression metric.
    before = statistics.median(item[metric] for item in measurements['baseline'])
    after = statistics.median(item[metric] for item in measurements['candidate'])
    ratio = before / after if metric.endswith('_gib_s') else after / before
    summary[metric] = {'baseline': before, 'candidate': after, 'cost_ratio': ratio}
passed = all(item['cost_ratio'] <= args.max_ratio for item in summary.values())
print(json.dumps({'within_target': passed, 'target': args.max_ratio, 'strict': args.strict, 'summary': summary}), flush=True)
raise SystemExit(1 if args.strict and not passed else 0)
