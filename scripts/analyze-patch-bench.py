#!/usr/bin/env python3
"""Summarize alternating paired timings using a seeded paired bootstrap."""
import argparse, json, math, random, statistics
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('files', nargs='+', type=Path)
args = parser.parse_args()
results = {}
for path in args.files:
    cases = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        case = cases.setdefault(row['case'], {'pairs': []})
        if 'pair' in row:
            case['pairs'].append(math.log(row['baseline_ns'] / row['candidate_ns']))
        else:
            case['evidence'] = row
    for case in cases.values():
        pairs = case.pop('pairs')
        if len(pairs) < 30:
            raise ValueError(f'{path}: insufficient pairs')
        rng = random.Random(20260922)
        samples = sorted(sum(rng.choices(pairs, k=len(pairs))) / len(pairs) for _ in range(10000))
        improvement = lambda value: 100 * (1 - math.exp(-value))
        estimate = improvement(statistics.mean(pairs))
        lo, hi = improvement(samples[249]), improvement(samples[9749])
        changed = case['evidence']['baseline']['artifact'] != case['evidence']['candidate']['artifact']
        case.update(code_changed=changed, pairs=len(pairs), improvement_percent=estimate, ci95_percent=[lo,hi],
                    gain=changed and estimate >= 3 and lo > 0, regression=changed and estimate < -3 and hi < 0)
    results[path.stem] = cases
print(json.dumps({'method':'geometric paired speed ratio; 10000 bootstrap resamples, seed 20260922',
                  'results':results}, indent=2))
