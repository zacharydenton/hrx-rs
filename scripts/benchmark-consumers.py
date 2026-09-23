#!/usr/bin/env python3
"""Measure saved consumer builds in alternating fresh processes.

Provide saved binaries for the public model projects under baseline-bin and candidate-bin.
Each invocation appends raw per-process records; no failed run is discarded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import time

repository = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description='Alternate saved consumer baseline/candidate binaries; GPU must be healthy.')
p.add_argument('--artifacts', type=Path, default=repository / 'artifacts/consumer-migration')
p.add_argument('--workspace', type=Path, default=repository.parent)
p.add_argument('--pairs', type=int, default=3)
p.add_argument('--only', nargs='*')
p.add_argument('--baseline-runtime', type=Path, help='Override the baseline native bundle directory')
p.add_argument('--candidate-runtime', type=Path, help='Override the candidate native bundle directory')
args = p.parse_args()
if args.pairs < 1:
    p.error('--pairs must be positive')
root = args.artifacts.resolve()
try:
    Path('/sys/class/drm/card1/device/gpu_busy_percent').read_text()
except OSError as error:
    raise SystemExit(f'GPU unavailable; restore device health before benchmarking: {error}')
runtimes = {arm: getattr(args, f'{arm}_runtime') for arm in ('baseline', 'candidate')}
for arm, runtime in runtimes.items():
    if runtime is not None:
        runtime = runtime.resolve()
        if not (runtime / 'libamdf.so').is_file():
            p.error(f'{arm} runtime does not contain libamdf.so: {runtime}')
        runtimes[arm] = runtime
inputs = root / 'inputs'
inputs.mkdir(exist_ok=True)
(inputs / 'arcface.rgb').write_bytes(bytes((i * 37 % 256 for i in range(112 * 112 * 3))))
(inputs / 'dino.f32').write_bytes(b''.join((struct.pack('<f', math.sin(i * 0.037) * 0.6) for i in range(3 * 224 * 224))))
snapshot = str(Path.home() / '.cache/huggingface/hub/models--Comfy-Org--MiniMax-H3/snapshots/a98869194787969724c7425d95d0ed73ce9202af')
plans = {
    'hrxdb': (
        'hrxdb',
        'hrxdb-bench',
        lambda out: [
            '--rows',
            '1000000',
            '--dimensions',
            '384',
            '--samples',
            '30',
            '--k',
            '10',
            '--output',
            str(out / 'report.json'),
        ],
    ),
    'dinov3': (
        'dinov3-hrx',
        'dinov3-hrx',
        lambda out: [
            '--offline',
            '--max-batch',
            '1',
            '--input',
            str(inputs / 'dino.f32'),
            '--benchmark',
            '30',
            '--output',
            str(out / 'output.f32'),
        ],
    ),
    'arcface': (
        'arcface-hrx',
        'arcface-hrx',
        lambda out: [
            '--offline',
            '--max-batch',
            '1',
            '--input',
            str(inputs / 'arcface.rgb'),
            '--benchmark',
            '30',
            '--output',
            str(out / 'output.f32'),
        ],
    ),
    'scrfd': (
        'scrfd-hrx',
        'scrfd-hrx',
        lambda out: [
            '--offline',
            '--max-batch',
            '1',
            '--input',
            str(args.workspace / 'scrfd-hrx/tests/fixtures/t1.png'),
            '--benchmark',
            '30',
            '--output',
            str(out / 'output.json'),
        ],
    ),
    'h3_audio': (
        'h3-hrx',
        'bench_runtime',
        lambda out: [
            snapshot,
            str(out / 'output.f32'),
        ],
    ),
    'krea_generation': (
        'krea2-hrx',
        'bench_runtime',
        lambda out: [
            str(out / 'output.rgb'),
        ],
    ),
}
if args.only and set(args.only) - set(plans):
    p.error(f'unknown workloads: {sorted(set(args.only) - set(plans))}')
campaign = str(time.time_ns())
results = []
for name, (repo, binary, flags) in plans.items():
    if args.only and name not in args.only:
        continue
    for pair in range(args.pairs):
        for arm in ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']:
            out = root / 'measurements' / campaign / name / f'{pair}-{arm}'
            out.mkdir(parents=True, exist_ok=True)
            exe = root / f'{arm}-bin' / repo / binary
            env = dict(os.environ, HF_HUB_OFFLINE='1', HRX_OFFLINE='1')
            for override in ['HRX_RUNTIME_DIR', 'HRX_LOOM_LIBRARY', 'HRX_GPU_LIBRARY', 'HRX_ROCR_LIBRARY', 'HRX_FABRIC_LIBRARY', 'HRX_AMDF_LIBRARY', 'HRX_NPU_LIBRARY']:
                env.pop(override, None)
            runtime = runtimes[arm]
            if runtime is not None:
                env['HRX_RUNTIME_DIR'] = str(runtime)
            command = [str(exe), *flags(out)]
            record = {'output_directory': str(out), 'workload': name, 'pair': pair, 'arm': arm, 'command': command, 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'started': time.time(), 'gpu_busy_before': Path('/sys/class/drm/card1/device/gpu_busy_percent').read_text().strip(), 'memory_pressure_before': Path('/proc/pressure/memory').read_text()}
            record['runtime_override'] = str(runtime) if runtime else None
            record['runtime_sha256'] = {lib.name: hashlib.sha256(lib.read_bytes()).hexdigest() for lib in sorted(runtime.glob('*.so'))} if runtime else None
            print(name, pair, arm, flush=True)
            with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
                try:
                    run = subprocess.run(command, cwd=repository, env=env, stdout=stdout, stderr=stderr, timeout=600)
                    record['exit_code'] = run.returncode
                except subprocess.TimeoutExpired:
                    record['exit_code'] = 124
            record['wall_seconds'] = time.time() - record['started']
            records = []
            for line in (out / 'stdout.log').read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            record['reports'] = records
            if record['exit_code'] == 0:
                if name == 'hrxdb':
                    report = json.loads((out / 'report.json').read_text())
                    record['median_ms'] = report['trials'][0]['search_median_ms']
                else:
                    record['median_ms'] = records[-1]['median_ms']
            for f in out.glob('output.*'):
                record.setdefault('outputs', {})[f.name] = {'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes': f.stat().st_size}
            (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            results.append(record)
            with (root / 'measurement-runs.jsonl').open('a') as history:
                history.write(json.dumps(record) + '\n')
            print('result', record['exit_code'], record.get('median_ms'), flush=True)
            if record['exit_code'] != 0:
                print((out / 'stderr.log').read_text()[-1500:], flush=True)
                break
        else:
            continue
        break

summary = {'campaign': campaign, 'requested_pairs': args.pairs, 'workloads': {}}
for name in dict.fromkeys(record['workload'] for record in results):
    runs = [record for record in results if record['workload'] == name]
    complete = [record for record in runs if record['exit_code'] == 0]
    arms = {arm: [record for record in complete if record['arm'] == arm]
            for arm in ['baseline', 'candidate']}
    entry = {'complete': all(len(arm) == args.pairs for arm in arms.values())}
    if entry['complete']:
        medians = {arm: statistics.median(record['median_ms'] for record in records)
                   for arm, records in arms.items()}
        ratios = [arms['candidate'][pair]['median_ms'] / arms['baseline'][pair]['median_ms']
                  for pair in range(args.pairs)]
        entry.update(median_ms=medians,
                     median_change_percent=100 * (medians['candidate'] / medians['baseline'] - 1),
                     paired_change_percent=[100 * (ratio - 1) for ratio in ratios])
        captured = [record.get('outputs') for record in complete]
        entry['captured_outputs_identical'] = (
            all(output == captured[0] for output in captured) if all(captured) else None)
    summary['workloads'][name] = entry
summary_path = root / 'measurements' / campaign / 'summary.json'
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
print(f'Summary: {summary_path}', flush=True)
if any(not entry['complete'] or entry.get('captured_outputs_identical') is False
       for entry in summary['workloads'].values()):
    raise SystemExit('Campaign incomplete or captured outputs differ; inspect the saved runs.')
