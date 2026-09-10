#!/usr/bin/env python3
"""Real SCRFD + matching DINOv3-L image-pipeline benchmark; no model downloads.

GPU SCRFD uses scrfd-loom. DINO uses resident HF/ROCm or a precompiled XDNA
VitisAI artifact. This measures application placement, not hrx graph overhead.
See README.md for timing boundaries, dependency setup and reproduction.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import resource
import select
import subprocess
import sys
import threading
import time
import traceback


def emit(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_hashes(directory):
    """Hash the config and every safetensors shard from a local HF checkpoint."""
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    names = {'config.json'}
    if index.is_file():
        names.add(index.name)
        names.update(json.loads(index.read_text())['weight_map'].values())
    else:
        names.add('model.safetensors')
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'checkpoint index contains an unsafe shard path: {name}')
    return {name: digest(directory / name) for name in sorted(names)}


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def worker(config_path, role, response_fd):
    import numpy as np
    config = json.loads(Path(config_path).read_text())
    response = os.fdopen(response_fd, 'w', buffering=1)
    start = time.perf_counter()
    try:
        torch = None
        arrays = {
            name: np.load(Path(config['output']) / f'{name}.npy', mmap_mode='r+')
            for name in ('pixels', 'canvas', 'mask') + (() if role == 'face' else (role + '_result',))
        }
        metadata = {'role': role, 'pid': os.getpid(), 'python': sys.version, 'numpy': np.__version__}
        if role == 'face':
            sys.path.insert(0, config['scrfd'])
            import cv2
            cv2.setNumThreads(1)
            from scrfd_loom import SCRFDLoom
            model = SCRFDLoom(max_batch=config['gpu_batch'])
            metadata['backend'] = 'SCRFD-10GF / Loom HIP'
        elif role == 'reference':
            import torch
            from transformers import AutoModel
            torch.set_num_threads(config['cpu_threads'])
            torch.set_num_interop_threads(1)
            model = AutoModel.from_pretrained(config['gpu_model'], local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval().cpu()
            metadata.update(backend='HF DINOv3-ViT-L/16 float32 CPU eager reference', torch=torch.__version__)
        elif role == 'gpu':
            import torch
            from transformers import AutoModel
            torch.set_num_threads(config['cpu_threads'])
            torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('GPU unavailable; CPU fallback is forbidden')
            model = AutoModel.from_pretrained(config['gpu_model'], local_files_only=True,
                                             dtype=torch.bfloat16, attn_implementation='sdpa').eval().cuda()
            x = torch.zeros((config['gpu_batch'], 3, 224, 224), device='cuda', dtype=torch.bfloat16)
            mask = torch.ones((config['gpu_batch'], 196), device='cuda', dtype=torch.float32)

            def forward():
                hidden = model(pixel_values=x).last_hidden_state.float()
                pooled = (hidden[:, 5:] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                return torch.stack((hidden[:, 0], pooled), dim=1)

            mode = config['gpu_mode']
            if mode == 'compile':
                forward = torch.compile(forward, mode='max-autotune', fullgraph=True, dynamic=False)
            with torch.inference_mode():
                if mode == 'graph':
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            result = forward()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = forward()
                else:
                    for _ in range(3):
                        result = forward()
                torch.cuda.synchronize()
            metadata.update(backend='HF DINOv3-ViT-L/16 bf16 SDPA', mode=mode,
                            torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = config['cpu_threads']
            options.inter_op_num_threads = 1
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            model = ort.InferenceSession(config['npu_model'],
                                        sess_options=options, providers=['VitisAIExecutionProvider'])
            if 'VitisAIExecutionProvider' not in model.get_providers():
                raise RuntimeError('NPU execution provider did not load')
            input_name = model.get_inputs()[0].name
            if model.get_inputs()[0].shape != [1, 3, 224, 224]:
                raise RuntimeError('NPU artifact must use fixed [1,3,224,224] input')
            metadata.update(backend='DINOv3-ViT-L/16 VitisAI EPContext',
                            onnxruntime=ort.__version__, providers=model.get_providers(),
                            cpu_fallback=False)
        devices = set()
        for fd in Path('/proc/self/fd').iterdir():
            try:
                target = str(fd.readlink())
                if target.startswith(('/dev/accel/', '/dev/dri/', '/dev/kfd')):
                    devices.add(target)
            except FileNotFoundError:
                pass
        metadata['open_accelerator_devices'] = sorted(devices)
        if role == 'npu' and not any(device.startswith('/dev/accel/') for device in devices):
            raise RuntimeError('NPU session did not open an accelerator device')
        metadata['setup_seconds'] = time.perf_counter() - start
        metadata['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        emit(response, {'ready': metadata})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('stop'):
                break
            requests = request.get('batch', [request])
            slots = [item['slot'] for item in requests]
            if not 1 <= len(slots) <= config['gpu_batch']:
                raise ValueError('invalid microbatch length')
            begin, cpu_begin = time.perf_counter(), cpu_seconds()
            details = [{} for _ in requests]
            if role == 'face':
                detections = model.detect_letterboxed(arrays['canvas'][slots],
                    [item['scale'] for item in requests], image_shapes=[tuple(item['shape']) for item in requests])
                details = [{'boxes': boxes.tolist(), 'landmarks': landmarks.tolist()} for boxes, landmarks in detections]
            elif role == 'gpu':
                with torch.inference_mode():
                    # Fixed graph shape; only partial final microbatches are padded.
                    pixels = np.zeros((config['gpu_batch'], 3, 224, 224), dtype=np.float32)
                    masks = np.ones((config['gpu_batch'], 196), dtype=np.float32)
                    pixels[:len(slots)] = arrays['pixels'][slots]
                    masks[:len(slots)] = arrays['mask'][slots]
                    x.copy_(torch.from_numpy(pixels))
                    mask.copy_(torch.from_numpy(masks))
                    if mode == 'graph':
                        graph.replay()
                    else:
                        result = forward()
                    arrays['gpu_result'][slots] = result[:len(slots)].cpu().numpy()
            else:
                from preprocess import pool_hidden
                # Both reference and NPU execute one image at a time.
                for slot in slots:
                    pixels = arrays['pixels'][slot:slot + 1]
                    if role == 'reference':
                        with torch.inference_mode():
                            hidden = model(pixel_values=torch.from_numpy(pixels.copy())).last_hidden_state.numpy()[0]
                    else:
                        hidden = model.run(None, {input_name: pixels})[0][0]
                    arrays[role + '_result'][slot] = pool_hidden(hidden, arrays['mask'][slot])
            elapsed = time.perf_counter() - begin
            cpu_used = cpu_seconds() - cpu_begin
            results = [{'id': item['id'], 'service_ms': elapsed * 1000 / len(slots),
                        'cpu_seconds': cpu_used / len(slots), **detail} for item, detail in zip(requests, details)]
            emit(response, {'id': request['id'], 'batch': results} if 'batch' in request else results[0])
    except BaseException:
        emit(response, {'error': traceback.format_exc()})
        raise
    finally:
        response.close()


class Worker:
    def __init__(self, config, role, environment):
        self.role = role
        self.timeout = config['timeout']
        self.lock = threading.Lock()
        reader, writer = os.pipe()
        self.response = os.fdopen(reader, 'rb', buffering=0)
        self.read_buffer = bytearray()
        self.log = open(Path(config['output']) / f'{role}.log', 'w')
        executable = config['npu_python'] if role == 'npu' else sys.executable
        self.process = subprocess.Popen([executable, str(Path(__file__).resolve()), '_worker',
            str(Path(config['output']) / 'config.json'), role, str(writer)],
            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
            text=True, bufsize=1, pass_fds=(writer,), env=environment)
        os.close(writer)
        try:
            self.metadata = self.read()['ready']
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.read_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.response], [], [], remaining)[0]:
                raise TimeoutError(f'{self.role} timed out; see {self.log.name}')
            chunk = os.read(self.response.fileno(), 65536)
            if not chunk:
                detail = ' during a response' if self.read_buffer else ''
                raise RuntimeError(f'{self.role} exited{detail}; see {self.log.name}')
            self.read_buffer.extend(chunk)
        line, _, rest = self.read_buffer.partition(b'\n')
        self.read_buffer = bytearray(rest)
        try:
            result = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(f'{self.role} returned an invalid response; see {self.log.name}') from error
        if not isinstance(result, dict):
            raise RuntimeError(f'{self.role} returned a non-object response; see {self.log.name}')
        if 'error' in result:
            raise RuntimeError(f'{self.role} failed; see {self.log.name}\n{result["error"]}')
        return result

    def call(self, request):
        with self.lock:
            emit(self.process.stdin, request)
            result = self.read()
            if result['id'] != request['id']:
                raise RuntimeError('worker returned the wrong image ID')
            return result

    def close(self):
        try:
            if self.process.poll() is None:
                emit(self.process.stdin, {'stop': True})
                self.process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.response.close()
        self.log.close()


def run_mode(mode, count, depth, gpu_batch, prepare, workers, arrays, pool, preprocess_pool):
    """Bounded image slots stay owned until both branch results are copied out."""
    pending, preprocessing = {}, {}
    free = list(range(depth))
    ready, done = [], []
    next_index = 0
    start, cpu_begin = time.perf_counter(), cpu_seconds()
    while next_index < count or preprocessing or ready or pending:
        while next_index < count and free:
            slot = free.pop()
            admitted = time.perf_counter()
            preprocessing[preprocess_pool.submit(prepare, next_index, slot)] = admitted
            next_index += 1
        for future in list(preprocessing):
            if future.done():
                ready.append((future.result(), preprocessing.pop(future)))
        ready.sort(key=lambda item: item[0]['id'])
        for batch_id, (batch, admitted, pair, backend) in list(pending.items()):
            if not all(future.done() for future in pair):
                continue
            responses = pair[0].result() if mode == 'gpu_serial' else tuple(f.result() for f in pair)
            finished = time.perf_counter()
            for index, request in enumerate(batch):
                slot = request['slot']
                done.append({'id': request['id'], 'latency_ms': (finished - admitted[index]) * 1000,
                             'preprocess_ms': request['preprocess_ms'],
                             'face': responses[0]['batch'][index], 'embedding': responses[1]['batch'][index],
                             'backend': backend, 'features': arrays[backend + '_result'][slot].copy()})
                free.append(slot)
            del pending[batch_id]
        while ready:
            backend = 'npu' if mode == 'mixed_parallel' else 'gpu'
            if mode == 'gpu_serial' and pending:
                break
            if mode == 'mixed_balanced':
                busy = {entry[3] for entry in pending.values()}
                available = [r for r in ('gpu', 'npu') if r not in busy]
                if not available:
                    break
                backend = available[0]
                if backend == 'gpu' and len(ready) < gpu_batch and 'npu' in available:
                    backend = 'npu'
            batch_size = 1 if mode == 'mixed_balanced' and backend == 'npu' else gpu_batch
            # Wait for a full GPU batch unless the corpus or bounded window is drained.
            if len(ready) < batch_size and (preprocessing or (next_index < count and free)):
                break
            group, ready = ready[:batch_size], ready[batch_size:]
            batch, admitted = [item[0] for item in group], [item[1] for item in group]
            packet = {'id': batch[0]['id'], 'batch': batch}
            if mode == 'gpu_serial':
                def serial(req=packet):
                    return workers['face'].call(req), workers['gpu'].call(req)
                pair = [pool.submit(serial)]
            else:
                pair = [pool.submit(workers['face'].call, packet), pool.submit(workers[backend].call, packet)]
            pending[packet['id']] = (batch, admitted, pair, backend)
        active = [f for f in preprocessing if not f.done()]
        active.extend(f for _, _, pair, _ in pending.values() for f in pair if not f.done())
        # Refill released slots before waiting, keeping preprocessing ahead of inference.
        if active and not (next_index < count and free):
            futures.wait(active, return_when=futures.FIRST_COMPLETED)
    elapsed = time.perf_counter() - start
    cpu_total = cpu_seconds() - cpu_begin + sum(r['face']['cpu_seconds'] + r['embedding']['cpu_seconds'] for r in done)
    return done, elapsed, cpu_total


def main():
    import numpy as np
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scrfd', type=Path, default=Path.home() / 'code/scrfd-loom')
    parser.add_argument('--gpu-model', type=Path, required=True, help='local HF DINOv3-L checkpoint')
    parser.add_argument('--npu-model', type=Path, required=True, help='standalone DINOv3-L EPContext .onnx file')
    parser.add_argument('--npu-python', type=Path, required=True, help='Ryzen AI Python with NumPy 1.x')
    parser.add_argument('--xrt-root', type=Path, default=Path('/usr'))
    parser.add_argument('--npu-lib-dir', type=Path, action='append', default=[], help='extra native library directory (repeatable)')
    parser.add_argument('--images', type=Path, required=True, help='directory of real JPEG/PNG images')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=16, help='repeat the corpus per timed round')
    parser.add_argument('--depth', type=int, default=32, help='maximum admitted images')
    parser.add_argument('--gpu-batch', type=int, default=8, help='GPU microbatch size, at most 64')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--preprocess-workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--modes', nargs='+', choices=['gpu_serial', 'gpu_parallel', 'mixed_parallel', 'mixed_balanced'],
                        default=['gpu_serial', 'gpu_parallel', 'mixed_parallel', 'mixed_balanced'])
    parser.add_argument('--gpu-mode', choices=['eager', 'graph', 'compile'], default='graph')
    parser.add_argument('--min-cosine', type=float, default=0.997)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.depth, args.cpu_threads, args.gpu_batch, args.preprocess_workers) < 1 or args.gpu_batch > min(64, args.depth):
        parser.error('counts must be positive and gpu-batch must be at most min(64, depth)')
    if not 0 < args.min_cosine <= 1 or args.timeout <= 0:
        parser.error('min-cosine must be in (0, 1] and timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['npu_lib_dir'] = [str(p.resolve()) for p in args.npu_lib_dir]
    gpu_config = json.loads((args.gpu_model / 'config.json').read_text())
    if gpu_config['hidden_size'] != 1024 or gpu_config['num_hidden_layers'] != 24:
        raise ValueError('this benchmark requires matched DINOv3-ViT-L/16 models')
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not paths:
        raise ValueError('provide at least one JPEG/PNG image')
    corpus = [(p, p.read_bytes()) for p in paths]
    if len(set(hashlib.sha256(data).hexdigest() for _, data in corpus)) != len(corpus):
        raise ValueError('duplicate image contents in corpus')
    config['corpus'] = [{'path': str(p.resolve()), 'sha256': hashlib.sha256(data).hexdigest()} for p, data in corpus]
    config['corpus_kind'] = 'single_image_repeated' if len(corpus) == 1 else 'multi_image'
    sources = args.images / 'SOURCES.json'
    if sources.is_file():
        config['corpus_sources'] = json.loads(sources.read_text())
        for item in config['corpus']:
            source = config['corpus_sources'].get(Path(item['path']).name)
            if source and source['sha256'] != item['sha256']:
                raise ValueError('image hash does not match SOURCES.json')
    config['host'] = {'uname': list(platform.uname()), 'cpu_count': os.cpu_count(),
                      'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), None)}
    config['sysfs'] = {str(p): p.read_text().strip() for pattern in
                       ('/sys/class/accel/accel*/device/fw_version', '/sys/class/drm/card*/device/power_dpm_force_performance_level')
                       for p in Path('/').glob(pattern.lstrip('/'))}
    config['source_revisions'] = {}
    for name, directory in (('benchmark', Path(__file__).parent), ('scrfd', args.scrfd)):
        revision = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True, capture_output=True)
        status = subprocess.run(['git', '-C', str(directory), 'status', '--porcelain'], text=True, capture_output=True)
        config['source_revisions'][name] = {'head': revision.stdout.strip(), 'status': status.stdout.strip()}
    config['script_sha256'] = digest(__file__)
    config['code_sha256'] = {name: digest(Path(__file__).with_name(name))
                             for name in ('bench.py', 'preprocess.py', 'npu_environment.py')}
    config['scrfd_artifacts_sha256'] = {str(p.relative_to(args.scrfd)): digest(p) for p in (args.scrfd / 'build').rglob('*')
        if p.is_file() and (p.name == 'libscrfd.so' or p.parent.name in ('weights', 'kernels'))}
    config['model_sha256'] = {'npu': digest(args.npu_model),
                              'gpu': checkpoint_hashes(args.gpu_model)}
    for name, shape, dtype in [('pixels', (args.depth, 3, 224, 224), np.float32),
                                ('canvas', (args.depth, 640, 640, 3), np.uint8),
                                ('mask', (args.depth, 196), np.float32),
                                ('gpu_result', (args.depth, 2, 1024), np.float32),
                                ('reference_result', (args.depth, 2, 1024), np.float32),
                                ('npu_result', (args.depth, 2, 1024), np.float32)]:
        array = np.lib.format.open_memmap(args.output / f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
        array[:] = 0
        array.flush()
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    arrays = {name: np.load(args.output / f'{name}.npy', mmap_mode='r+') for name in ('pixels', 'canvas', 'mask', 'gpu_result', 'npu_result', 'reference_result')}
    sys.path.insert(0, str(args.scrfd.resolve()))
    import cv2
    cv2.setNumThreads(1)
    from scrfd_loom import letterbox_into
    from preprocess import dino_input
    from npu_environment import environment as npu_environment_for
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS=str(args.cpu_threads), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    npu_environment = npu_environment_for(environment, args.npu_python, args.output,
                                          args.xrt_root, args.npu_lib_dir)
    config['npu_environment'] = {key: npu_environment[key] for key in ('XILINX_XRT', 'LD_LIBRARY_PATH')}
    config['preprocessing'] = 'centered bicubic thumbnail 224, no upscale, RGB ImageNet normalization, patch-center mask'
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    workers = {}
    try:
        for role in ('face', 'gpu', 'npu', 'reference'):
            print(f'Loading {role} worker...', flush=True)
            workers[role] = Worker(config, role, npu_environment if role == 'npu' else environment)
            print(json.dumps(workers[role].metadata), flush=True)
        config['workers'] = {role: worker.metadata for role, worker in workers.items()}
        (args.output / 'config.json').write_text(json.dumps(config, indent=2))

        def prepare(index, slot):
            begin = time.perf_counter()
            image = Image.open(io.BytesIO(corpus[index % len(corpus)][1])).convert('RGB')
            bgr = np.asarray(image)[:, :, ::-1].copy()
            scale = letterbox_into(bgr, arrays['canvas'][slot])
            arrays['pixels'][slot], arrays['mask'][slot] = dino_input(image)
            if arrays['mask'][slot].sum() == 0:
                raise ValueError('image has no non-padding DINO patches')
            return {'id': index, 'slot': slot, 'scale': scale, 'shape': list(bgr.shape[:2]),
                    'preprocess_ms': (time.perf_counter() - begin) * 1000}

        # Reference every corpus item through both backends before timing.
        references = []
        quality = []
        for index in range(len(corpus)):
            request = prepare(index, 0)
            face = workers['face'].call(request)
            workers['gpu'].call(request)
            gpu = arrays['gpu_result'][0].copy()
            workers['npu'].call(request)
            npu = arrays['npu_result'][0].copy()
            cosine = (gpu * npu).sum(1) / (np.linalg.norm(gpu, axis=1) * np.linalg.norm(npu, axis=1))
            if not np.isfinite(cosine).all() or cosine.min() < args.min_cosine:
                raise RuntimeError(f'quality failed for {corpus[index][0]}: {cosine.tolist()}')
            references.append({'face': face, 'gpu': gpu, 'npu': npu})
            reference_quality = {}
            workers['reference'].call(request)
            f32 = arrays['reference_result'][0].copy()
            for backend, actual in (('gpu', gpu), ('npu', npu)):
                values = (actual * f32).sum(1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(f32, axis=1))
                if not np.isfinite(values).all() or values.min() < args.min_cosine:
                    raise RuntimeError(f'{backend} failed float32 CPU reference: {values.tolist()}')
                reference_quality[backend + '_vs_f32'] = values.tolist()
            quality.append({**reference_quality, 'image': str(corpus[index][0]), 'cls_cosine': float(cosine[0]), 'pooled_cosine': float(cosine[1]), 'faces': len(face['boxes'])})
        (args.output / 'quality.json').write_text(json.dumps(quality, indent=2))
        workers.pop('reference').close()
        print('Quality passed against float32 CPU for every image', flush=True)
        rows = []
        modes = tuple(dict.fromkeys(args.modes))
        with (futures.ThreadPoolExecutor(max_workers=args.depth * 2) as pool,
              futures.ThreadPoolExecutor(max_workers=args.preprocess_workers) as preprocess_pool):
            for round_index in range(args.rounds):
                offset = round_index % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    count = len(corpus) * args.repeats
                    done, elapsed, cpu_total = run_mode(mode, count, args.depth, args.gpu_batch,
                        prepare, workers, arrays, pool, preprocess_pool)
                    # Validate all outputs after stopping the timer; count both branches.
                    for result in done:
                        if result['face']['id'] != result['id'] or result['embedding']['id'] != result['id']:
                            raise RuntimeError('mismatched per-image branch results')
                        reference = references[result['id'] % len(corpus)]
                        if not np.allclose(result['features'], reference[result['backend']], rtol=1e-3, atol=1e-3):
                            raise RuntimeError('embedding changed or wrong image/slot returned')
                        for key in ('boxes', 'landmarks'):
                            value, expected = np.asarray(result['face'][key]), np.asarray(reference['face'][key])
                            if value.shape != expected.shape or not np.allclose(value, expected, rtol=1e-4, atol=1e-3):
                                raise RuntimeError('face output changed or wrong image/slot returned')
                        del result['features']
                    if sorted(r['id'] for r in done) != list(range(count)):
                        raise RuntimeError('lost or duplicate image results')
                    row = {'round': round_index, 'mode': mode, 'images': count, 'seconds': elapsed,
                           'images_per_second': count / elapsed,
                           'latency_p50_ms': float(np.median([r['latency_ms'] for r in done])),
                           'latency_p95_ms': float(np.percentile([r['latency_ms'] for r in done], 95)),
                           'cpu_core_equivalents': cpu_total / elapsed,
                           'mean_preprocess_ms': float(np.mean([r['preprocess_ms'] for r in done])),
                           'mean_face_service_ms_per_image': float(np.mean([r['face']['service_ms'] for r in done])),
                           'mean_embedding_service_ms_per_image': {backend: float(np.mean([r['embedding']['service_ms'] for r in done if r['backend'] == backend])) for backend in ('gpu', 'npu') if any(r['backend'] == backend for r in done)},
                           'faces': sum(len(r['face']['boxes']) for r in done),
                           'embedding_images': {backend: sum(r['backend'] == backend for r in done) for backend in ('gpu', 'npu')}}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    with (args.output / 'measurements.jsonl').open('a') as stream:
                        emit(stream, row)
                    (args.output / f'round-{round_index}-{mode}.json').write_text(json.dumps(done))
        medians = {mode: float(np.median([r['images_per_second'] for r in rows if r['mode'] == mode])) for mode in modes}
        gpu_values = [value for mode, value in medians.items() if mode.startswith('gpu_')]
        best_gpu = max(gpu_values) if gpu_values else None
        summary = {'median_images_per_second': medians,
                   'corpus_kind': config['corpus_kind'], 'unique_images': len(corpus),
                   'images_with_float32_cpu_reference': len(quality),
                   'mixed_speedup_over_best_gpu': {mode: medians[mode] / best_gpu for mode in ('mixed_parallel', 'mixed_balanced') if mode in medians and best_gpu},
                   'min_cls_cosine': min(r['cls_cosine'] for r in quality),
                   'min_pooled_cosine': min(r['pooled_cosine'] for r in quality),
                   'timing': 'cached compressed images; includes decode, preprocess, IPC, copies, models, face decode/NMS, pooling and result delivery; excludes setup and validation',
                   'scope': 'application placement using existing model runtimes; not an hrx-native or zero-copy benchmark'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        for item in workers.values():
            item.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
