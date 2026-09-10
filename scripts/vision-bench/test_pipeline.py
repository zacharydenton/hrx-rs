"""CPU tests for slot ownership and image/result association under reordering."""
import concurrent.futures as futures
import threading
import time
import unittest

import numpy as np

from bench import run_mode


class FakeWorker:
    def __init__(self, role, inputs, arrays):
        self.role, self.inputs, self.arrays = role, inputs, arrays
        self.lock = threading.Lock()

    def call(self, packet):
        with self.lock:
            results = []
            for request in packet['batch']:
                time.sleep(0.0005 if self.role == 'npu' else 0.001)
                actual = self.inputs[request['slot']]
                if self.role != 'face':
                    self.arrays[self.role + '_result'][request['slot']] = actual
                results.append({'id': request['id'], 'observed': int(actual),
                                'cpu_seconds': 0, 'service_ms': 0})
            return {'id': packet['id'], 'batch': results}


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, mode, count, depth, batch):
        inputs = np.full(depth, -1, dtype=int)
        arrays = {role + '_result': np.full((depth, 2), -1, dtype=int) for role in ('gpu', 'npu')}
        workers = {role: FakeWorker(role, inputs, arrays) for role in ('face', 'gpu', 'npu')}

        def prepare(index, slot):
            # Out-of-order preprocessing and repeated slot reuse expose lifetime bugs.
            time.sleep(0.002 if index % 3 == 0 else 0.0001)
            inputs[slot] = index
            return {'id': index, 'slot': slot, 'preprocess_ms': 0}

        with (futures.ThreadPoolExecutor(max_workers=16) as pool,
              futures.ThreadPoolExecutor(max_workers=4) as preprocessing):
            results, elapsed, _ = run_mode(mode, count, depth, batch, prepare,
                                           workers, arrays, pool, preprocessing)
        self.assertGreater(elapsed, 0)
        self.assertEqual(sorted(item['id'] for item in results), list(range(count)))
        for item in results:
            self.assertEqual(item['face']['observed'], item['id'])
            self.assertEqual(item['embedding']['observed'], item['id'])
            np.testing.assert_array_equal(item['features'], [item['id']] * 2)
        if mode == 'mixed_balanced' and count > batch:
            self.assertEqual({item['backend'] for item in results}, {'gpu', 'npu'})

    def test_reordering_and_partial_batches(self):
        for mode in ('gpu_serial', 'gpu_parallel', 'mixed_parallel', 'mixed_balanced'):
            for count, depth, batch in ((37, 8, 4), (19, 3, 3), (7, 2, 1), (1, 4, 4)):
                with self.subTest(mode=mode, count=count, depth=depth, batch=batch):
                    self.run_pipeline(mode, count, depth, batch)

    def test_worker_failure_propagates(self):
        class BrokenWorker:
            def call(self, packet):
                raise RuntimeError('inference failed')

        with (futures.ThreadPoolExecutor(max_workers=2) as pool,
              futures.ThreadPoolExecutor(max_workers=1) as preprocessing):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                run_mode('gpu_serial', 1, 1, 1,
                         lambda index, slot: {'id': index, 'slot': slot, 'preprocess_ms': 0},
                         {'face': BrokenWorker(), 'gpu': BrokenWorker()}, {}, pool, preprocessing)


if __name__ == '__main__':
    unittest.main()
