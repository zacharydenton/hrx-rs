"""Pipe framing and checkpoint provenance tests, without model dependencies."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bench import Worker, checkpoint_hashes, digest


class TransportTests(unittest.TestCase):
    def reader(self, payload):
        reader, writer = os.pipe()
        os.write(writer, payload)
        os.close(writer)
        worker = Worker.__new__(Worker)
        worker.role = 'npu'
        worker.timeout = 0.1
        worker.log = SimpleNamespace(name='npu.log')
        worker.response = os.fdopen(reader, 'rb', buffering=0)
        worker.read_buffer = bytearray()
        self.addCleanup(worker.response.close)
        return worker

    def test_multiple_lines_in_one_pipe_read(self):
        worker = self.reader(b'{"id": 1}\n{"id": 2}\n')
        self.assertEqual(worker.read(), {'id': 1})
        self.assertEqual(worker.read(), {'id': 2})

    def test_truncated_response_points_to_worker_log(self):
        with self.assertRaisesRegex(RuntimeError, 'npu exited during a response; see npu.log'):
            self.reader(b'{"id":').read()

    def test_invalid_response_points_to_worker_log(self):
        with self.assertRaisesRegex(RuntimeError, 'npu returned an invalid response; see npu.log'):
            self.reader(b'{broken}\n').read()

    def test_hashes_all_shards_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = {'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors', 'c': 'part1.safetensors'}}
            (root / 'model.safetensors.index.json').write_text(json.dumps(index))
            for name in ('config.json', 'part1.safetensors', 'part2.safetensors'):
                (root / name).write_text(name)
            hashes = checkpoint_hashes(root)
            self.assertEqual(set(hashes), {p.name for p in root.iterdir()})
            for name, sha in hashes.items():
                self.assertEqual(sha, digest(root / name))
            (root / 'part2.safetensors').write_text('changed')
            self.assertNotEqual(hashes['part2.safetensors'], checkpoint_hashes(root)['part2.safetensors'])


if __name__ == '__main__':
    unittest.main()
