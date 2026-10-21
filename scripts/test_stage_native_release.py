"""Regression checks for ordered native patch verification."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'stage_native_release', Path(__file__).with_name('stage-native-release.py'))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class OrderedPatches(unittest.TestCase):
    def test_overlapping_stack_and_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'upstream.tar.gz'
            with tarfile.open(archive, 'w:gz') as output:
                member = tarfile.TarInfo('upstream/source.txt')
                contents = b'base\n'
                member.size = len(contents)
                output.addfile(member, io.BytesIO(contents))
            patches = ['first.patch', 'second.patch']
            for name, before, after in [
                (patches[0], 'base', 'first'),
                (patches[1], 'first', 'second'),
            ]:
                (root / name).write_text(
                    '--- a/source.txt\n+++ b/source.txt\n@@ -1 +1 @@\n'
                    f'-{before}\n+{after}\n')
            source = root / 'source'
            source.mkdir()
            target = source / 'source.txt'
            target.write_text('second\n')
            previous = stage.REPO
            stage.REPO = root
            try:
                stage.verify_patched_sources(source, archive, patches)
                target.write_text('unreviewed change\n')
                with self.assertRaisesRegex(ValueError, 'Source differs'):
                    stage.verify_patched_sources(source, archive, patches)
                target.unlink()
                with self.assertRaisesRegex(ValueError, 'Source differs'):
                    stage.verify_patched_sources(source, archive, patches)
            finally:
                stage.REPO = previous


if __name__ == '__main__':
    unittest.main()
