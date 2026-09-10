"""Known geometry and numeric checks for the benchmark-owned input contract."""
import unittest

import numpy as np
from PIL import Image

from preprocess import dino_input, pool_hidden


class PreprocessTests(unittest.TestCase):
    def test_square_normalization(self):
        image = Image.new('RGB', (512, 512), (255, 0, 0))
        pixels, mask = dino_input(image)
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(pixels.shape, (3, 224, 224))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(pixels.flags.c_contiguous)
        np.testing.assert_allclose(pixels[:, 100, 100],
                                   [(1 - .485) / .229, -.456 / .224, -.406 / .225], rtol=1e-6)
        self.assertEqual(mask.sum(), 196)

    def test_letterbox_masks_and_no_upscale(self):
        for dimensions, patches in (((448, 224), 98), ((224, 448), 98), ((64, 32), 8)):
            with self.subTest(dimensions=dimensions):
                _, mask = dino_input(Image.new('RGB', dimensions))
                self.assertEqual(mask.sum(), patches)
        with self.assertRaisesRegex(ValueError, 'no DINO patch centers'):
            dino_input(Image.new('RGB', (1, 1)))

    def test_pool_ignores_registers_and_padding(self):
        hidden = np.full((201, 1024), -100, dtype=np.float32)
        hidden[0] = 7
        hidden[5] = 2
        hidden[7] = 4
        mask = np.zeros(196, dtype=np.float32)
        mask[[0, 2]] = 1
        result = pool_hidden(hidden, mask)
        np.testing.assert_array_equal(result[0], np.full(1024, 7))
        np.testing.assert_array_equal(result[1], np.full(1024, 3))
        with self.assertRaisesRegex(ValueError, 'expected DINOv3-L'):
            pool_hidden(hidden[:200], mask)


if __name__ == '__main__':
    unittest.main()
