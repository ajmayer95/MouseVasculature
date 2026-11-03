import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from mousevasculature.viewer import load_mask


class MaskTests(unittest.TestCase):
    def test_calibration_and_binary_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mask.tif'
            data = np.zeros((4, 5, 6), dtype=np.uint16)
            data[1, 2, 3] = 255
            tifffile.imwrite(path, data, imagej=True, resolution=(2, 4),
                             metadata={'axes': 'ZYX', 'spacing': 3, 'unit': 'um'})
            mask, scale = load_mask(path)
            self.assertEqual(scale, (3, .25, .5))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(mask.sum(), 1)

    def test_missing_calibration_requires_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mask.tif'
            tifffile.imwrite(path, np.zeros((4, 5, 6), dtype=np.uint8), metadata={'axes': 'ZYX'})
            with self.assertRaises(ValueError):
                load_mask(path)
            self.assertEqual(load_mask(path, (1, 2, 3))[1], (1, 2, 3))
