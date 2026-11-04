import unittest

import numpy as np
from skimage.measure import euler_number

from mousevasculature.preprocess import close_tunnels


def tunnels(mask):
    from scipy import ndimage as ndi
    _, components = ndi.label(mask, structure=np.ones((3, 3, 3)))
    return components - euler_number(mask, connectivity=3)


class CloseTunnelTests(unittest.TestCase):
    def test_plugs_pinhole_through_thick_rod(self):
        mask = np.zeros((20, 11, 11), dtype=bool)
        mask[2:18, 3:8, 3:8] = True
        mask[10, 5, 3:8] = False  # one-voxel hole drilled through the rod
        self.assertEqual(tunnels(mask), 1)
        repaired, report = close_tunnels(mask, 1)
        self.assertEqual(tunnels(repaired), 0)
        self.assertEqual(report['accepted_patches'], 1)
        self.assertTrue(np.all(repaired[mask]))  # never removes foreground
        # Interior hole voxels are plugged; the two surface openings remain shallow dimples.
        self.assertEqual(int(repaired.sum() - mask.sum()), 3)

    def test_does_not_bridge_separate_vessels(self):
        mask = np.zeros((20, 9, 12), dtype=bool)
        mask[2:18, 3:6, 2:5] = True
        mask[2:18, 3:6, 6:9] = True  # one-voxel gap
        repaired, report = close_tunnels(mask, 1)
        np.testing.assert_array_equal(repaired, mask)
        self.assertGreater(report['rejected_bridging_patches'], 0)

    def test_does_not_close_gap_that_would_create_a_loop(self):
        mask = np.zeros((20, 9, 12), dtype=bool)
        mask[2:18, 3:6, 2:5] = True
        mask[2:18, 3:6, 6:9] = True
        mask[2:4, 3:6, 2:9] = True  # rods joined at one end: a narrow U
        repaired, _ = close_tunnels(mask, 1)
        self.assertEqual(tunnels(repaired), 0)
        np.testing.assert_array_equal(repaired, mask)

    def test_keeps_large_ring(self):
        mask = np.zeros((5, 25, 25), dtype=bool)
        yy, xx = np.ogrid[:25, :25]
        r = np.hypot(yy - 12, xx - 12)
        mask[1:4] = (r >= 7) & (r <= 9)
        repaired, _ = close_tunnels(mask, 1)
        self.assertEqual(tunnels(mask), 1)
        self.assertEqual(tunnels(repaired), 1)
