import unittest

import numpy as np

from mousevasculature.tissue_domain import (local_thickness_radius, region_volumes, sum_of_squares_shares,
                                            void_components)


class LocalThicknessTests(unittest.TestCase):
    def test_gap_between_two_vessel_sheets(self):
        # Vessel sheets at x = 0 and x = 20 (1 µm voxels): the widest ball in the gap has radius 10.
        mask = np.zeros((30, 30, 21), bool)
        mask[:, :, 0] = mask[:, :, 20] = True
        radius = local_thickness_radius(mask, (1., 1., 1.), step=1., max_radius=50.)
        self.assertTrue(np.all(radius[:, :, 1:20] == 10))
        self.assertTrue(np.all(radius[mask] == 0))

    def test_face_assumption_bounds_voids(self):
        mask = np.zeros((21, 21, 21), bool)
        mask[10, 10, 10] = True
        open_ = local_thickness_radius(mask, (1., 1., 1.), False, step=1., max_radius=50.)
        closed = local_thickness_radius(mask, (1., 1., 1.), True, step=1., max_radius=50.)
        self.assertLess(closed.max(), open_.max())
        self.assertTrue(np.all(closed <= open_))


class DomainTests(unittest.TestCase):
    def test_region_volumes_cover_centred_grid(self):
        domain = np.zeros((50, 50, 50), bool)
        domain[:25] = True  # lower half in z (10 µm voxels: 0-250 µm)
        dom, tot = region_volumes(domain, np.array([10., 10., 10.]))
        self.assertEqual(len(tot), 8)
        self.assertTrue(all(v == 24 ** 3 * 1000 for v in tot.values()))  # 240 µm regions from 10 µm
        self.assertEqual(dom[(0, 0, 0)], tot[(0, 0, 0)])
        self.assertEqual(dom[(1, 0, 0)], 0)

    def test_void_components_and_faces(self):
        void = np.zeros((10, 10, 10), bool)
        void[0:2, 0:2, 0:2] = True       # touches a face
        void[4:6, 4:6, 4:6] = True       # interior
        comps = sorted(void_components(void, (1., 1., 1.)), key=lambda c: c['touches_face'])
        self.assertEqual([c['touches_face'] for c in comps], [False, True])
        self.assertEqual([c['volume_um3'] for c in comps], [8., 8.])

    def test_sum_of_squares_shares(self):
        rows = [dict(cube=f'{a}_ROI{c}', v=v) for a, v in (('LS025', 1.), ('LS026', 3.)) for c in (1, 2) for _ in range(4)]
        shares = sum_of_squares_shares(rows, 'v')
        self.assertAlmostEqual(shares['animals'], 1.)
        self.assertAlmostEqual(shares['within_cube'], 0.)
