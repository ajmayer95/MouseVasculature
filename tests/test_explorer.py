import unittest

import numpy as np

from mousevasculature.explorer import Network, branch_labels


def edge(i, a, b, points, length):
    return dict(id=i, source=a, target=b, points_zyx=points, length_um=length,
                statistics=dict(component_id=0, length_um=length, chord_um=length, tortuosity=1.,
                                radius_mean_um=2., radius_min_um=1., radius_max_um=3.,
                                bridge=False, touches_boundary=False, short_branch=False, self_loop=a == b))


def square_with_tail():
    # Nodes 0-1-2-3 form a square (branches 0-3); branch 4 is a tail from node 0 to 4;
    # branch 5 is a long parallel to branch 0.
    edges = [edge(0, 0, 1, [[0, 0, 0], [0, 0, 1]], 1.), edge(1, 1, 2, [[0, 0, 1], [0, 1, 1]], 1.),
             edge(2, 2, 3, [[0, 1, 1], [0, 1, 0]], 1.), edge(3, 3, 0, [[0, 1, 0], [0, 0, 0]], 1.),
             edge(4, 0, 4, [[0, 0, 0], [1, 0, 0]], 1.), edge(5, 0, 1, [[0, 0, 0], [0, 0, 1]], 10.)]
    edges[4]['statistics']['bridge'] = True
    graph = dict(nodes=[dict(id=i, point_zyx=[0, 0, 0]) for i in range(5)], edges=edges,
                 loops=[dict(loop_id=0, branch_ids=[0, 1, 2, 3], length_um=4., kind='cycle', small=True,
                             extent_um=1., min_radius_um=1., touches_boundary=False, has_short_branch=False),
                        dict(loop_id=1, branch_ids=[0, 5], length_um=11., kind='parallel', small=True,
                             extent_um=1., min_radius_um=1., touches_boundary=False, has_short_branch=False)],
                 bifurcations=[])
    return graph


class ExplorerTests(unittest.TestCase):
    def test_shortest_loop_path_and_component(self):
        net = Network(square_with_tail())
        self.assertEqual(sorted(net.shortest_loop(1)), [0, 1, 2, 3])
        self.assertEqual(sorted(net.shortest_loop(5)), [0, 5])  # parallel pair is shortest
        self.assertIsNone(net.shortest_loop(4))  # bridge
        self.assertEqual(net.path_between(4, 2), [4, 3, 2])
        self.assertEqual(net.component(4), [0, 1, 2, 3, 4, 5])
        self.assertEqual([l['loop_id'] for l in net.loops_of[0]], [0, 1])
        self.assertIn('bridge', net.describe_branch(4))
        self.assertIn('shortest loop through it: 4 µm', net.describe_branch(1))

    def test_branch_labels_follow_nearest_centerline(self):
        mask = np.zeros((3, 3, 9), dtype=bool)
        mask[:, :, 0:4] = True
        mask[:, :, 5:9] = True  # separate component with no centerline
        graph = dict(edges=[dict(id=0, points_zyx=[[1, 1, 0], [1, 1, 1]]),
                            dict(id=1, points_zyx=[[1, 1, 2], [1, 1, 3]])])
        labels = branch_labels(mask, graph, np.ones(3))
        self.assertEqual(labels[0, 0, 0], 1)
        self.assertEqual(labels[2, 2, 3], 2)
        self.assertTrue(np.all(labels[:, :, 5:] == 0))
        self.assertTrue(np.all(labels[~mask] == 0))


class SurfaceAndCacheTests(unittest.TestCase):
    def test_branch_colours_gradient_and_missing_values(self):
        from mousevasculature.explorer import branch_colours, NO_VALUE
        net = Network(square_with_tail())
        net.stats[0]['radius_mean_um'] = 5.
        colours, legend = branch_colours(net, 'radius (µm)')
        self.assertEqual(legend['kind'], 'gradient')
        self.assertFalse(np.allclose(colours[0], colours[1]))
        colours, legend = branch_colours(net, 'smallest loop perimeter (µm)')
        np.testing.assert_allclose(colours[4], NO_VALUE)  # bridge has no loop
        colours, _ = branch_colours(net, 'loop / bridge / crop')
        self.assertFalse(np.allclose(colours[4], colours[0]))

    def test_surface_vertices_map_to_branches(self):
        from mousevasculature.explorer import vessel_surface
        mask = np.zeros((12, 12, 30), dtype=bool)
        mask[4:8, 4:8, 2:28] = True
        graph = dict(edges=[dict(id=0, points_zyx=[[6, 6, x] for x in range(3, 15)]),
                            dict(id=1, points_zyx=[[6, 6, x] for x in range(15, 27)])])
        vertices, faces, branch = vessel_surface(mask, graph, np.ones(3))
        self.assertTrue(len(faces) > 0 and set(branch) == {0, 1})
        self.assertTrue(np.all(branch[vertices[:, 2] < 10] == 0))
        self.assertTrue(np.all(branch[vertices[:, 2] > 20] == 1))

    def test_results_are_reused_only_when_current(self):
        import json, tempfile
        from pathlib import Path
        from mousevasculature.pipeline import is_current
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(is_current(folder))
            (folder / 'statistics.json').write_text(json.dumps(dict(schema_version=4, analysis_variant='filled+closed',
                                                                     voxel_size_um=[2., 1.8, 1.8])))
            for f in ('branch_graph.json', 'skeleton.npz'):
                (folder / f).write_text('')
            self.assertFalse(is_current(folder))  # repaired mask missing
            (folder / 'analysis_mask.npz').write_text('')
            self.assertTrue(is_current(folder))
            self.assertTrue(is_current(folder, (2., 1.8, 1.8)))
            self.assertFalse(is_current(folder, (1.8, 1.8, 1.8)))  # calibration changed
            (folder / 'statistics.json').write_text(json.dumps(dict(schema_version=2, analysis_variant='filled+closed')))
            self.assertFalse(is_current(folder))


class StatsPanelTests(unittest.TestCase):
    def test_metrics_skip_missing_values_and_weighted_median(self):
        from mousevasculature.stats_panel import METRICS, rank_text, weighted_median
        graph = square_with_tail()
        graph['edges'][2]['statistics']['tortuosity'] = None  # e.g. a self-loop has no chord
        values, weights = METRICS['vessel diameter (µm, length-weighted)'][0](graph)
        np.testing.assert_allclose(values, 4.)
        self.assertEqual(len(weights), 6)
        self.assertEqual(len(METRICS['tortuosity (length / chord)'][0](graph)[0]), 5)
        self.assertEqual(len(METRICS['loop perimeter (µm)'][0](graph)[0]), 2)
        self.assertEqual(weighted_median(np.array([1., 2., 3.]), np.array([1., 1., 10.])), 3.)
        self.assertEqual(rank_text(5, [1, 5, 9]), '2 of 3')
