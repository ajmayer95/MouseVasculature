import tempfile
import unittest
from pathlib import Path

import numpy as np

from mousevasculature.analysis import analyze, vectorize
import networkx as nx

from mousevasculature.statistics import (bifurcations, contract_junctions, loop_lengths, measure, minimum_loops,
                                        merge_degree_two, murray_exponent, save_statistics, tissue_distance)


class StatisticsTests(unittest.TestCase):
    def test_straight_tube_calibration_radius_and_topology(self):
        mask = np.zeros((15, 15, 15), dtype=bool)
        yy, xx = np.ogrid[:15, :15]
        mask[2:13, :, :] = ((yy - 7)**2 + (xx - 7)**2 <= 9)
        skeleton = np.zeros_like(mask)
        skeleton[3:12, 7, 7] = True
        graph = vectorize(skeleton, (2, 1, 1))
        summary, branches, _, _ = measure(mask, skeleton, graph)
        self.assertEqual(summary['components'], 1)
        self.assertEqual(summary['endpoints'], 2)
        self.assertEqual(summary['independent_cycles'], 0)
        self.assertEqual(summary['bridges'], 1)
        self.assertEqual(branches[0]['length_um'], 16)
        self.assertEqual(branches[0]['tortuosity'], 1)
        # Center-to-background EDT on this digital circle is sqrt(10).
        self.assertAlmostEqual(branches[0]['radius_mean_um'], np.sqrt(10))
        self.assertFalse(branches[0]['touches_boundary'])

    def test_bent_path_tortuosity(self):
        skeleton = np.zeros((12, 12, 12), dtype=bool)
        skeleton[2:7, 3, 3] = True
        skeleton[6, 3:8, 3] = True
        _, rows, _, _ = measure(skeleton, skeleton, vectorize(skeleton, (1, 1, 1)))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]['tortuosity'], np.sqrt(2))

    def test_loop_and_disconnected_isolate(self):
        skeleton = np.zeros((7, 7, 7), dtype=bool)
        for y, x in [(1, 2), (2, 3), (3, 2), (2, 1)]:
            skeleton[1, y, x] = True
        skeleton[5, 5, 5] = True
        summary, rows, _, _ = measure(skeleton, skeleton, vectorize(skeleton, (1, 1, 1)))
        self.assertEqual(summary['components'], 2)
        self.assertEqual(summary['independent_cycles'], 1)
        self.assertEqual(summary['bridges'], 0)
        self.assertIsNone(rows[0]['tortuosity'])
        self.assertEqual(summary['loop_length_um']['n'], 1)
        self.assertAlmostEqual(summary['loop_length_um']['median'], 4 * np.sqrt(2))
        self.assertEqual(summary['looped_length_fraction'], 1)

    def test_junction_contraction_keeps_parallel_paths(self):
        nodes = [dict(id=i, point_zyx=[i, 0, 0], kind='junction') for i in range(4)]
        def edge(i, a, b, pts):
            return dict(id=i, source=a, target=b, points_zyx=pts, length_um=1.)
        graph = dict(nodes=nodes, voxel_size_um=[1, 1, 1], edges=[
            edge(0, 0, 1, [[0,0,0], [1,0,0]]),
            edge(1, 2, 3, [[2,0,0], [3,0,0]]),
            edge(2, 1, 2, [[1,0,0], [1,1,0], [2,0,0]]),
            edge(3, 0, 3, [[0,0,0], [0,1,0], [3,0,0]])])
        result = contract_junctions(graph)
        self.assertEqual(len(result['nodes']), 2)
        self.assertEqual(len(result['edges']), 2)
        self.assertEqual(result['absorbed_junction_edge_count'], 2)
        self.assertEqual(result['edges'][0]['source'], result['edges'][1]['source'])
        self.assertEqual(result['edges'][0]['target'], result['edges'][1]['target'])
        closed = merge_degree_two(result, graph)
        self.assertEqual(len(closed['edges']), 1)
        self.assertEqual(closed['edges'][0]['source'], closed['edges'][0]['target'])
        self.assertEqual(closed['edges'][0]['points_zyx'][0], closed['edges'][0]['points_zyx'][-1])

    def test_empty_exports_valid_json_and_headers(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        skeleton, graph = analyze(mask, (1, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            summary, _ = save_statistics(tmp, 'empty.tif', mask, skeleton, graph, plots=False)
            self.assertEqual(summary['branches'], 0)
            self.assertEqual(summary['length_um']['n'], 0)
            self.assertTrue((Path(tmp) / 'branches.csv').read_text().startswith('branch_id,'))

    def test_boundary_censoring(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[:, 2, 2] = True
        _, branches, _, _ = measure(mask, mask, vectorize(mask, (1, 1, 1)))
        self.assertTrue(branches[0]['touches_boundary'])

    def test_merge_degree_two_uses_voxel_route(self):
        raw_nodes = [dict(id=i, point_zyx=p, kind=k) for i, p, k in [
            (0, [1,1,1], 'endpoint'), (1, [2,1,1], 'junction'),
            (2, [2,2,1], 'junction'), (3, [3,2,1], 'endpoint')]]
        edges = [dict(id=i, source=a, target=b, points_zyx=[raw_nodes[a]['point_zyx'], raw_nodes[b]['point_zyx']], length_um=1.)
                 for i, (a,b) in enumerate([(0,1), (1,2), (2,3)])]
        raw = dict(nodes=raw_nodes, edges=edges, voxel_size_um=[1,1,1])
        graph = merge_degree_two(contract_junctions(raw), raw)
        self.assertEqual(len(graph['edges']), 1)
        self.assertEqual(graph['edges'][0]['length_um'], 3)
        self.assertEqual(graph['edges'][0]['points_zyx'], [n['point_zyx'] for n in raw_nodes])
        self.assertEqual(graph['junction_connector_length_um'], 1)


class ExtendedMetricTests(unittest.TestCase):
    def test_loop_basis_counts_parallel_branches_and_self_loops(self):
        g = nx.MultiGraph()
        g.add_edge(0, 1, key=0, length_um=2.)
        g.add_edge(0, 1, key=1, length_um=3.)
        g.add_edge(1, 1, key=2, length_um=5.)
        g.add_edge(1, 2, key=3, length_um=7.)
        lengths = loop_lengths(g)
        self.assertEqual(len(lengths), g.number_of_edges() - g.number_of_nodes() + 1)
        np.testing.assert_allclose(lengths, [5, 5])

    def test_loop_basis_is_minimal_on_grid_with_diagonal(self):
        g = nx.MultiGraph()
        for i, (a, b) in enumerate(nx.grid_2d_graph(3, 3).edges()):
            g.add_edge(a, b, key=i, length_um=1.)
        g.add_edge((0, 0), (2, 2), key=99, length_um=10.)  # long chord; still one extra independent cycle
        lengths = loop_lengths(g)
        np.testing.assert_allclose(lengths, [4, 4, 4, 4, 14])

    def test_loop_membership_and_kinds(self):
        g = nx.MultiGraph()
        g.add_edge(0, 1, key=0, length_um=2.)
        g.add_edge(0, 1, key=1, length_um=3.)
        g.add_edge(1, 1, key=2, length_um=6.)
        for key, (a, b) in enumerate([(1, 2), (2, 3), (3, 1)], start=3):
            g.add_edge(a, b, key=key, length_um=10.)
        loops = minimum_loops(g)
        self.assertEqual([(l['kind'], l['branch_ids'], l['length_um']) for l in loops],
                         [('parallel', [0, 1], 5.), ('self_loop', [2], 6.), ('cycle', [3, 4, 5], 30.)])
        self.assertEqual([l['loop_id'] for l in loops], [0, 1, 2])

    def test_murray_exponent(self):
        self.assertAlmostEqual(murray_exponent(2 ** (1 / 3), 1, 1), 3, places=6)
        self.assertIsNone(murray_exponent(1, 1, .5))

    def test_bifurcation_ratio_and_angle(self):
        def edge(i, points):
            return dict(id=i, source=0, target=i + 1, points_zyx=points)
        graph = dict(edges=[edge(0, [[10, 10, 10], [0, 10, 10]]),
                            edge(1, [[10, 10, 10], [10, 30, 10]]),
                            edge(2, [[10, 10, 10], [10, 10, 30]])])
        rows = [dict(branch_id=i, radius_mean_um=r, touches_boundary=False, short_branch=False)
                for i, r in enumerate([2 ** (1 / 3) * 4, 4, 4])]
        g = nx.MultiGraph()
        for e in graph['edges']:
            g.add_edge(e['source'], e['target'], key=e['id'])
        (b,) = bifurcations(g, graph, rows, np.ones(3), {n: 0 for n in g})
        self.assertEqual(b['parent_branch'], 0)
        self.assertAlmostEqual(b['murray_ratio'], 1)
        self.assertAlmostEqual(b['branching_angle_deg'], 90)
        self.assertTrue(b['resolved'])

    def test_tissue_distance_exact_inside_crop(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = False
        stats, exact = tissue_distance(mask, np.array([2., 1., 1.]))
        self.assertEqual(stats['n'], 1)
        self.assertEqual(stats['mean'], 1)
        self.assertEqual(exact, 1)
