import unittest

import numpy as np

from mousevasculature.entanglement import cycle_polyline, find_links, linking_number
from mousevasculature.explorer import Network


def ring(centre, normal_axis, radius=10, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = radius * np.cos(t), radius * np.sin(t)
    axes = [a for a in range(3) if a != normal_axis]
    points = np.zeros((n, 3))
    points[:, axes[0]], points[:, axes[1]] = c, s
    return points + centre


def as_graph(rings):
    """Each ring becomes a self-loop branch on its own node."""
    edges, nodes = [], []
    for i, points in enumerate(rings):
        closed = np.vstack([points, points[:1]]).round(3).tolist()
        edges.append(dict(id=i, source=i, target=i, points_zyx=closed, length_um=1.,
                          statistics=dict(bridge=False, component_id=i)))
        nodes.append(dict(id=i, point_zyx=closed[0]))
    loops = [dict(loop_id=i, branch_ids=[i], length_um=1.) for i in range(len(rings))]
    return dict(nodes=nodes, edges=edges, loops=loops, bifurcations=[])


class LinkingTests(unittest.TestCase):
    def test_hopf_link_is_plus_or_minus_one(self):
        a = ring(np.zeros(3), 2)                 # circle in the plane of axes 0 and 1
        b = ring(np.array([10., 0., 0.]), 1)     # circle in the plane of axes 0 and 2, through a's centre
        self.assertAlmostEqual(abs(linking_number(a, b)), 1, places=6)

    def test_separate_and_coplanar_rings_are_unlinked(self):
        a = ring(np.zeros(3), 2)
        self.assertAlmostEqual(linking_number(a, ring(np.array([40., 0., 0.]), 1)), 0, places=6)  # apart
        self.assertAlmostEqual(linking_number(a, ring(np.array([25., 0., 0.]), 2)), 0, places=6)  # coplanar

    def test_double_winding(self):
        core = ring(np.zeros(3), 2, radius=10)
        u = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        # a coil that wraps twice around the core circle's tube
        coil = np.stack([(10 + 3 * np.cos(2 * u)) * np.cos(u), (10 + 3 * np.cos(2 * u)) * np.sin(u), 3 * np.sin(2 * u)], 1)
        self.assertAlmostEqual(abs(linking_number(coil, core)), 2, places=5)

    def test_find_links_on_graph(self):
        graph = as_graph([ring(np.array([50., 50., 50.]), 2), ring(np.array([60., 50., 50.]), 1),
                          ring(np.array([50., 50., 120.]), 2)])
        links, summary = find_links(graph, np.ones(3))
        self.assertEqual(summary['cycles'], 3)
        self.assertEqual(len(links), 1)
        self.assertEqual(sorted([links[0]['cycle_a'], links[0]['cycle_b']]), [[0], [1]])

    def test_cycle_polyline_orders_branches(self):
        edges = [dict(id=0, source=0, target=1, points_zyx=[[0, 0, 0], [0, 0, 5]], length_um=5.),
                 dict(id=1, source=2, target=1, points_zyx=[[0, 5, 5], [0, 0, 5]], length_um=5.),
                 dict(id=2, source=2, target=0, points_zyx=[[0, 5, 5], [0, 0, 0]], length_um=7.)]
        for e in edges:
            e['statistics'] = dict(bridge=False, component_id=0)
        net = Network(dict(nodes=[dict(id=i, point_zyx=[0, 0, 0]) for i in range(3)], edges=edges, loops=[]))
        points, nodes = cycle_polyline(net, [0, 1, 2])
        self.assertEqual(nodes, {0, 1, 2})
        self.assertEqual(points[:, 2].tolist(), [0, 5, 5, 5, 5, 0])


class ExhaustiveTests(unittest.TestCase):
    def test_finds_link_through_combined_cycle(self):
        # Ring 0 is linked with ring 1; ring 1 is split into two branches between nodes 1 and 2,
        # so the linked partner is a two-branch cycle rather than a single listed loop.
        from mousevasculature.entanglement import exhaustive_links
        a = ring(np.array([50., 50., 50.]), 2)
        b = ring(np.array([60., 50., 50.]), 1)
        half = len(b) // 2
        edges = [dict(id=0, source=0, target=0, points_zyx=np.vstack([a, a[:1]]).tolist(), length_um=63.),
                 dict(id=1, source=1, target=2, points_zyx=b[:half + 1].tolist(), length_um=31.),
                 dict(id=2, source=2, target=1, points_zyx=np.vstack([b[half:], b[:1]]).tolist(), length_um=31.)]
        for e in edges:
            e['statistics'] = dict(bridge=False, component_id=0)
        graph = dict(nodes=[dict(id=i, point_zyx=[0, 0, 0]) for i in range(3)], edges=edges,
                     loops=[dict(loop_id=0, branch_ids=[0], length_um=63.)], bifurcations=[])
        findings, summary = exhaustive_links(graph, np.ones(3))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['partner'], [1, 2])
        self.assertEqual(abs(findings[0]['linking_number']), 1)
