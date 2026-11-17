import unittest

from mousevasculature.loop_topology import multigraph, shortest_cycle


def sizes(edges):
    g = multigraph(edges)
    return {k: shortest_cycle(g, u, v, k) for u, v, k, _ in edges}


class ShortestCycleTests(unittest.TestCase):
    def test_triangle_and_square(self):
        for n in (3, 4):
            result = sizes([(i, (i + 1) % n, i, 10.) for i in range(n)])
            self.assertTrue(all(r['branches'] == n and r['length_um'] == 10. * n for r in result.values()))

    def test_dangling_branch_is_bridge(self):
        result = sizes([(0, 1, 0, 1.), (1, 2, 1, 1.), (2, 3, 2, 1.), (3, 0, 3, 1.), (0, 9, 4, 5.)])
        self.assertIsNone(result[4])
        self.assertEqual(result[0]['branches'], 4)

    def test_parallel_branches_are_kept(self):
        # Two branches join nodes 0 and 1 directly; a long triangle also passes through them.
        result = sizes([(0, 1, 0, 3.), (0, 1, 1, 4.), (1, 2, 2, 50.), (2, 0, 3, 50.)])
        self.assertEqual((result[0]['branches'], result[0]['length_um']), (2, 7.))
        self.assertEqual(result[1]['members'], [1, 0])
        self.assertEqual(result[2]['branches'], 3)  # through the shorter parallel branch
        self.assertEqual(result[2]['length_um'], 103.)

    def test_self_loop(self):
        result = sizes([(0, 0, 0, 12.), (0, 1, 1, 3.)])
        self.assertEqual(result[0]['branches'], 1)
        self.assertIsNone(result[1])

    def test_length_weighting_differs_from_fewest_branches(self):
        # Branch 0 joins 0-1. Detour A: 2 long branches (200 µm); detour B: 4 short ones (4 µm).
        edges = [(0, 1, 0, 1.), (0, 2, 1, 100.), (2, 1, 2, 100.),
                 (0, 3, 3, 1.), (3, 4, 4, 1.), (4, 5, 5, 1.), (5, 1, 6, 1.)]
        r = sizes(edges)[0]
        self.assertEqual((r['branches'], r['length_um']), (5, 5.))
        self.assertEqual((r['branches_hops'], r['length_hops_um']), (3, 201.))

    def test_graph_is_restored(self):
        g = multigraph([(0, 1, 0, 1.), (1, 2, 1, 1.), (2, 0, 2, 1.)])
        shortest_cycle(g, 0, 1, 0)
        self.assertEqual(sorted(g.edges(keys=True)), [(0, 1, 0), (0, 2, 2), (1, 2, 1)])


class ContractTests(unittest.TestCase):
    def test_degree_two_nodes_are_merged(self):
        from mousevasculature.loop_topology import contract_degree_two
        # Square 0-1-2-3 with tails at 0 and 2: nodes 1 and 3 have degree 2, so the square
        # becomes two parallel 0-2 branches (3 and 2 µm) and the tails stay.
        g = contract_degree_two(multigraph([(0, 1, 0, 1.), (1, 2, 1, 2.), (2, 3, 2, 1.), (3, 0, 3, 1.), (0, 9, 4, 1.), (2, 8, 5, 1.)]))
        self.assertNotIn(1, g)
        self.assertNotIn(3, g)
        self.assertEqual(sorted(d['length_um'] for *_, d in g.edges(data=True)), [1., 1., 2., 3.])
        self.assertEqual(g.number_of_edges(0, 2), 2)
