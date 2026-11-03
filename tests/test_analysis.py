import unittest
import numpy as np
from mousevasculature.analysis import analyze, vectorize


class GraphTests(unittest.TestCase):
    def check_coverage(self, skeleton, graph):
        expected = set()
        points = set(map(tuple, np.argwhere(skeleton)))
        for p in points:
            for q in points:
                if p < q and max(abs(a-b) for a, b in zip(p, q)) <= 1:
                    # Diagonals with an intermediate voxel are redundant.
                    if any(r not in (p, q) and all(min(a,b) <= c <= max(a,b)
                           for a,b,c in zip(p,q,r)) for r in points):
                        continue
                    expected.add((p, q))
        actual = []
        for edge in graph['edges']:
            path = list(map(tuple, edge['points_zyx']))
            actual.extend(tuple(sorted((a, b))) for a, b in zip(path, path[1:]))
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))

    def test_line_and_physical_length(self):
        mask = np.zeros((9, 9, 9), bool)
        mask[2:7, 4, 4] = True
        skeleton, graph = analyze(mask, (2, 1, 1))
        self.assertEqual(len(graph['edges']), 1)
        self.assertEqual(len(graph['nodes']), 2)
        self.assertEqual(graph['edges'][0]['length_um'], 8)
        self.check_coverage(skeleton, graph)

    def test_loop_is_preserved(self):
        skeleton = np.zeros((3, 5, 5), bool)
        for y, x in [(1, 2), (2, 3), (3, 2), (2, 1)]:
            skeleton[1, y, x] = True
        graph = vectorize(skeleton, (1, 1, 1))
        self.assertEqual(len(graph['edges']), 1)
        self.assertEqual(graph['edges'][0]['source'], graph['edges'][0]['target'])
        self.check_coverage(skeleton, graph)

    def test_junction_disconnected_and_isolated(self):
        skeleton = np.zeros((12, 12, 12), bool)
        skeleton[2:8, 4, 4] = True
        skeleton[4, 4:8, 4] = True
        skeleton[10, 10, 10] = True
        graph = vectorize(skeleton, (1, 1, 1))
        self.assertTrue(any(n['kind'] == 'junction' for n in graph['nodes']))
        self.assertTrue(any(n['kind'] == 'isolated' for n in graph['nodes']))
        self.check_coverage(skeleton, graph)

    def test_empty(self):
        graph = vectorize(np.zeros((3, 3, 3), bool), (1, 1, 1))
        self.assertEqual(graph['nodes'], [])
        self.assertEqual(graph['edges'], [])
