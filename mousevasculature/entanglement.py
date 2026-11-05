"""Topological entanglement of vessel loops: Gauss linking numbers between disjoint cycles.

Two loops that share no junction can still pass through each other like chain links.
The Gauss linking number of two closed curves counts this: 0 when they can be pulled
apart without crossing, ±1 for a single chain link, more for repeated winding. For
closed polygons it is computed exactly as a sum of signed solid angles over all segment
pairs (Klenin & Langowski, Biopolymers 2000), so no smoothing or sampling is involved.
A linking number of 0 does not prove two curves are unlinked (e.g. Whitehead link),
but nonzero values are certain for the polygons given.
"""
import itertools

import numpy as np

from .explorer import Network


def cycle_polyline(net, branch_ids):
    """Ordered closed polyline (voxel coordinates) and node set for a cycle of branches.

    Consecutive branches meet at a junction node; their attachment voxels may differ by a
    few voxels across a contracted junction region, which becomes one straight segment.
    """
    edges = [net.edges[b] for b in branch_ids]
    if len(edges) == 1:
        points = np.asarray(edges[0]['points_zyx'], dtype=float)
        return points, {edges[0]['source']}
    remaining = list(edges)
    first = remaining.pop(0)
    node = first['target']
    points = list(first['points_zyx'])
    nodes = {first['source'], first['target']}
    while remaining:
        nxt = next((e for e in remaining if node in (e['source'], e['target'])), None)
        if nxt is None:
            raise ValueError(f'branches {branch_ids} do not form a cycle')
        remaining.remove(nxt)
        path = nxt['points_zyx'] if nxt['source'] == node else nxt['points_zyx'][::-1]
        points.extend(path)
        node = nxt['target'] if nxt['source'] == node else nxt['source']
        nodes.add(node)
    return np.asarray(points, dtype=float), nodes


def linking_number(a, b, jitter=1e-4, seed=0):
    """Gauss linking number of two closed polygons (N×3 and M×3 arrays; closure implied).

    A tiny random displacement (much smaller than the voxel spacing) removes coplanar
    degeneracies of integer voxel coordinates; it cannot change the result for curves
    that are separated by more than the displacement.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float) + rng.uniform(-jitter, jitter, (len(a), 3))
    b = np.asarray(b, float) + rng.uniform(-jitter, jitter, (len(b), 3))
    p1, p2 = a, np.roll(a, -1, axis=0)
    p3, p4 = b, np.roll(b, -1, axis=0)
    p1, p2 = p1[:, None], p2[:, None]
    r13, r14, r23, r24 = p3 - p1, p4 - p1, p3 - p2, p4 - p2

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    n1 = unit(np.cross(r13, r14))
    n2 = unit(np.cross(r14, r24))
    n3 = unit(np.cross(r24, r23))
    n4 = unit(np.cross(r23, r13))
    dot = lambda u, v: np.clip(np.sum(u * v, axis=-1), -1, 1)
    omega = np.arcsin(dot(n1, n2)) + np.arcsin(dot(n2, n3)) + np.arcsin(dot(n3, n4)) + np.arcsin(dot(n4, n1))
    sign = np.sign(np.sum(np.cross(p4 - p3, p2 - p1) * r13, axis=-1))
    return float(np.nansum(omega * sign) / (4 * np.pi))


def candidate_cycles(net):
    """Distinct cycles: the minimum basis loops plus the shortest cycle through each branch."""
    cycles = {}
    for loop in net.loops:
        cycles.setdefault(frozenset(loop['branch_ids']), ('basis', loop['loop_id']))
    for branch in net.edges:
        if net.stats[branch].get('bridge'):
            continue
        route = net.shortest_loop(branch)
        if route:
            cycles.setdefault(frozenset(route), ('shortest through branch', branch))
    return cycles


def find_links(branch_graph, scale, min_abs=.5):
    """Linking numbers for all pairs of node-disjoint candidate cycles whose boxes overlap.

    Returns (links, summary): `links` lists pairs with |linking number| ≥ min_abs.
    """
    net = Network(branch_graph)
    scale = np.asarray(scale, float)
    cycles = []
    for branches, origin in candidate_cycles(net).items():
        points, nodes = cycle_polyline(net, sorted(branches))
        physical = points * scale
        cycles.append(dict(branches=sorted(branches), origin=origin, nodes=nodes, points=physical,
                           low=physical.min(0), high=physical.max(0),
                           length_um=net.length(branches)))
    tested = 0
    links = []
    for i, j in itertools.combinations(range(len(cycles)), 2):
        a, b = cycles[i], cycles[j]
        if a['nodes'] & b['nodes']:
            continue  # cycles that touch cannot be cleanly linked or unlinked
        if np.any(a['high'] < b['low']) or np.any(b['high'] < a['low']):
            continue  # disjoint bounding boxes: linking number is 0
        tested += 1
        lk = linking_number(a['points'], b['points'])
        if abs(lk) >= min_abs:
            gap = min_distance(a['points'], b['points'])
            links.append(dict(linking_number=int(round(lk)), raw=lk, cycle_a=a['branches'], cycle_b=b['branches'],
                              length_a_um=a['length_um'], length_b_um=b['length_um'], min_distance_um=gap,
                              origin_a=a['origin'], origin_b=b['origin']))
    disjoint_pairs = sum(1 for a, b in itertools.combinations(cycles, 2) if not a['nodes'] & b['nodes'])
    return links, dict(cycles=len(cycles), disjoint_pairs=disjoint_pairs, pairs_tested=tested, linked_pairs=len(links))


def min_distance(a, b):
    from scipy.spatial import cKDTree
    return float(cKDTree(b).query(a)[0].min())


def complement_cycle_basis(net, removed_nodes):
    """Fundamental cycle basis (lists of branch ids) of the network with `removed_nodes` deleted."""
    import networkx as nx
    h = net.g.copy()
    h.remove_nodes_from(removed_nodes)
    simple = nx.Graph()
    for a, b, k, d in h.edges(keys=True, data=True):
        if a != b and (not simple.has_edge(a, b) or d['length_um'] < simple[a][b]['length_um']):
            simple.add_edge(a, b, length_um=d['length_um'], key=k)
    tree = nx.minimum_spanning_tree(simple, weight='length_um')
    tree_keys = {d['key'] for *_, d in tree.edges(data=True)}
    basis = []
    for a, b, k in h.edges(keys=True):
        if k in tree_keys:
            continue
        if a == b:
            basis.append([k])
            continue
        path = nx.shortest_path(tree, a, b)
        basis.append([k] + [tree[u][v]['key'] for u, v in zip(path[:-1], path[1:])])
    return basis


def exhaustive_links(branch_graph, scale, min_abs=.5):
    """For each candidate cycle C, test whether C links ANY cycle of the network avoiding C's nodes.

    Linking number with C is additive over the cycle space of the remaining network, so
    C links some disjoint cycle there exactly when it has nonzero linking number with at
    least one cycle of a basis of that space. Returns (findings, summary).
    """
    net = Network(branch_graph)
    scale = np.asarray(scale, float)
    findings, tests, seen = [], 0, set()
    cache = {}

    def polyline(branches):
        key = frozenset(branches)
        if key not in cache:
            cache[key] = cycle_polyline(net, sorted(branches))[0] * scale
        return cache[key]

    candidates = candidate_cycles(net)
    for branches, origin in candidates.items():
        c_points, c_nodes = cycle_polyline(net, sorted(branches))
        c_points = c_points * scale
        for other in complement_cycle_basis(net, c_nodes):
            tests += 1
            lk = linking_number(c_points, polyline(other))
            pair = frozenset((frozenset(branches), frozenset(other)))
            if abs(lk) >= min_abs and pair not in seen:
                seen.add(pair)  # the same pair can be found from either cycle
                findings.append(dict(linking_number=int(round(lk)), raw=lk, cycle=sorted(branches), origin=origin,
                                     partner=sorted(other), length_um=net.length(branches),
                                     partner_length_um=net.length(other),
                                     min_distance_um=min_distance(c_points, polyline(other))))
    return findings, dict(cycles=len(candidates), basis_tests=tests, linked=len(findings))
