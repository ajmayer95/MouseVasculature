"""Exploratory, calibrated mask and branch statistics; no flow assumptions."""
import csv
import importlib.metadata
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.measure import euler_number


BRANCH_FIELDS = ['branch_id', 'raw_edge_id', 'source', 'target', 'component_id',
                 'length_um', 'chord_um', 'tortuosity', 'radius_mean_um',
                 'radius_min_um', 'radius_max_um', 'radius_median_um',
                 'touches_boundary', 'self_loop', 'short_branch', 'bridge']


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, median=None, p10=None, p90=None, min=None, max=None)
    return dict(n=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)),
                min=float(values.min()), max=float(values.max()))


def contract_junctions(graph):
    """Collapse connected junction voxels, retaining external paths and loops.

    Only direct, two-voxel junction-to-junction edges are absorbed. Their geometry
    and cycle rank remain available in the raw graph. Branch lengths run between
    attachment voxels, not through region centroids.
    """
    junction_ids = {n['id'] for n in graph['nodes'] if n['kind'] == 'junction'}
    regions = nx.Graph()
    regions.add_nodes_from(sorted(junction_ids))
    internal_ids = set()
    for e in graph['edges']:
        if e['source'] in junction_ids and e['target'] in junction_ids and len(e['points_zyx']) == 2:
            regions.add_edge(e['source'], e['target'])
            internal_ids.add(e['id'])
    groups = [sorted(c) for c in nx.connected_components(regions)]
    groups.extend([n['id']] for n in graph['nodes'] if n['id'] not in junction_ids)
    groups.sort(key=lambda c: c[0])
    original = {n['id']: n for n in graph['nodes']}
    mapping, nodes = {}, []
    for i, group in enumerate(groups):
        mapping.update({j: i for j in group})
        nodes.append(dict(id=i, point_zyx=np.mean([original[j]['point_zyx'] for j in group], axis=0).tolist(),
                          raw_node_ids=group, junction_voxels=len(group) if group[0] in junction_ids else 0))
    edges = [dict(e, id=i, raw_edge_id=e['id'], source=mapping[e['source']], target=mapping[e['target']])
             for i, e in enumerate(e for e in graph['edges'] if e['id'] not in internal_ids)]
    return dict(nodes=nodes, edges=edges, coordinate_order='ZYX', coordinate_units='voxel',
                voxel_size_um=graph['voxel_size_um'],
                absorbed_junction_edge_count=len(internal_ids),
                absorbed_junction_length_um=sum(e['length_um'] for e in graph['edges'] if e['id'] in internal_ids))


def network(graph):
    g = nx.MultiGraph()
    g.add_nodes_from(n['id'] for n in graph['nodes'])
    for e in graph['edges']:
        g.add_edge(e['source'], e['target'], key=e['id'], length_um=e['length_um'])
    return g


def merge_degree_two(graph, raw_graph):
    """Trace maximal branches, joining across degree-two junction regions.

    Use the shortest path through original junction voxels for each connector;
    never substitute a straight chord across an unresolved region.
    """
    g = network(graph)
    raw_nodes = {n['id']: n for n in raw_graph['nodes']}
    voxel_ids = {tuple(n['point_zyx']): n['id'] for n in raw_graph['nodes']}
    internal = nx.Graph()
    for e in raw_graph['edges']:
        if (len(e['points_zyx']) == 2 and raw_nodes[e['source']]['kind'] == 'junction'
                and raw_nodes[e['target']]['kind'] == 'junction'):
            internal.add_edge(e['source'], e['target'], weight=e['length_um'])
    by_id = {e['id']: e for e in graph['edges']}
    visited, kept_nodes, edges = set(), set(), []
    scale = np.asarray(graph['voxel_size_um'])
    connector_length = 0.

    def trace(start, first):
        nonlocal connector_length
        current, edge_id = start, first
        points, originals = [], []
        while True:
            e = by_id[edge_id]
            visited.add(edge_id)
            path = e['points_zyx'] if e['source'] == current else e['points_zyx'][::-1]
            if points:
                if points[-1] != path[0]:
                    a, b = voxel_ids[tuple(points[-1])], voxel_ids[tuple(path[0])]
                    route = nx.shortest_path(internal, a, b, weight='weight')
                    connector = [raw_nodes[i]['point_zyx'] for i in route]
                    connector_length += float(np.linalg.norm(np.diff(connector, axis=0) * scale, axis=1).sum())
                    points.extend(connector[1:])
                points.extend(path[1:])
            else:
                points.extend(path)
            originals.append(e['raw_edge_id'])
            current = e['target'] if current == e['source'] else e['source']
            if current == start or g.degree(current) != 2:
                if current == start and g.degree(start) == 2 and points[-1] != points[0]:
                    route = nx.shortest_path(internal, voxel_ids[tuple(points[-1])],
                                             voxel_ids[tuple(points[0])], weight='weight')
                    connector = [raw_nodes[i]['point_zyx'] for i in route]
                    connector_length += float(np.linalg.norm(np.diff(connector, axis=0) * scale, axis=1).sum())
                    points.extend(connector[1:])
                break
            edge_id = next(k for _, _, k in g.edges(current, keys=True) if k not in visited)
        kept_nodes.update((start, current))
        length = float(np.linalg.norm(np.diff(points, axis=0) * scale, axis=1).sum())
        edges.append(dict(id=len(edges), source=start, target=current, points_zyx=points,
                          length_um=length, raw_edge_id=originals[0], raw_edge_ids=originals))

    for node in sorted(g):
        if g.degree(node) != 2:
            kept_nodes.add(node)
            for _, _, key in g.edges(node, keys=True):
                if key not in visited:
                    trace(node, key)
    for edge in graph['edges']:
        if edge['id'] not in visited:
            trace(edge['source'], edge['id'])
    return dict(graph, nodes=[n for n in graph['nodes'] if n['id'] in kept_nodes],
                edges=edges, junction_connector_length_um=connector_length,
                degree_two_regions_merged=g.number_of_nodes() - len(kept_nodes))


LOOP_FIELDS = ['loop_id', 'kind', 'length_um', 'extent_um', 'small', 'branch_ids', 'centroid_zyx',
               'min_radius_um', 'touches_boundary', 'has_short_branch']
BIFURCATION_FIELDS = ['node_id', 'component_id', 'parent_branch', 'child_branches',
                      'parent_radius_um', 'child_radius_large_um', 'child_radius_small_um',
                      'murray_ratio', 'murray_exponent', 'child_asymmetry', 'branching_angle_deg',
                      'touches_boundary', 'has_short_branch', 'resolved']
TANGENT_LENGTH_UM = 10.
RESOLVED_RADIUS_VOXELS = 2.
SMALL_LOOP_UM = 50.
# Vessel ends this close to a face (beyond their radius) are counted as crop-cut: in these
# tracings 80% of ends that do not reach a face stop within 20 µm of one (vs ~22% by chance).
CROP_END_MARGIN_UM = 20.


def minimum_loops(g):
    """Return an exact minimum-weight cycle basis as loops of branch ids.

    Self-loops and parallel branches are subdivided into three equal segments so that
    a simple weighted graph represents the multigraph exactly. Bridges and tree-like
    parts are removed first, and each biconnected block is solved separately.
    Basis size equals the cycle rank. Each loop is classified as `self_loop` (one
    branch), `parallel` (two branches joining the same two nodes) or `cycle`.
    """
    simple = nx.Graph()
    for a, b, key, d in g.edges(keys=True, data=True):
        if a != b and g.number_of_edges(a, b) == 1:
            simple.add_edge(a, b, weight=d['length_um'], branch=key)
            continue
        m0, m1, w = ('branch', key, 0), ('branch', key, 1), d['length_um'] / 3
        simple.add_edge(a, m0, weight=w, branch=key)
        simple.add_edge(m0, m1, weight=w, branch=key)
        simple.add_edge(m1, b, weight=w, branch=key)
    loops = []
    for block in nx.biconnected_components(simple):
        if len(block) < 3:
            continue
        sub = simple.subgraph(block)
        edges = list(sub.edges(data='branch'))
        for weight, bits in horton_cycle_basis(sub):
            branches = sorted({edges[i][2] for i in range(bits.bit_length()) if bits >> i & 1})
            kind = 'self_loop' if len(branches) == 1 else 'parallel' if len(branches) == 2 else 'cycle'
            loops.append(dict(length_um=weight, branch_ids=branches, kind=kind))
    loops.sort(key=lambda loop: loop['length_um'])
    for i, loop in enumerate(loops):
        loop['loop_id'] = i
    return loops


def loop_lengths(g):
    """Perimeters of a minimum-weight cycle basis, ascending."""
    return [loop['length_um'] for loop in minimum_loops(g)]


def horton_cycle_basis(graph):
    """Exact minimum cycle basis of a connected simple graph (Horton, 1987).

    Returns (weight, bitmask) pairs; bit i is the i-th edge of graph.edges().
    networkx 3.2's minimum_cycle_basis returned non-minimal bases on these graphs.
    Candidates are shortest-path cycles through each vertex and edge; they are added
    greedily by weight when independent over GF(2).
    """
    edge_bit = {frozenset(e): i for i, e in enumerate(graph.edges())}
    rank = graph.number_of_edges() - graph.number_of_nodes() + 1
    candidates = []
    for v in graph:
        distance, paths = nx.single_source_dijkstra(graph, v, weight='weight')
        for x, y, d in graph.edges(data=True):
            px, py = paths[x], paths[y]
            if px[-2:-1] == [y] or py[-2:-1] == [x] or set(px) & set(py) != {v}:
                continue
            cycle = px + py[::-1]
            bits = 0
            for a, b in zip(cycle[:-1], cycle[1:]):
                bits |= 1 << edge_bit[frozenset((a, b))]
            candidates.append((distance[x] + distance[y] + d['weight'], bits))
    candidates.sort(key=lambda c: c[0])
    pivots, basis, seen = {}, [], set()
    for weight, original in candidates:
        if original in seen:
            continue
        seen.add(original)
        bits = original
        while bits:
            top = bits.bit_length() - 1
            if top not in pivots:
                pivots[top] = bits
                basis.append((float(weight), original))
                break
            bits ^= pivots[top]
        if len(basis) == rank:
            break
    return basis


def describe_loops(loops, branch_graph, rows, scale):
    """Add centroid, extent and crop/short flags to each loop."""
    by_id = {e['id']: e for e in branch_graph['edges']}
    stats = {r['branch_id']: r for r in rows}
    for loop in loops:
        points = np.concatenate([np.asarray(by_id[b]['points_zyx'], dtype=float) for b in loop['branch_ids']])
        centroid = points.mean(axis=0)
        loop['centroid_zyx'] = centroid.tolist()
        loop['extent_um'] = float(2 * np.linalg.norm((points - centroid) * scale, axis=1).max())
        loop['small'] = loop['length_um'] < SMALL_LOOP_UM
        loop['touches_boundary'] = any(stats[b]['touches_boundary'] for b in loop['branch_ids'])
        loop['has_short_branch'] = any(stats[b]['short_branch'] for b in loop['branch_ids'])
        loop['min_radius_um'] = min(stats[b]['radius_min_um'] for b in loop['branch_ids'])
    return loops


def tangent(points, scale, length_um=TANGENT_LENGTH_UM):
    """Unit direction from the first point to the point about length_um along the centerline."""
    points = np.asarray(points, dtype=float) * scale
    arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    end = points[min(int(np.searchsorted(arc, length_um)), len(points) - 1)]
    vector = end - points[0]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def murray_exponent(parent, large, small):
    """Solve parent^x = large^x + small^x; None if no finite positive solution exists."""
    from scipy.optimize import brentq
    if min(parent, large, small) <= 0 or parent <= large:
        return None
    f = lambda x: 1 - (large / parent) ** x - (small / parent) ** x
    return float(brentq(f, 1e-6, 200)) if f(200) > 0 else None


def bifurcations(g, branch_graph, rows, scale, membership):
    """Describe each simple three-branch node. The widest branch is called the parent.

    This is a geometric convention, not a flow direction.
    """
    by_id = {e['id']: e for e in branch_graph['edges']}
    stats = {r['branch_id']: r for r in rows}
    resolved_radius = RESOLVED_RADIUS_VOXELS * float(np.min(scale))
    result = []
    for node in sorted(g):
        edges = [k for *_, k in g.edges(node, keys=True)]
        if len(edges) != 3 or len(set(edges)) != 3 or any(by_id[k]['source'] == by_id[k]['target'] for k in edges):
            continue
        ordered = sorted(edges, key=lambda k: -stats[k]['radius_mean_um'])
        parent, large, small = (stats[k]['radius_mean_um'] for k in ordered)
        directions = []
        for k in ordered[1:]:
            e = by_id[k]
            points = e['points_zyx'] if e['source'] == node else e['points_zyx'][::-1]
            directions.append(tangent(points, scale))
        angle = (float(np.degrees(np.arccos(np.clip(directions[0] @ directions[1], -1, 1))))
                 if all(d is not None for d in directions) else None)
        boundary = any(stats[k]['touches_boundary'] for k in edges)
        short = any(stats[k]['short_branch'] for k in edges)
        result.append(dict(node_id=node, component_id=membership[node], parent_branch=ordered[0],
                           child_branches=' '.join(map(str, ordered[1:])),
                           parent_radius_um=parent, child_radius_large_um=large, child_radius_small_um=small,
                           murray_ratio=parent ** 3 / (large ** 3 + small ** 3),
                           murray_exponent=murray_exponent(parent, large, small),
                           child_asymmetry=small / large, branching_angle_deg=angle,
                           touches_boundary=boundary, has_short_branch=short,
                           resolved=bool(not boundary and not short and small >= resolved_radius)))
    return result


def tissue_distance(mask, scale):
    """Distance from each background voxel to the nearest foreground voxel, in µm.

    Vessels outside the crop are unknown, so each value is an upper bound. It is exact
    when no point outside the ROI could be closer than the measured distance.
    """
    mask = mask.astype(bool)
    if not mask.any() or mask.all():
        return distribution([]), None
    distance = ndi.distance_transform_edt(~mask, sampling=scale)
    face = None
    for axis, (n, s) in enumerate(zip(mask.shape, scale)):
        index = np.arange(n, dtype=float)
        d = np.minimum(index + 1, n - index) * s
        d = d.reshape([-1 if i == axis else 1 for i in range(3)])
        face = d if face is None else np.minimum(face, d)
    background = ~mask
    values = distance[background]
    exact = float(np.mean(values <= np.broadcast_to(face, mask.shape)[background]))
    return distribution(values), exact


def face_distance_um(points, shape, scale):
    """Distance in µm from each voxel centre to the nearest ROI face (faces lie half a voxel out)."""
    points = np.asarray(points, dtype=float)
    shape, scale = np.asarray(shape, dtype=float), np.asarray(scale, dtype=float)
    return np.minimum((points + .5) * scale, (shape - .5 - points) * scale).min(axis=1)


def reaches_face(points, radii, shape, scale):
    """Centerline points whose vessel cross-section reaches an ROI face.

    Thinning pulls skeleton ends back from a crop face by about the vessel radius, so a
    vessel cut by the crop rarely has a centerline voxel on the face itself. A point counts
    when its distance to the face is at most its EDT radius plus one (largest) voxel.
    """
    return face_distance_um(points, shape, scale) <= np.asarray(radii) + float(np.max(scale))


def measure(mask, skeleton, raw_graph):
    scale = np.asarray(raw_graph['voxel_size_um'])
    contracted = contract_junctions(raw_graph)
    junction_sizes = [n['junction_voxels'] for n in contracted['nodes'] if n['junction_voxels']]
    branch_graph = merge_degree_two(contracted, raw_graph)
    g, raw = network(branch_graph), network(raw_graph)
    components = sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c)))
    membership = {node: i for i, c in enumerate(components) for node in c}
    # Pad with background so objects at a crop face have finite EDT values.
    distance = ndi.distance_transform_edt(np.pad(mask.astype(bool), 1), sampling=scale)[1:-1, 1:-1, 1:-1]
    simple = nx.Graph(g)
    bridge_pairs = {frozenset((a, b)) for a, b in nx.bridges(simple) if g.number_of_edges(a, b) == 1}
    rows = []
    shape = np.asarray(mask.shape)
    orientation = np.zeros((3, 3))
    crop_ends = {}
    for e in branch_graph['edges']:
        points = np.asarray(e['points_zyx'], dtype=int)
        radii = distance[tuple(points.T)]
        steps = np.linalg.norm(np.diff(points, axis=0) * scale, axis=1)
        length = float(steps.sum())
        directions = np.diff(points, axis=0) * scale / steps[:, None]
        orientation += np.einsum('n,ni,nj->ij', steps, directions, directions)
        chord = float(np.linalg.norm((points[-1] - points[0]) * scale))
        loop = e['source'] == e['target']
        at_face = reaches_face(points, radii, shape, scale)
        end_gap = face_distance_um(points[[0, -1]], shape, scale) - radii[[0, -1]]
        crop_ends.setdefault(e['source'], []).append(bool(end_gap[0] <= max(CROP_END_MARGIN_UM, scale.max())))
        crop_ends.setdefault(e['target'], []).append(bool(end_gap[-1] <= max(CROP_END_MARGIN_UM, scale.max())))
        row = dict(branch_id=e['id'], raw_edge_id=e['raw_edge_id'], source=e['source'], target=e['target'],
                   component_id=membership[e['source']], length_um=length, chord_um=chord,
                   tortuosity=max(1., length / chord) if chord > 0 and not loop else None,
                   radius_mean_um=float(np.sum((radii[:-1] + radii[1:]) * .5 * steps) / length),
                   radius_min_um=float(radii.min()), radius_max_um=float(radii.max()),
                   radius_median_um=float(np.median(radii)),
                   touches_boundary=bool(np.any(at_face)),
                   self_loop=loop, short_branch=bool(length < 3 * scale.min()),
                   bridge=frozenset((e['source'], e['target'])) in bridge_pairs)
        rows.append(row)
        e['radius_profile_um'] = radii.tolist()
        e['statistics'] = row
    del distance
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    del labels
    foreground = int(np.count_nonzero(mask))
    roi_volume = float(mask.size * np.prod(scale))
    length = sum(r['length_um'] for r in rows)
    degrees = dict(g.degree())
    for n in branch_graph['nodes']:
        n['degree'] = degrees[n['id']]
        n['component_id'] = membership[n['id']]
    component_rows = []
    for i, c in enumerate(components):
        sub = g.subgraph(c)
        component_rows.append(dict(component_id=i, nodes=len(c), branches=sub.number_of_edges(),
                                   independent_cycles=sub.number_of_edges() - len(c) + 1,
                                   length_um=sum(d['length_um'] for *_, d in sub.edges(data=True))))
    warnings = []
    if junction_sizes and max(junction_sizes) > 27:
        warnings.append('Large junction regions (>27 voxels): skeleton may contain sheets or unresolved clusters. Branch counts and lengths require visual QC.')
    if any(r['touches_boundary'] for r in rows):
        warnings.append('Boundary-touching branches are crop-censored; radii near crop faces use padded background.')
    warnings.append('EDT radii estimate distance to background in the supplied mask, not lumen radius unless masks represent filled vessels.')
    summary = dict(schema_version=4, voxel_size_um=scale.tolist(), roi_volume_um3=roi_volume,
                   foreground_volume_um3=float(foreground * np.prod(scale)),
                   foreground_fraction=foreground / mask.size,
                   mask_components_26=int(count), largest_mask_component_fraction=float(sizes.max() / foreground) if foreground else None,
                   mask_euler_characteristic_26=int(euler_number(mask.astype(bool), connectivity=3)),
                   skeleton_voxels=int(np.count_nonzero(skeleton)),
                   raw_nodes=raw.number_of_nodes(), raw_edges=raw.number_of_edges(),
                   raw_independent_cycles=raw.number_of_edges() - raw.number_of_nodes() + nx.number_connected_components(raw),
                   nodes=g.number_of_nodes(), branches=g.number_of_edges(), components=len(components),
                   independent_cycles=g.number_of_edges() - g.number_of_nodes() + len(components),
                   endpoints=sum(d == 1 for d in degrees.values()),
                   crop_endpoints=sum(d == 1 and any(crop_ends.get(n, [])) for n, d in degrees.items()),
                   interior_endpoints=sum(d == 1 and not any(crop_ends.get(n, [])) for n, d in degrees.items()),
                   crop_end_margin_um=CROP_END_MARGIN_UM,
                   branch_nodes=sum(d >= 3 for d in degrees.values()),
                   isolated_nodes=sum(d == 0 for d in degrees.values()), self_loops=nx.number_of_selfloops(g),
                   bridges=sum(r['bridge'] for r in rows),
                   total_branch_length_um=length, length_density_mm_per_mm3=length / roi_volume * 1e6,
                   branch_density_per_mm3=len(rows) / roi_volume * 1e9,
                   length_weighted_radius_um=sum(r['length_um'] * r['radius_mean_um'] for r in rows) / length if length else None,
                   largest_component_length_fraction=max((c['length_um'] for c in component_rows), default=0) / length if length else None,
                   boundary_branch_fraction=sum(r['touches_boundary'] for r in rows) / len(rows) if rows else None,
                   short_branch_fraction=sum(r['short_branch'] for r in rows) / len(rows) if rows else None,
                   absorbed_junction_length_um=branch_graph['absorbed_junction_length_um'],
                   junction_connector_length_um=branch_graph['junction_connector_length_um'],
                   degree_two_regions_merged=branch_graph['degree_two_regions_merged'],
                   max_junction_region_voxels=max(junction_sizes, default=0),
                   length_um=distribution([r['length_um'] for r in rows]),
                   radius_mean_um=distribution([r['radius_mean_um'] for r in rows]),
                   tortuosity=distribution([r['tortuosity'] for r in rows]),
                   degree_histogram={str(k): list(degrees.values()).count(k) for k in sorted(set(degrees.values()))},
                   warnings=warnings,
                   preprocessing=raw_graph.get('preprocessing', {}),
                   versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-image', 'networkx')})
    eigenvalues = np.linalg.eigvalsh(orientation / length)[::-1] if length else np.zeros(3)
    summary['orientation_eigenvalues'] = eigenvalues.tolist()
    summary['orientation_anisotropy'] = float(eigenvalues[0] - eigenvalues[-1]) if length else None
    summary['interior_branch_length_um'] = distribution([r['length_um'] for r in rows if not r['touches_boundary']])
    filled = summary['preprocessing'].get('fill_enclosed_holes', False)
    summary['filled_mask_tunnel_count'] = (summary['mask_components_26'] - summary['mask_euler_characteristic_26']) if filled else None
    summary['graph_minus_mask_tunnel_count'] = (summary['independent_cycles'] - summary['filled_mask_tunnel_count']) if filled else None
    if filled and summary['graph_minus_mask_tunnel_count'] != 0:
        warnings.append('Graph cycle rank differs from filled-mask tunnel count: inspect junction contraction and voxel connectivity before interpreting topology.')

    # Loopiness
    branch_nodes = summary['branch_nodes']
    loop_records = describe_loops(minimum_loops(g), branch_graph, rows, scale)
    loops = [loop['length_um'] for loop in loop_records]
    summary['cycle_density_per_mm3'] = summary['independent_cycles'] / roi_volume * 1e9
    summary['meshedness'] = summary['independent_cycles'] / (2 * branch_nodes - 5) if branch_nodes >= 3 else None
    summary['looped_length_fraction'] = sum(r['length_um'] for r in rows if not r['bridge']) / length if length else None
    summary['loop_length_um'] = distribution(loops)
    large_loops = [x for x in loops if x >= SMALL_LOOP_UM]
    summary['small_loop_threshold_um'] = SMALL_LOOP_UM
    summary['small_loops'] = len(loops) - len(large_loops)
    summary['large_cycle_density_per_mm3'] = len(large_loops) / roi_volume * 1e9
    summary['large_loop_length_um'] = distribution(large_loops)
    if loops and summary['small_loops'] / len(loops) > .25:
        warnings.append(f'Over 25% of minimum-basis loops are shorter than {SMALL_LOOP_UM:g} µm; '
                        'small mask tunnels may inflate cycle counts. Compare large_cycle_density_per_mm3.')

    # Hierarchy
    junctions = bifurcations(g, branch_graph, rows, scale, membership)
    resolved = [b for b in junctions if b['resolved']]
    summary['bifurcations'] = len(junctions)
    summary['resolved_bifurcations'] = len(resolved)
    summary['resolved_radius_threshold_um'] = RESOLVED_RADIUS_VOXELS * float(scale.min())
    summary['murray_ratio'] = distribution([b['murray_ratio'] for b in resolved])
    summary['murray_exponent'] = distribution([b['murray_exponent'] for b in resolved])
    summary['murray_exponent_unsolved'] = sum(b['murray_exponent'] is None for b in resolved)
    summary['child_asymmetry'] = distribution([b['child_asymmetry'] for b in resolved])
    summary['branching_angle_deg'] = distribution([b['branching_angle_deg'] for b in junctions
                                                   if not b['touches_boundary'] and not b['has_short_branch']])
    interior = [r for r in rows if not r['touches_boundary'] and not r['self_loop']]
    summary['radius_length_spearman'] = (float(spearmanr([r['radius_mean_um'] for r in interior],
                                                         [r['length_um'] for r in interior])[0])
                                         if len(interior) >= 3 else None)
    if summary['radius_length_spearman'] is not None and not np.isfinite(summary['radius_length_spearman']):
        summary['radius_length_spearman'] = None
    branch_graph['bifurcations'] = junctions
    branch_graph['loop_lengths_um'] = loops
    branch_graph['loops'] = loop_records
    summary['loop_kinds'] = {k: sum(l['kind'] == k for l in loop_records) for k in ('cycle', 'parallel', 'self_loop')}
    summary['small_loop_kinds'] = {k: sum(l['kind'] == k and l['small'] for l in loop_records) for k in ('cycle', 'parallel', 'self_loop')}

    # Tissue supply
    summary['tissue_distance_um'], summary['tissue_distance_exact_fraction'] = tissue_distance(mask, scale)
    if summary['tissue_distance_exact_fraction'] is not None and summary['tissue_distance_exact_fraction'] < .9:
        warnings.append('Over 10% of tissue-to-vessel distances are crop-limited upper bounds.')
    if junctions and len(resolved) < 10:
        warnings.append('Fewer than 10 resolved bifurcations: Murray statistics are unreliable.')
    return summary, rows, component_rows, branch_graph


def variant(preprocessing):
    if preprocessing.get('close_tunnels'):
        return 'filled+closed'
    return 'filled' if preprocessing.get('fill_enclosed_holes') else 'original'


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_statistics(folder, source, mask, skeleton, graph, plots=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    summary, rows, components, branch_graph = measure(mask, skeleton, graph)
    summary['source_file'] = Path(source).name
    summary['analysis_variant'] = variant(summary['preprocessing'])
    (folder / 'statistics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (folder / 'branch_graph.json').write_text(json.dumps(branch_graph, indent=2, allow_nan=False) + '\n')
    write_csv(folder / 'branches.csv', rows, BRANCH_FIELDS)
    write_csv(folder / 'components.csv', components, ['component_id', 'nodes', 'branches', 'independent_cycles', 'length_um'])
    write_csv(folder / 'bifurcations.csv', branch_graph['bifurcations'], BIFURCATION_FIELDS)
    write_csv(folder / 'loops.csv', [dict(l, branch_ids=' '.join(map(str, l['branch_ids'])),
                                          centroid_zyx=' '.join(f'{v:.1f}' for v in l['centroid_zyx']))
                                     for l in branch_graph['loops']], LOOP_FIELDS)
    if plots:
        plot_statistics(folder / 'statistics.png', rows, summary, branch_graph)
    print(f"Statistics: {len(rows)} branches, {summary['components']} components, "
          f"{summary['independent_cycles']} independent graph cycles", flush=True)
    for warning in summary['warnings']:
        print('QC:', warning, flush=True)
    return summary, branch_graph


def plot_statistics(path, rows, summary, branch_graph=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    resolved = [b for b in (branch_graph or {}).get('bifurcations', []) if b['resolved']]
    for ax, key, label in zip([axes[0, 0], axes[0, 1], axes[0, 2]], ['length_um', 'radius_mean_um', 'tortuosity'],
                             ['Branch length (µm)', 'EDT radius, branch mean (µm)', 'Tortuosity (length / chord)']):
        values = [r[key] for r in rows if r[key] is not None]
        ax.hist(values, bins=min(40, max(1, int(np.sqrt(len(values))))), color='#147d92')
        ax.set(xlabel=label, ylabel='Branches')
        ax.text(.98, .95, f'n={len(values)}', transform=ax.transAxes, ha='right', va='top')
    degrees = summary['degree_histogram']
    axes[1, 0].bar([int(k) for k in degrees], list(degrees.values()), color='#7956a0')
    axes[1, 0].set(xlabel='Node degree (self-loop contributes 2)', ylabel='Nodes')
    for ax, values, label, color in [
            (axes[1, 1], (branch_graph or {}).get('loop_lengths_um', []), 'Minimum-basis loop length (µm)', '#147d92'),
            (axes[1, 2], [np.log2(b['murray_ratio']) for b in resolved], 'log2 Murray ratio, resolved (0 = cube law)', '#a35328')]:
        ax.hist(values, bins=min(40, max(1, int(np.sqrt(len(values))))), color=color)
        ax.set(xlabel=label, ylabel='Count')
        ax.text(.98, .95, f'n={len(values)}', transform=ax.transAxes, ha='right', va='top')
    axes[1, 2].axvline(0, color='0.3', lw=1, ls='--')
    fig.suptitle(summary['source_file'] + f" ({summary['analysis_variant']})" + '\nExploratory statistics after junction-region contraction')
    fig.text(.5, .015, 'Includes short and crop-boundary branches. EDT radius assumes filled vessel masks. See statistics.json QC.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


DISTRIBUTIONS = ('length_um', 'radius_mean_um', 'tortuosity', 'loop_length_um', 'large_loop_length_um', 'murray_ratio',
                 'murray_exponent', 'child_asymmetry', 'branching_angle_deg', 'tissue_distance_um')


def save_comparison(folder, summaries):
    # Refresh the comparison from every completed analysis in this output folder.
    # Original and hole-filled variants remain separate rows.
    saved = sorted(Path(folder).glob('*/statistics.json'))
    if saved:
        summaries = [json.loads(p.read_text()) for p in saved]
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        for key in DISTRIBUTIONS:
            row.update({f'{key}_{stat}': value for stat, value in s.get(key, {}).items()})
        for key in ('loop_kinds', 'small_loop_kinds'):
            row.update({f'{key}_{kind}': n for kind, n in s.get(key, {}).items()})
        row['qc_warnings'] = ' | '.join(s['warnings'])
        row['fill_enclosed_holes'] = s.get('preprocessing', {}).get('fill_enclosed_holes', False)
        row['added_voxels'] = s.get('preprocessing', {}).get('added_voxels', 0)
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(Path(folder) / 'summary.csv', rows, fields)
        plot_comparison(Path(folder) / 'overview.png', rows)


def plot_comparison(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sources = sorted({r['source_file'] for r in rows})
    positions = {s: i for i, s in enumerate(sources)}
    fig, axes = plt.subplots(1, 7, figsize=(22, max(5, len(sources) * .35)), sharey=True)
    variants = [('original', '^', '#a35328'), ('filled', 'o', '#147d92'), ('filled+closed', 's', '#4a3aa8')]
    for ax, key, label in zip(axes, ['length_density_mm_per_mm3', 'radius_mean_um_median',
                                   'tortuosity_median', 'large_cycle_density_per_mm3', 'looped_length_fraction',
                                   'murray_ratio_median', 'tissue_distance_um_median'],
                             ['Length density (mm/mm³)', 'Median branch EDT radius (µm)',
                              'Median branch tortuosity', 'Loops ≥50 µm per mm³', 'Length fraction in loops',
                              'Median Murray ratio (resolved)', 'Median tissue–vessel distance (µm)']):
        for variant, marker, color in variants:
            selected = [r for r in rows if r.get('analysis_variant', 'original') == variant and r.get(key) is not None]
            ax.scatter([r[key] for r in selected], [positions[r['source_file']] for r in selected],
                       marker=marker, color=color, label=variant, s=30)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(sources)), [s.replace('_Filaments.tif', '') for s in sources])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Mask variant', fontsize=8, loc='upper right', ncol=2)
    fig.suptitle('Mouse prostate vasculature — exploratory ROI statistics')
    fig.text(.5, .015, 'ROIs are not independent mice. Original-mask cavities can strongly bias graph measurements; review QC before comparisons.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)
