"""Voxel skeletons and centerline graphs with redundant diagonals removed."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize


def vectorize(skeleton, scale):
    """Trace degree-two chains; preserve junction voxels, loops and isolates.

    Coordinates are ZYX voxel indices. Lengths use calibrated micrometers.
    Adjacent junction voxels remain separate nodes (no junction contraction).
    """
    coords = np.argwhere(skeleton)
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    offsets = [p for p in product((-1, 0, 1), repeat=3) if p != (0, 0, 0)]
    adjacency = []
    for p in coords:
        neighbors = []
        for d in offsets:
            q = tuple(p + d)
            if q not in lookup:
                continue
            # Avoid diagonal shortcuts when a shorter local voxel route exists.
            # Retain diagonals needed for genuinely diagonal centerlines.
            intermediates = product(*[(0, v) if v else (0,) for v in d])
            if any(s != (0, 0, 0) and s != d and tuple(p + s) in lookup
                   for s in intermediates):
                continue
            neighbors.append(lookup[q])
        adjacency.append(sorted(neighbors))
    node_voxels = {i for i, neighbors in enumerate(adjacency) if len(neighbors) != 2}
    nodes = []
    node_ids = {}

    def add_node(i, kind=None):
        if i not in node_ids:
            node_ids[i] = len(nodes)
            degree = len(adjacency[i])
            nodes.append(dict(id=len(nodes), point_zyx=coords[i].tolist(),
                              degree=degree, kind=kind or ('isolated' if degree == 0 else
                              'endpoint' if degree == 1 else 'junction')))
        return node_ids[i]

    for i in sorted(node_voxels):
        add_node(i)
    visited = set()
    edges = []

    def trace(start, neighbor):
        chain = [start, neighbor]
        visited.add(tuple(sorted((start, neighbor))))
        previous, current = start, neighbor
        while current not in node_ids:
            following = next(j for j in adjacency[current] if j != previous)
            visited.add(tuple(sorted((current, following))))
            chain.append(following)
            previous, current = current, following
        points = coords[chain]
        length = float(np.linalg.norm(np.diff(points, axis=0) * scale, axis=1).sum())
        edges.append(dict(id=len(edges), source=node_ids[start], target=node_ids[current],
                          length_um=length, points_zyx=points.tolist()))

    for i in sorted(node_voxels):
        for j in adjacency[i]:
            if tuple(sorted((i, j))) not in visited:
                trace(i, j)
    # Components composed entirely of degree-two voxels need a loop anchor.
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if tuple(sorted((i, j))) not in visited:
                add_node(i, 'loop_anchor')
                trace(i, j)
    return dict(schema_version=1, coordinate_order='ZYX', coordinate_units='voxel',
                voxel_size_um=list(scale), connectivity=26,
                diagonal_shortcuts_removed=True, nodes=nodes, edges=edges,
                skeleton_voxels=len(coords))


def analyze(mask, scale):
    skeleton = (skeletonize(mask.astype(bool), method='lee') > 0).astype(np.uint8)
    return skeleton, vectorize(skeleton, scale)


def save_analysis(folder, stem, skeleton, graph, source):
    folder = Path(folder) / stem
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'skeleton.npz', skeleton=skeleton,
                        voxel_size_um=graph['voxel_size_um'])
    graph = dict(graph, source_file=Path(source).name, method='skimage skeletonize lee')
    (folder / 'graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    print(f"Saved {folder}: {graph['skeleton_voxels']:,} skeleton voxels, "
          f"{len(graph['nodes']):,} nodes, {len(graph['edges']):,} paths", flush=True)


def add_analysis_layers(viewer, skeleton, graph, scale, name, visible=True, color_by=None):
    layer = viewer.add_labels(skeleton, scale=scale, name=f'{name} skeleton',
                              rendering='iso_categorical', visible=False)
    layer.editable = False
    vectors, edge_ids = [], []
    features = {'edge_id': edge_ids, 'length_um': [], 'radius_mean_um': [], 'component_id': []}
    for edge in graph['edges']:
        points = np.asarray(edge['points_zyx'], dtype=float)
        vectors.extend(np.stack((points[:-1], np.diff(points, axis=0)), axis=1))
        edge_ids.extend([edge['id']] * (len(points) - 1))
        stats = edge.get('statistics', {})
        for key in ('length_um', 'radius_mean_um', 'component_id'):
            features[key].extend([stats.get(key, edge.get(key, 0))] * (len(points) - 1))
    if vectors:
        colors = dict(length='length_um', radius='radius_mean_um', component='component_id')
        layer = viewer.add_vectors(np.asarray(vectors), ndim=3, scale=scale,
                                   vector_style='line',
                                   edge_color=colors.get(color_by, 'cyan'), edge_width=2, opacity=1,
                                   edge_colormap='viridis',
                                   name=f'{name} centerlines' + (f' ({colors[color_by]})' if color_by else ''), visible=visible,
                                   features=features)
        # napari 0.5.6's Qt controls select the first feature during creation.
        # Reapply the requested mapping after the layer is attached to the viewer.
        layer.edge_color = colors.get(color_by, 'cyan')
        if color_by:
            layer.edge_color_mode = 'cycle' if color_by == 'component' else 'colormap'
        layer.editable = False
    for kind, color in [('endpoint', 'yellow'), ('junction', 'magenta'),
                        ('isolated', 'white'), ('loop_anchor', 'orange')]:
        points = [n['point_zyx'] for n in graph['nodes']
                  if n.get('kind', 'endpoint' if n.get('degree') == 1 else
                           'junction' if n.get('degree', 0) >= 3 else
                           'isolated' if n.get('degree') == 0 else 'loop_anchor') == kind]
        if points:
            layer = viewer.add_points(np.asarray(points), ndim=3, scale=scale,
                                      size=3, face_color=color, name=f'{name} {kind}',
                                      visible=visible and kind == 'endpoint')
            layer.editable = False
