"""Exploratory, calibrated mask and branch statistics; no flow assumptions."""
import csv
import importlib.metadata
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage as ndi
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
    for e in branch_graph['edges']:
        points = np.asarray(e['points_zyx'], dtype=int)
        radii = distance[tuple(points.T)]
        steps = np.linalg.norm(np.diff(points, axis=0) * scale, axis=1)
        length = float(steps.sum())
        directions = np.diff(points, axis=0) * scale / steps[:, None]
        orientation += np.einsum('n,ni,nj->ij', steps, directions, directions)
        chord = float(np.linalg.norm((points[-1] - points[0]) * scale))
        loop = e['source'] == e['target']
        row = dict(branch_id=e['id'], raw_edge_id=e['raw_edge_id'], source=e['source'], target=e['target'],
                   component_id=membership[e['source']], length_um=length, chord_um=chord,
                   tortuosity=max(1., length / chord) if chord > 0 and not loop else None,
                   radius_mean_um=float(np.sum((radii[:-1] + radii[1:]) * .5 * steps) / length),
                   radius_min_um=float(radii.min()), radius_max_um=float(radii.max()),
                   radius_median_um=float(np.median(radii)),
                   touches_boundary=bool(np.any((points == 0) | (points == shape - 1))),
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
    summary = dict(schema_version=1, voxel_size_um=scale.tolist(), roi_volume_um3=roi_volume,
                   foreground_volume_um3=float(foreground * np.prod(scale)),
                   foreground_fraction=foreground / mask.size,
                   mask_components_26=int(count), largest_mask_component_fraction=float(sizes.max() / foreground) if foreground else None,
                   mask_euler_characteristic_26=int(euler_number(mask.astype(bool), connectivity=3)),
                   skeleton_voxels=int(np.count_nonzero(skeleton)),
                   raw_nodes=raw.number_of_nodes(), raw_edges=raw.number_of_edges(),
                   raw_independent_cycles=raw.number_of_edges() - raw.number_of_nodes() + nx.number_connected_components(raw),
                   nodes=g.number_of_nodes(), branches=g.number_of_edges(), components=len(components),
                   independent_cycles=g.number_of_edges() - g.number_of_nodes() + len(components),
                   endpoints=sum(d == 1 for d in degrees.values()), branch_nodes=sum(d >= 3 for d in degrees.values()),
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
    return summary, rows, component_rows, branch_graph


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
    summary['analysis_variant'] = 'filled' if summary['preprocessing'].get('fill_enclosed_holes') else 'original'
    (folder / 'statistics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (folder / 'branch_graph.json').write_text(json.dumps(branch_graph, indent=2, allow_nan=False) + '\n')
    write_csv(folder / 'branches.csv', rows, BRANCH_FIELDS)
    write_csv(folder / 'components.csv', components, ['component_id', 'nodes', 'branches', 'independent_cycles', 'length_um'])
    if plots:
        plot_statistics(folder / 'statistics.png', rows, summary)
    print(f"Statistics: {len(rows)} branches, {summary['components']} components, "
          f"{summary['independent_cycles']} independent graph cycles", flush=True)
    for warning in summary['warnings']:
        print('QC:', warning, flush=True)
    return summary, branch_graph


def plot_statistics(path, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(axes.flat[:3], ['length_um', 'radius_mean_um', 'tortuosity'],
                             ['Branch length (µm)', 'EDT radius, branch mean (µm)', 'Tortuosity (length / chord)']):
        values = [r[key] for r in rows if r[key] is not None]
        ax.hist(values, bins=min(40, max(1, int(np.sqrt(len(values))))), color='#147d92')
        ax.set(xlabel=label, ylabel='Branches')
        ax.text(.98, .95, f'n={len(values)}', transform=ax.transAxes, ha='right', va='top')
    degrees = summary['degree_histogram']
    axes[1, 1].bar([int(k) for k in degrees], list(degrees.values()), color='#7956a0')
    axes[1, 1].set(xlabel='Node degree (self-loop contributes 2)', ylabel='Nodes')
    fig.suptitle(summary['source_file'] + f" ({summary['analysis_variant']})" + '\nExploratory statistics after junction-region contraction')
    fig.text(.5, .015, 'Includes short and crop-boundary branches. EDT radius assumes filled vessel masks. See statistics.json QC.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_comparison(folder, summaries):
    # Refresh the comparison from every completed analysis in this output folder.
    # Original and hole-filled variants remain separate rows.
    saved = sorted(Path(folder).glob('*/statistics.json'))
    if saved:
        summaries = [json.loads(p.read_text()) for p in saved]
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        for key in ('length_um', 'radius_mean_um', 'tortuosity'):
            row.update({f'{key}_{stat}': value for stat, value in s[key].items()})
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
    fig, axes = plt.subplots(1, 4, figsize=(14, max(5, len(sources) * .35)), sharey=True)
    variants = [('original', '^', '#a35328'), ('filled', 'o', '#147d92')]
    for ax, key, label in zip(axes, ['length_density_mm_per_mm3', 'radius_mean_um_median',
                                   'tortuosity_median', 'independent_cycles'],
                             ['Length density (mm/mm³)', 'Median branch EDT radius (µm)',
                              'Median branch tortuosity', 'Independent graph cycles']):
        for variant, marker, color in variants:
            selected = [r for r in rows if r.get('analysis_variant', 'original') == variant and r.get(key) is not None]
            ax.scatter([r[key] for r in selected], [positions[r['source_file']] for r in selected],
                       marker=marker, color=color, label=variant, s=30)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(sources)), [s.replace('_Filaments.tif', '') for s in sources])
    axes[0].invert_yaxis()
    axes[-1].legend(title='Mask variant', fontsize=8)
    fig.suptitle('Mouse prostate vasculature — exploratory ROI statistics')
    fig.text(.5, .015, 'ROIs are not independent mice. Original-mask cavities can strongly bias graph measurements; review QC before comparisons.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)
