"""Shortest loop through every vessel, in the style of Ji et al. (2021), Neuron, Fig. 3e.

For each branch e = (u, v) of the vessel graph (junction-to-junction or junction-to-tip
segments; degree-2 points are already merged by the pipeline): remove e, find the shortest
u -> v path weighted by centerline length, and report the loop it closes with e, as a
number of branches and as a perimeter in µm. A branch with no alternative path is a bridge
and gets no loop size. As a sensitivity check the same is done with unweighted paths
(fewest branches). Parallel branches and self-loops keep their own ids (loop sizes 2 and 1).

Loops are always computed on a whole cube's network; results are then assigned to 250 µm
subregions (8 per 500 µm cube) for the per-region distributions, so the subdivision never
creates artificial bridges. Only existing results are read; nothing is overwritten.

Run:  python -m mousevasculature.loop_topology [--results results] [--variant filled+closed]
"""
import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np

# Brain reference reported by Ji et al. (2021); value confirmed by the user, no brain data here.
BRAIN_MEAN_LOOP_BRANCHES = 8.97
BRAIN_REFERENCE_LABEL = 'brain, Ji et al. 2021 (reported mean 8.97 ± 0.53)'
REGION_UM = 250.
ARTIFACT_LOOP_UM = 50.  # matches statistics.SMALL_LOOP_UM
MAX_BIN = 16            # loop sizes above this go into one '>16' bin
MIN_REGION_EDGES = 15   # regions with fewer looped branches are left out of the box plots


def multigraph(edges):
    """MultiGraph from (u, v, key, length_um) tuples."""
    g = nx.MultiGraph()
    for u, v, key, length in edges:
        g.add_edge(u, v, key=key, length_um=float(length))
    return g


def _path_edges(g, nodes):
    """Shortest parallel edge between consecutive nodes: list of (key, length)."""
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        key, data = min(g[a][b].items(), key=lambda kv: kv[1]['length_um'])
        out.append((key, data['length_um']))
    return out


def shortest_cycle(g, u, v, key):
    """Shortest loop through edge (u, v, key), by length and by number of branches.

    Returns dict(branches, length_um, branches_hops, length_hops_um, members) or None for a
    bridge. `members` lists the branch keys of the length-weighted loop.
    """
    own = g.edges[u, v, key]['length_um']
    if u == v:
        return dict(branches=1, length_um=own, branches_hops=1, length_hops_um=own, members=[key])
    data = dict(g.edges[u, v, key])
    g.remove_edge(u, v, key)
    try:
        # Read the detour's branches while e is still removed, so e can't be its own detour.
        path = _path_edges(g, nx.dijkstra_path(g, u, v, weight='length_um'))
        hop_path = _path_edges(g, nx.shortest_path(g, u, v))
    except nx.NetworkXNoPath:
        return None
    finally:
        g.add_edge(u, v, key=key, **data)
    return dict(branches=len(path) + 1, length_um=own + sum(l for _, l in path),
                branches_hops=len(hop_path) + 1, length_hops_um=own + sum(l for _, l in hop_path),
                members=[key] + [k for k, _ in path])


def midpoint(points_um):
    """Point at half the arc length of a centerline."""
    points_um = np.asarray(points_um, float)
    if len(points_um) == 1:
        return points_um[0]
    steps = np.linalg.norm(np.diff(points_um, axis=0), axis=1)
    cumulative = np.concatenate([[0], np.cumsum(steps)])
    i = min(np.searchsorted(cumulative, cumulative[-1] / 2), len(points_um) - 1)
    return points_um[i]


def cube_edges(name, graph, summary, region_um=REGION_UM):
    """Per-branch records for one cube (loops computed on the whole cube)."""
    scale = np.asarray(graph['voxel_size_um'], float)
    g = multigraph((e['source'], e['target'], e['id'], e['length_um']) for e in graph['edges'])
    degree = dict(g.degree())
    rows = []
    for e in graph['edges']:
        s = e['statistics']
        cycle = shortest_cycle(g, e['source'], e['target'], e['id'])
        mid = midpoint(np.asarray(e['points_zyx'], float) * scale)
        region = np.clip((mid // region_um).astype(int), 0, 1)
        rows.append(dict(
            cube=name, branch_id=e['id'], source=e['source'], target=e['target'],
            degree_source=degree[e['source']], degree_target=degree[e['target']],
            length_um=e['length_um'], diameter_um=2 * s['radius_mean_um'] if s.get('radius_mean_um') else None,
            tortuosity=s.get('tortuosity'), touches_boundary=bool(s.get('touches_boundary')),
            short_branch=bool(s.get('short_branch')), bridge=cycle is None,
            loop_branches=cycle and cycle['branches'], loop_length_um=cycle and cycle['length_um'],
            loop_branches_hops=cycle and cycle['branches_hops'], loop_length_hops_um=cycle and cycle['length_hops_um'],
            loop_kind=None if cycle is None else {1: 'self_loop', 2: 'parallel'}.get(cycle['branches'], 'cycle'),
            likely_artifact=bool(cycle and cycle['length_um'] < ARTIFACT_LOOP_UM),
            mid_z_um=mid[0], mid_y_um=mid[1], mid_x_um=mid[2],
            region=f'{name}:z{region[0]}y{region[1]}x{region[2]}'))
    return rows, g


def cube_descriptors(name, g, rows, summary, deletion_fractions, repeats=30, seed=0):
    """Size-normalised network descriptors and a random branch-deletion curve for one cube."""
    volume_mm3 = summary['roi_volume_um3'] / 1e9
    nodes, edges, comps = g.number_of_nodes(), g.number_of_edges(), nx.number_connected_components(g)
    degrees = np.array([d for _, d in g.degree()])
    junctions = degrees[degrees >= 3]
    interior = [r for r in rows if not r['touches_boundary']]
    out = dict(
        cube=name, nodes=nodes, branches=edges, components=comps, mean_degree=float(degrees.mean()),
        degree2_nodes=int((degrees == 2).sum()), cyclomatic=edges - nodes + comps,
        cyclomatic_per_mm3=(edges - nodes + comps) / volume_mm3, cyclomatic_per_junction=(edges - nodes + comps) / max(1, len(junctions)),
        nodes_per_mm3=nodes / volume_mm3, branches_per_mm3=edges / volume_mm3,
        degree1_fraction=float((degrees == 1).mean()), degree3_of_junctions=float((junctions == 3).mean()),
        degree4plus_of_junctions=float((junctions >= 4).mean()),
        looped_fraction=np.mean([not r['bridge'] for r in rows]),
        looped_length_fraction=sum(r['length_um'] for r in rows if not r['bridge']) / sum(r['length_um'] for r in rows),
        bridge_fraction=np.mean([r['bridge'] for r in rows]),
        bridge_fraction_interior=np.mean([r['bridge'] for r in interior]) if interior else None,
        boundary_branch_fraction=np.mean([r['touches_boundary'] for r in rows]),
        mean_loop_branches=np.mean([r['loop_branches'] for r in rows if not r['bridge']]),
        median_loop_length_um=np.median([r['loop_length_um'] for r in rows if not r['bridge']]))
    # Random deletion: fraction of nodes left in the largest connected piece.
    rng = np.random.default_rng(seed)
    keys = list(g.edges(keys=True))
    start = max(len(c) for c in nx.connected_components(g))
    curve = []
    for f in deletion_fractions:
        values = []
        for _ in range(repeats):
            h = g.copy()
            drop = rng.choice(len(keys), int(round(f * len(keys))), replace=False)
            h.remove_edges_from([keys[i] for i in drop])
            values.append(max(len(c) for c in nx.connected_components(h)) / start)
        curve.append(float(np.mean(values)))
    return out, curve


def load_cubes(results, variant_suffix):
    cubes = {}
    for folder in sorted(Path(results).glob(f'*_Filaments{variant_suffix}')):
        try:
            graph = json.loads((folder / 'branch_graph.json').read_text())
            summary = json.loads((folder / 'statistics.json').read_text())
        except (OSError, ValueError):
            continue
        cubes[folder.name.split('_Filaments')[0]] = (graph, summary)
    return cubes


def bin_label(n):
    return str(n) if n <= MAX_BIN else f'>{MAX_BIN}'


def region_distributions(rows, field='loop_branches'):
    """region -> probability per loop-size bin (1..MAX_BIN, >MAX_BIN), for regions with enough branches."""
    by_region = {}
    for r in rows:
        if not r['bridge']:
            by_region.setdefault(r['region'], []).append(min(r[field], MAX_BIN + 1))
    out = {}
    for region, sizes in by_region.items():
        if len(sizes) >= MIN_REGION_EDGES:
            counts = np.bincount(sizes, minlength=MAX_BIN + 2)[1:]
            out[region] = counts / counts.sum()
    return out


def cube_bootstrap(rows, statistic, n=2000, seed=0):
    """95% interval of a statistic, resampling whole cubes (branches within a cube are not independent)."""
    rng = np.random.default_rng(seed)
    by_cube = {}
    for r in rows:
        by_cube.setdefault(r['cube'], []).append(r)
    names = list(by_cube)
    values = [statistic([r for c in rng.choice(names, len(names)) for r in by_cube[c]]) for _ in range(n)]
    return np.percentile(values, [2.5, 97.5])


def contract_degree_two(g):
    """Merge degree-2 nodes into one branch (summing lengths), as the pipeline does."""
    g = g.copy()
    next_key = max((k for *_, k in g.edges(keys=True)), default=0) + 1
    changed = True
    while changed:
        changed = False
        for n in list(g.nodes):
            if n in g and g.degree(n) == 2 and not g.has_edge(n, n):
                (a, l1), (b, l2) = [(v, d['length_um']) for _, v, d in g.edges(n, data=True)]
                g.remove_node(n)
                g.add_edge(a, b, key=next_key, length_um=l1 + l2)
                next_key += 1
                changed = True
    g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
    return g


def window_bias(cubes, sizes=(200, 250, 300, 375, 450, 500), cube_um=500.):
    """Loop statistics recomputed inside smaller windows of each cube.

    Branches are kept when their midpoint lies in the window; junctions left with two
    branches are merged again. Shows how strongly the window size biases loop counts, and
    whether 500 µm cubes are large enough to have converged.
    """
    import itertools
    results = []
    for size in sizes:
        corners = [(0., 0., 0.)] if size >= cube_um else list(itertools.product((0., cube_um - size), repeat=3))
        means, looped = [], []
        for graph, _ in cubes.values():
            scale = np.asarray(graph['voxel_size_um'], float)
            mids = {e['id']: midpoint(np.asarray(e['points_zyx'], float) * scale) for e in graph['edges']}
            for corner in corners:
                low, high = np.asarray(corner), np.asarray(corner) + size
                g = multigraph((e['source'], e['target'], e['id'], e['length_um']) for e in graph['edges']
                               if np.all(mids[e['id']] >= low) and np.all(mids[e['id']] < high))
                if size < cube_um:
                    g = contract_degree_two(g)
                if not g.number_of_edges():
                    continue
                loops = [c['branches'] for u, v, k in list(g.edges(keys=True)) if (c := shortest_cycle(g, u, v, k))]
                looped.append(len(loops) / g.number_of_edges())
                if loops:
                    means.append(np.mean(loops))
        results.append(dict(window_um=size, windows=len(looped), mean_loop_branches=float(np.mean(means)),
                            sd_loop_branches=float(np.std(means, ddof=1)), looped_fraction=float(np.mean(looped))))
    return results


def figure_d(bias, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout='constrained')
    w = [b['window_um'] for b in bias]
    m = np.array([b['mean_loop_branches'] for b in bias])
    sd = np.array([b['sd_loop_branches'] for b in bias])
    ax.errorbar(w, m, yerr=sd, marker='o', color='#08519c', capsize=3, label='prostate: mean ± SD over windows')
    ax.axhline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', label='brain, Ji et al. 2021 (reported mean;\nwhole network, not windowed)')
    twin = ax.twinx()
    twin.plot(w, [b['looped_fraction'] for b in bias], marker='s', color='#31a354', linestyle='--', label='fraction of branches on a loop')
    twin.set_ylim(0, 1), twin.set_ylabel('fraction on a loop', color='#31a354', fontsize=9)
    ax.set_xlabel('window side (µm); loops computed inside the window only')
    ax.set_ylabel('mean branches in shortest loop')
    ax.set_title('Window-size bias: loop counts have not converged at 500 µm', fontsize=9)
    lines = ax.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    ax.legend(lines[0] + extra[0], lines[1] + extra[1], fontsize=7, frameon=False, loc='lower right')
    style(ax)
    save(fig, out, 'figD_window_size_bias')


# ---- figures -------------------------------------------------------------------------
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def figure_a(rows, out):
    import matplotlib.pyplot as plt
    looped = [r for r in rows if not r['bridge']]
    regions = region_distributions(rows)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    bins = np.arange(1, MAX_BIN + 2)
    data = np.array(list(regions.values()))
    ax.boxplot([data[:, i] for i in range(len(bins))], positions=bins, widths=.6, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor='#9ecae1', edgecolor='#3182bd'),
               medianprops=dict(color='#08519c'), whiskerprops=dict(color='#3182bd'), capprops=dict(color='#3182bd'))
    mean = np.mean([r['loop_branches'] for r in looped])
    ax.axvline(mean, color='#08519c', linestyle='--', linewidth=1.2, label=f'prostate mean {mean:.2f} (computed)')
    ax.axvline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', linewidth=1.5, label=BRAIN_REFERENCE_LABEL)
    ax.set_xticks(bins, [bin_label(b) for b in bins])
    ax.set_xlabel('branches in shortest loop through a branch (length-weighted)')
    ax.set_ylabel('probability per region')
    ax.set_title(f'Prostate: {len(regions)} regions of {REGION_UM:.0f} µm (≥{MIN_REGION_EDGES} looped branches each), '
                 f'15 cubes', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    style(ax)
    for field, rows_, label, colour in [
            ('loop_branches', looped, 'length-weighted (all looped branches)', '#08519c'),
            ('loop_branches_hops', looped, 'fewest branches (sensitivity)', '#6baed6'),
            ('loop_branches', [r for r in looped if not r['touches_boundary']], 'length-weighted, interior branches only', '#31a354'),
            ('loop_branches', [r for r in looped if not r['likely_artifact']], f'length-weighted, loops ≥ {ARTIFACT_LOOP_UM:.0f} µm', '#756bb1')]:
        sizes = np.minimum([r[field] for r in rows_], MAX_BIN + 1)
        p = np.bincount(sizes, minlength=MAX_BIN + 2)[1:] / len(sizes)
        bx.step(bins, p, where='mid', color=colour, label=f'{label}: mean {np.mean([r[field] for r in rows_]):.2f}, n={len(sizes)}')
    bx.axvline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', linewidth=1.5, label='brain, Ji et al. 2021 (reported mean)')
    bx.set_xticks(bins, [bin_label(b) for b in bins])
    bx.set_xlabel('branches in shortest loop')
    bx.set_ylabel('probability (pooled)')
    bx.set_title('Pooled prostate distributions and sensitivity checks', fontsize=9)
    bx.legend(fontsize=6.5, frameon=False)
    style(bx)
    save(fig, out, 'figA_shortest_loop_branches')


def figure_b(cubes, rows, out):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm
    by_key = {(r['cube'], r['branch_id']): r for r in rows}
    names = list(cubes)
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(names) / cols)), cols, figsize=(15, 9.6), layout='constrained')
    cmap = plt.get_cmap('viridis_r', MAX_BIN - 1)
    norm = BoundaryNorm(np.arange(1.5, MAX_BIN + 1.5), cmap.N)
    for ax, name in zip(axes.flat, names):
        graph, _ = cubes[name]
        scale = np.asarray(graph['voxel_size_um'], float)
        grey, coloured, values = [], [], []
        for e in graph['edges']:
            xy = (np.asarray(e['points_zyx'], float) * scale)[:, [2, 1]]
            if len(xy) < 2:
                continue
            r = by_key[(name, e['id'])]
            if r['bridge']:
                grey.append(xy)
            else:
                coloured.append(xy)
                values.append(min(r['loop_branches'], MAX_BIN))
        ax.add_collection(LineCollection(grey, colors='#c8c8c8', linewidths=.6))
        lines = LineCollection(coloured, cmap=cmap, norm=norm, linewidths=1.)
        lines.set_array(np.asarray(values, float))
        ax.add_collection(lines)
        ax.set_xlim(0, 500), ax.set_ylim(500, 0), ax.set_aspect('equal')
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(name, fontsize=8)
        for edge in (250,):
            ax.axvline(edge, color='#eeeeee', linewidth=.5, zorder=0), ax.axhline(edge, color='#eeeeee', linewidth=.5, zorder=0)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    bar = fig.colorbar(lines, ax=axes, shrink=.6, ticks=range(2, MAX_BIN + 1))
    bar.set_label(f'branches in shortest loop ({MAX_BIN} = {MAX_BIN} or more); grey = bridge')
    fig.suptitle('Shortest-loop size per branch, top view (X–Y projection of 3D 500 µm cubes; faint lines = 250 µm regions)',
                 fontsize=10)
    save(fig, out, 'figB_spatial_loop_size')


def figure_c(rows, descriptors, curves, fractions, out):
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, spearmanr
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout='constrained')
    looped = [r for r in rows if not r['bridge']]

    ax = axes[0, 0]  # degree distribution of junctions
    labels = ['3', '4', '5+']
    per_cube = np.array([[d['degree3_of_junctions']] for d in descriptors])
    degrees = {}
    for r in rows:
        degrees[(r['cube'], r['source'])] = r['degree_source']
        degrees[(r['cube'], r['target'])] = r['degree_target']
    pooled = np.array(list(degrees.values()))
    junction = pooled[pooled >= 3]
    shares = [(junction == 3).mean(), (junction == 4).mean(), (junction >= 5).mean()]
    ax.bar(labels, shares, color='#6baed6')
    ax.scatter(np.zeros(len(per_cube)) + np.random.default_rng(0).uniform(-.15, .15, len(per_cube)), per_cube[:, 0],
               s=10, color='#08519c', zorder=3, label='per cube (degree 3)')
    ax.set_ylabel('fraction of junctions'), ax.set_xlabel('junction degree')
    ax.set_title(f'Junction degree (endpoints: {(pooled == 1).mean():.0%} of all nodes)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = axes[0, 1]  # looped vs bridge per cube
    order = np.argsort([d['looped_fraction'] for d in descriptors])
    names = [descriptors[i]['cube'] for i in order]
    looped_f = np.array([descriptors[i]['looped_fraction'] for i in order])
    ax.barh(names, looped_f, color='#3182bd', label='on a loop')
    ax.barh(names, 1 - looped_f, left=looped_f, color='#c8c8c8', label='bridge')
    ax.scatter([1 - descriptors[i]['bridge_fraction_interior'] for i in order], names, marker='|', s=60, color='k',
               label='on a loop, interior branches only')
    ax.set_xlim(0, 1), ax.set_xlabel('fraction of branches')
    ax.set_title('Branches with an alternative path', fontsize=9)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(.4, -.1), ncol=3)
    style(ax)

    ax = axes[0, 2]  # perimeter
    perimeters = np.array([r['loop_length_um'] for r in looped])
    interior = np.array([r['loop_length_um'] for r in looped if not r['touches_boundary']])
    bins = np.geomspace(max(5, perimeters.min()), perimeters.max(), 35)
    ax.hist(perimeters, bins=bins, density=True, color='#9ecae1', label=f'all looped, median {np.median(perimeters):.0f} µm')
    ax.hist(interior, bins=bins, density=True, histtype='step', color='#31a354', label=f'interior only, median {np.median(interior):.0f} µm')
    ax.axvline(ARTIFACT_LOOP_UM, color='#756bb1', linestyle=':', label=f'{ARTIFACT_LOOP_UM:.0f} µm artifact threshold')
    ax.set_xscale('log'), ax.set_yticks([])
    ax.set_xlabel('shortest-loop perimeter (µm)'), ax.set_title('Shortest-loop perimeter per branch', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    style(ax)

    ax = axes[1, 0]  # random deletion
    curves = np.array(curves)
    ax.fill_between(fractions, curves.min(0), curves.max(0), color='#9ecae1', alpha=.6, label='range over cubes')
    ax.plot(fractions, curves.mean(0), color='#08519c', label='mean of 15 cubes')
    ax.set_xlabel('fraction of branches removed at random'), ax.set_ylabel('largest piece / largest piece before removal\n(nodes)', fontsize=8)
    ax.set_title('Random branch deletion (analogous to Ji et al. Fig. 3h;\nmeasure not verified against the paper)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    ax = axes[1, 1]  # loop size by diameter
    diam = np.array([r['diameter_um'] for r in looped], float)
    sizes = np.array([r['loop_branches'] for r in looped])
    low, high = np.percentile(diam, [33.3, 66.7])
    groups = [(diam <= low, f'thinnest third (≤{low:.1f} µm)', '#fdae6b'),
              ((diam > low) & (diam < high), 'middle third', '#bdbdbd'),
              (diam >= high, f'widest third (≥{high:.1f} µm)', '#e6550d')]
    b = np.arange(1, MAX_BIN + 2)
    for mask, label, colour in groups:
        p = np.bincount(np.minimum(sizes[mask], MAX_BIN + 1), minlength=MAX_BIN + 2)[1:] / mask.sum()
        ax.step(b, p, where='mid', color=colour, label=f'{label}: mean {sizes[mask].mean():.2f}')
    test = mannwhitneyu(sizes[groups[0][0]], sizes[groups[2][0]])
    rho = spearmanr(diam, sizes)
    ax.set_xticks(b[::2], [bin_label(x) for x in b[::2]])
    ax.set_xlabel('branches in shortest loop'), ax.set_ylabel('probability')
    ax.set_title(f'Loop size by vessel diameter (Spearman ρ = {rho[0]:.2f};\nthinnest vs widest Mann–Whitney p = {test.pvalue:.2g})',
                 fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    style(ax)

    ax = axes[1, 2]  # region loop size vs local length density
    by_region = {}
    for r in rows:
        by_region.setdefault(r['region'], []).append(r)
    xs, ys = [], []
    region_mm3 = (REGION_UM / 1000) ** 3
    for region_rows in by_region.values():
        loops_ = [r['loop_branches'] for r in region_rows if not r['bridge']]
        if len(loops_) >= MIN_REGION_EDGES:
            xs.append(sum(r['length_um'] for r in region_rows) / 1000 / region_mm3)
            ys.append(np.mean(loops_))
    rho = spearmanr(xs, ys)
    ax.scatter(xs, ys, s=12, color='#3182bd')
    ax.axhline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', label='brain, Ji et al. 2021 (reported mean)')
    ax.set_xlabel('local length density (mm/mm³)')
    ax.set_ylabel('mean branches in shortest loop')
    ax.set_title(f'{len(xs)} regions of {REGION_UM:.0f} µm: Spearman ρ = {rho[0]:.2f} (p = {rho[1]:.2g})', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    save(fig, out, 'figC_topology_comparisons')
    return dict(diameter_spearman=float(spearmanr(diam, sizes)[0]), diameter_mannwhitney_p=float(test.pvalue),
                diameter_terciles_um=[float(low), float(high)],
                diameter_tercile_means=[float(sizes[m].mean()) for m, _, _ in groups],
                density_spearman=float(rho[0]), density_p=float(rho[1]), regions_used=len(xs))


def save(fig, out, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out) / f'{stem}.{ext}', dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(results='results', variant_suffix='_filled_closed', out=None):
    import matplotlib
    matplotlib.use('Agg')
    out = Path(out or Path(results) / 'ji2021_comparison')
    out.mkdir(parents=True, exist_ok=True)
    cubes = load_cubes(results, variant_suffix)
    if not cubes:
        raise SystemExit(f'No saved results matching *_Filaments{variant_suffix} in {results}')
    fractions = np.round(np.arange(0, .95, .05), 2)
    rows, descriptors, curves = [], [], []
    for name, (graph, summary) in cubes.items():
        cube_rows, g = cube_edges(name, graph, summary)
        d, curve = cube_descriptors(name, g, cube_rows, summary, fractions)
        rows += cube_rows
        descriptors.append(d)
        curves.append(curve)
    looped = [r for r in rows if not r['bridge']]
    regions = region_distributions(rows)
    region_means = [np.mean([r['loop_branches'] for r in rows if r['region'] == k and not r['bridge']]) for k in regions]
    mean_ci = cube_bootstrap(rows, lambda rs: np.mean([r['loop_branches'] for r in rs if not r['bridge']]))
    figure_a(rows, out)
    figure_b(cubes, rows, out)
    extra = figure_c(rows, descriptors, curves, fractions, out)
    bias = window_bias(cubes)
    figure_d(bias, out)
    write_csv(out / 'window_bias.csv', bias)
    write_csv(out / 'branch_loops.csv', rows)
    write_csv(out / 'cube_descriptors.csv', descriptors)
    write_csv(out / 'random_deletion.csv', [dict(cube=d['cube'], **{f'removed_{f}': c for f, c in zip(fractions, curve)})
                                            for d, curve in zip(descriptors, curves)])
    sizes = np.array([r['loop_branches'] for r in looped])
    hops = np.array([r['loop_branches_hops'] for r in looped])
    interior = np.array([r['loop_branches'] for r in looped if not r['touches_boundary']])
    clean = np.array([r['loop_branches'] for r in looped if not r['likely_artifact']])
    summary = dict(
        cubes=len(cubes), branches=len(rows), looped_branches=len(looped), bridges=len(rows) - len(looped),
        mean_loop_branches=float(sizes.mean()), median_loop_branches=float(np.median(sizes)),
        mean_loop_branches_cube_bootstrap_95ci=[float(x) for x in mean_ci],
        region_mean_loop_branches_mean_sd=[float(np.mean(region_means)), float(np.std(region_means, ddof=1))],
        regions=len(regions), region_size_um=REGION_UM,
        mean_loop_branches_fewest_hops=float(hops.mean()), hop_differs_fraction=float(np.mean(hops != sizes)),
        mean_loop_branches_interior=float(interior.mean()), mean_loop_branches_without_artifacts=float(clean.mean()),
        artifact_loops=int(sum(r['likely_artifact'] for r in looped)),
        self_loops=int(sum(r['loop_kind'] == 'self_loop' for r in looped)),
        parallel=int(sum(r['loop_kind'] == 'parallel' for r in looped)),
        median_loop_length_um=float(np.median([r['loop_length_um'] for r in looped])),
        looped_fraction=len(looped) / len(rows),
        looped_fraction_interior=float(np.mean([not r['bridge'] for r in rows if not r['touches_boundary']])),
        boundary_branch_fraction=float(np.mean([r['touches_boundary'] for r in rows])),
        degree2_nodes=int(sum(d['degree2_nodes'] for d in descriptors)),
        brain_reference=dict(mean_loop_branches=BRAIN_MEAN_LOOP_BRANCHES, source=BRAIN_REFERENCE_LABEL,
                             note='published value, confirmed by the user; not recomputed (no brain data)'),
        window_bias=bias, **extra)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary, out


def main(argv=None):
    from .pipeline import VARIANTS
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--results', default='results')
    parser.add_argument('--variant', default='filled+closed', choices=list(VARIANTS))
    parser.add_argument('--out', default=None, help='output folder (default: <results>/ji2021_comparison)')
    args = parser.parse_args(argv)
    summary, out = run(args.results, VARIANTS[args.variant], args.out)
    print(json.dumps(summary, indent=2))
    print(f'Figures and tables in {out}')


if __name__ == '__main__':
    main()
