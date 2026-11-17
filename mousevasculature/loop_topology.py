"""Shortest loop through every vessel, in the style of Ji et al. (2021), Neuron, Fig. 3e.

Reference: Ji X, Ferreira T, Friedman B, Liu R, Liechty H, Bas E, Chandrashekar J,
Kleinfeld D (2021). Brain microvasculature has a common topology with local differences in
geometry that match metabolic load. Neuron 109(7):1168-1187.e13.
https://doi.org/10.1016/j.neuron.2021.02.006

Definitions follow the paper's STAR Methods:
- Loop size (Fig. 3E, p. e9): "number of vessel segments in the shortest loop measured by
  segment length", per branch. For each branch e = (u, v) (degree-2 points are already
  merged by the pipeline): remove e, find the shortest u -> v path weighted by centerline
  length, and report the loop it closes with e, in branches and µm. A branch with no
  alternative path is a bridge. Fewest-branch loops are a sensitivity check.
- Box-plot unit (Fig. 3E): per-240-µm-cube distributions. Loops are computed on the whole
  graph (whole brain in the paper; whole 500 µm cube here) and then assigned to 240 µm
  regions, so the subdivision never creates artificial bridges.
- Graph refinement (Algorithm 1, p. e5): iteratively delete free-ended branches <= 2 µm,
  self-loops <= 30 µm, and one branch of every two-branch loop with a branch <= 15 µm. The
  paper deletes the dimmer branch; binary masks carry no intensity, so the shorter one goes.
  Applied here as a sensitivity variant (`ji_refine`); the paper's classifier steps are not.
- Fig. 3H (p. e10): remove a fraction p of branches at random, record the fraction of nodes
  in the largest connected component s(p), 500 repeats; the threshold is the p maximising
  |ds/dp|. Brain: networks fall apart after 0.433-0.447 of branches are removed.

Only existing results are read; nothing is overwritten.

Run:  python -m mousevasculature.loop_topology [--results results] [--variant filled+closed]
"""
import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np

# Brain reference values from Ji et al. (2021); no brain data are included here.
BRAIN_MEAN_LOOP_BRANCHES = 8.97   # p. 1175: 8.97 ± 0.53 (mean ± SE) branches in the shortest loop
BRAIN_LOOP_SE = 0.53
BRAIN_REFERENCE_LABEL = 'brain, Ji et al. 2021 (mean 8.97 ± 0.53 SE)'
BRAIN_PERCOLATION_REMOVED = (0.433, 0.447)  # p. 1176, Fig. 3H: fraction of branches removed at breakdown
CITATION = ('Ji et al. (2021) Neuron 109:1168-1187. Brain microvasculature has a common topology with local '
            'differences in geometry that match metabolic load. doi:10.1016/j.neuron.2021.02.006')
# Fraction of branches whose shortest loop has k branches in brain, Fig. 3E. Only k = 2 is
# stated in the text (0.011, median); k = 3 and 4 are ranges READ BY EYE from the box-plot
# medians and must be replaced by exact values (digitised or from the authors) before citing.
# Brain branches are nearly all on loops, so these approximate fractions of all branches.
BRAIN_FIG3E_BY_EYE = {2: (0.011, 0.011), 3: (0.015, 0.03), 4: (0.03, 0.05)}
SMALL_LOOP_SETS = {'≤3': (1, 3), '3–4': (3, 4), '≤4': (1, 4)}
REGION_UM = 240.   # Ji et al. analyse 240-µm cubes
CUBE_UM = 500.     # prostate ROI edge (calibration.json)
ARTIFACT_LOOP_UM = 50.  # matches statistics.SMALL_LOOP_UM
# Ji et al. Algorithm 1 thresholds (µm)
JI_STUB_UM, JI_SELF_LOOP_UM, JI_PARALLEL_UM = 2., 30., 15.
PERCOLATION_REPEATS = 500
MAX_BIN = 16            # loop sizes above this go into one '>16' bin
MIN_REGION_EDGES = 15   # regions with fewer looped branches are left out of the box plots
MIN_WINDOW_EDGES = 10   # windows with fewer branches are left out of the small-loop fractions


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


def region_index(mid_um, region_um=REGION_UM, cube_um=CUBE_UM):
    """Index (z, y, x) of the centred region grid cell holding a point, or None in the margin.

    A 500 µm cube holds 2 x 2 x 2 regions of 240 µm; the grid is centred, leaving a 10 µm
    margin on every face whose branches count in pooled statistics but in no region.
    """
    n = int(cube_um // region_um)
    offset = (cube_um - n * region_um) / 2
    idx = np.floor((np.asarray(mid_um, float) - offset) / region_um).astype(int)
    return tuple(int(i) for i in idx) if np.all((idx >= 0) & (idx < n)) else None


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
        region = region_index(mid, region_um)
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
            loop_members=' '.join(map(str, sorted(cycle['members']))) if cycle else '',
            mid_z_um=mid[0], mid_y_um=mid[1], mid_x_um=mid[2],
            region=None if region is None else f'{name}:z{region[0]}y{region[1]}x{region[2]}'))
    return rows, g


def ji_refine(g):
    """Ji et al. (2021) Algorithm 1 on a branch graph; returns (refined copy, deletion counts).

    Repeats until nothing changes: delete free-ended branches <= 2 µm; delete self-loops
    <= 30 µm; for each pair of nodes joined by two or more branches where one is <= 15 µm,
    delete one (the shorter: the paper drops the dimmer one, and binary masks carry no
    intensity); then merge degree-2 nodes, as the paper's graph update does.
    """
    g = g.copy()
    counts = dict(stubs=0, self_loops=0, parallel=0)
    while True:
        changed = 0
        for u, v, k, d in list(g.edges(keys=True, data=True)):
            if u != v and min(g.degree(u), g.degree(v)) == 1 and d['length_um'] <= JI_STUB_UM:
                g.remove_edge(u, v, k)
                counts['stubs'] += 1
                changed += 1
        for u, v, k, d in list(g.edges(keys=True, data=True)):
            if u == v and d['length_um'] <= JI_SELF_LOOP_UM:
                g.remove_edge(u, v, k)
                counts['self_loops'] += 1
                changed += 1
        for u, v in {tuple(sorted((a, b), key=str)) for a, b in g.edges() if a != b}:
            parallel = sorted(g[u][v].items(), key=lambda kv: kv[1]['length_um'])
            if len(parallel) >= 2 and parallel[0][1]['length_um'] <= JI_PARALLEL_UM:
                g.remove_edge(u, v, parallel[0][0])
                counts['parallel'] += 1
                changed += 1
        g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
        g = contract_degree_two(g)
        if not changed:
            return g, counts


def percolation_curve(g, fractions, repeats=PERCOLATION_REPEATS, seed=0):
    """Ji et al. Fig. 3H: mean fraction of all nodes in the largest component after removing
    a fraction of branches at random."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    index = {n: i for i, n in enumerate(g.nodes)}
    pairs = np.array([(index[u], index[v]) for u, v in g.edges()], int).reshape(-1, 2)
    n_nodes, n_edges = len(index), len(pairs)
    rng = np.random.default_rng(seed)
    curve = []
    for f in fractions:
        keep_n = n_edges - int(round(f * n_edges))
        values = np.empty(repeats)
        for r in range(repeats):
            kept = pairs[rng.permutation(n_edges)[:keep_n]]
            adjacency = coo_matrix((np.ones(len(kept)), (kept[:, 0], kept[:, 1])), shape=(n_nodes, n_nodes))
            _, labels = connected_components(adjacency, directed=False)
            values[r] = np.bincount(labels).max() / n_nodes
        curve.append(float(values.mean()))
    return curve


def percolation_threshold(fractions, curve):
    """Removed fraction where the mean curve falls fastest (max |ds/dp|), as in Ji et al."""
    fractions, curve = np.asarray(fractions, float), np.asarray(curve, float)
    slopes = np.abs(np.diff(curve) / np.diff(fractions))
    i = int(np.argmax(slopes))
    return float((fractions[i] + fractions[i + 1]) / 2)


def cube_descriptors(name, g, rows, summary, deletion_fractions, repeats=PERCOLATION_REPEATS, seed=0):
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
    # Ji et al. Fig. 3H: fraction of all nodes in the largest component after random removal.
    curve = percolation_curve(g, deletion_fractions, repeats, seed)
    out['percolation_threshold_removed'] = percolation_threshold(deletion_fractions, curve)
    return out, curve


def brain_small_fraction(lo, hi):
    """(low, high) brain fraction of branches on loops of lo..hi branches, from BRAIN_FIG3E_BY_EYE."""
    keys = [k for k in BRAIN_FIG3E_BY_EYE if lo <= k <= hi]
    return sum(BRAIN_FIG3E_BY_EYE[k][0] for k in keys), sum(BRAIN_FIG3E_BY_EYE[k][1] for k in keys)


def small_fractions(sizes):
    """Fraction of ALL branches (bridges count as not small) on a shortest loop in each SMALL_LOOP_SETS range."""
    s = np.array([0 if x is None else x for x in sizes])
    return {name: float(np.mean((s >= lo) & (s <= hi))) for name, (lo, hi) in SMALL_LOOP_SETS.items()}


def load_annotations(folder):
    """Branch-id sets of loops annotated in the explorer, by label ('real', 'artifact', 'unsure')."""
    path = Path(folder) / 'loop_annotations.csv'
    out = {}
    if path.exists():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                out.setdefault(row['label'], []).append(frozenset(int(b) for b in row['branch_ids'].split()))
    return out


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
        if not r['bridge'] and r['region'] is not None:
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


def window_bias(cubes, sizes=(200, 240, 300, 375, 450, 500), cube_um=CUBE_UM):
    """Loop statistics recomputed inside smaller windows of each cube.

    Branches are kept when their midpoint lies in the window; junctions left with two
    branches are merged again. Shows how strongly the window size biases loop counts, and
    whether 500 µm cubes are large enough to have converged.
    """
    import itertools
    results = []
    for size in sizes:
        corners = [(0., 0., 0.)] if size >= cube_um else list(itertools.product((0., cube_um - size), repeat=3))
        means, looped, small = [], [], []
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
                cycles = [shortest_cycle(g, u, v, k) for u, v, k in list(g.edges(keys=True))]
                loops = [c['branches'] for c in cycles if c]
                looped.append(len(loops) / g.number_of_edges())
                if loops:
                    means.append(np.mean(loops))
                if g.number_of_edges() >= MIN_WINDOW_EDGES:
                    small.append(small_fractions([c and c['branches'] for c in cycles]))
        results.append(dict(window_um=size, windows=len(looped), mean_loop_branches=float(np.mean(means)),
                            sd_loop_branches=float(np.std(means, ddof=1)), looped_fraction=float(np.mean(looped)),
                            small_windows=len(small),
                            **{f'small_{name}': float(np.mean([s[name] for s in small])) for name in SMALL_LOOP_SETS}))
    return results


def figure_d(bias, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout='constrained')
    w = [b['window_um'] for b in bias]
    m = np.array([b['mean_loop_branches'] for b in bias])
    sd = np.array([b['sd_loop_branches'] for b in bias])
    ax.errorbar(w, m, yerr=sd, marker='o', color='#08519c', capsize=3, label='prostate: mean ± SD over windows')
    ax.axhspan(BRAIN_MEAN_LOOP_BRANCHES - BRAIN_LOOP_SE, BRAIN_MEAN_LOOP_BRANCHES + BRAIN_LOOP_SE, color='#d95f02', alpha=.12, lw=0)
    ax.axhline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', label='brain, Ji et al. 2021 (mean ± SE;\nwhole-brain graph, not windowed)')
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


def review_small_loops(rows, annotations):
    """Small-loop fractions before and after removing loops annotated as artifacts in the explorer.

    A branch's shortest loop counts as reviewed when its branch set equals an annotated loop.
    Branches whose shortest loop was marked 'artifact' are counted as not on a small loop
    (their true shortest loop is unknown but larger).
    """
    status = []
    for r in rows:
        members = frozenset(int(b) for b in r['loop_members'].split()) if r['loop_members'] else None
        labels = annotations.get(r['cube'], {})
        status.append(next((label for label, sets in labels.items() if members in sets), None) if members else None)
    small = [r['loop_branches'] is not None and r['loop_branches'] <= max(hi for _, hi in SMALL_LOOP_SETS.values())
             for r in rows]
    reviewed = sum(1 for s, sm in zip(status, small) if sm and s)
    cleaned = [None if s == 'artifact' else r['loop_branches'] for r, s in zip(rows, status)]
    return dict(
        annotated_loops={label: sum(len(c.get(label, [])) for c in annotations.values()) for label in ('real', 'artifact', 'unsure')},
        small_loop_branches=int(sum(small)), small_loop_branches_reviewed=int(reviewed),
        branches_on_artifact_loops=int(sum(s == 'artifact' for s in status)),
        before=small_fractions([r['loop_branches'] for r in rows]),
        after_removing_artifacts=small_fractions(cleaned))


# ---- figures -------------------------------------------------------------------------
def figure_e(bias, review, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout='constrained')
    w = [b['window_um'] for b in bias]
    colours = {'≤3': '#08519c', '3–4': '#31a354', '≤4': '#756bb1'}
    for i, (name, (lo, hi)) in enumerate(SMALL_LOOP_SETS.items()):
        ax.plot(w, [b[f'small_{name}'] for b in bias], marker='o', color=colours[name], label=f'prostate, loops of {name} branches')
        low, high = brain_small_fraction(lo, hi)
        ax.add_patch(plt.Rectangle((520 + 18 * i, low), 14, max(high - low, .002), color=colours[name], alpha=.45, lw=0))
        if review['small_loop_branches_reviewed']:
            ax.scatter([500], [review['after_removing_artifacts'][name]], marker='x', s=50, color=colours[name], zorder=4)
    ax.text(538, ax.get_ylim()[1] * .02 + .002, 'brain\n(by eye,\nFig. 3E)', fontsize=6.5, ha='center', va='bottom')
    ax.set_xlim(180, 590)
    ax.set_xticks(w)
    ax.set_xlabel('window side (µm); loops computed inside the window only\n'
                  '(< 500 µm: 8 overlapping corner windows per cube; 500 µm: the 15 whole cubes)', fontsize=8.5)
    ax.set_ylabel('fraction of ALL branches on a loop this small')
    reviewed = review['small_loop_branches_reviewed']
    note = (f"× = whole cube after removing loops marked artifact ({reviewed}/{review['small_loop_branches']} small-loop branches reviewed)"
            if reviewed else 'no explorer annotations yet: review 3–4-branch loops to test for segmentation artifacts')
    ax.set_title('Small loops: prostate values rise with window size, so 500 µm values are lower bounds\n' + note, fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    style(ax)
    save(fig, out, 'figE_small_loops')
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def figure_a(rows, out, refined_sizes=None):
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
    ax.axvspan(BRAIN_MEAN_LOOP_BRANCHES - BRAIN_LOOP_SE, BRAIN_MEAN_LOOP_BRANCHES + BRAIN_LOOP_SE, color='#d95f02', alpha=.12, lw=0)
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
    if refined_sizes is not None and len(refined_sizes):
        sizes = np.minimum(refined_sizes, MAX_BIN + 1)
        p = np.bincount(sizes, minlength=MAX_BIN + 2)[1:] / len(sizes)
        bx.step(bins, p, where='mid', color='#e7298a', linestyle='--',
                label=f'graph refined as Ji et al. Alg. 1: mean {np.mean(refined_sizes):.2f}, n={len(sizes)}')
    bx.axvline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', linewidth=1.5, label='brain, Ji et al. 2021 (mean)')
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
        for edge in (10, 250, 490):
            ax.axvline(edge, color='#eeeeee', linewidth=.5, zorder=0), ax.axhline(edge, color='#eeeeee', linewidth=.5, zorder=0)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    bar = fig.colorbar(lines, ax=axes, shrink=.6, ticks=range(2, MAX_BIN + 1))
    bar.set_label(f'branches in shortest loop ({MAX_BIN} = {MAX_BIN} or more); grey = bridge')
    fig.suptitle('Shortest-loop size per branch, top view (X–Y projection of 3D 500 µm cubes; faint lines = 240 µm regions)',
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
    thresholds = [d['percolation_threshold_removed'] for d in descriptors]
    ax.fill_between(fractions, curves.min(0), curves.max(0), color='#9ecae1', alpha=.6, label='range over cubes')
    ax.plot(fractions, curves.mean(0), color='#08519c', label=f'mean of {len(curves)} cubes')
    ax.axvspan(*BRAIN_PERCOLATION_REMOVED, color='#d95f02', alpha=.3, lw=0, label='brain breakdown, Ji et al. (0.433–0.447)')
    ax.axvline(np.median(thresholds), color='#08519c', linestyle='--', linewidth=1,
               label=f'prostate breakdown, median {np.median(thresholds):.2f} (range {min(thresholds):.2f}–{max(thresholds):.2f})')
    ax.set_xlabel('fraction of branches removed at random'), ax.set_ylabel('fraction of all nodes in largest component', fontsize=8)
    ax.set_title(f'Random branch removal as in Ji et al. Fig. 3H\n({PERCOLATION_REPEATS} repeats; breakdown = steepest drop)', fontsize=9)
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
        if r['region'] is not None:
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
    ax.axhline(BRAIN_MEAN_LOOP_BRANCHES, color='#d95f02', linestyle=':', label='brain, Ji et al. 2021 (mean)')
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
    fractions = np.round(np.arange(0, .95, .02), 2)
    rows, descriptors, curves = [], [], []
    refined_sizes, refined_branches, refined_bridges = [], 0, 0
    refine_counts = dict(stubs=0, self_loops=0, parallel=0)
    for name, (graph, summary) in cubes.items():
        cube_rows, g = cube_edges(name, graph, summary)
        d, curve = cube_descriptors(name, g, cube_rows, summary, fractions)
        rows += cube_rows
        descriptors.append(d)
        curves.append(curve)
        refined, counts = ji_refine(g)
        for k in refine_counts:
            refine_counts[k] += counts[k]
        for u, v, k in list(refined.edges(keys=True)):
            refined_branches += 1
            cycle = shortest_cycle(refined, u, v, k)
            if cycle is None:
                refined_bridges += 1
            else:
                refined_sizes.append(cycle['branches'])
    looped = [r for r in rows if not r['bridge']]
    regions = region_distributions(rows)
    region_means = [np.mean([r['loop_branches'] for r in rows if r['region'] == k and not r['bridge']]) for k in regions]
    mean_ci = cube_bootstrap(rows, lambda rs: np.mean([r['loop_branches'] for r in rs if not r['bridge']]))
    figure_a(rows, out, refined_sizes)
    figure_b(cubes, rows, out)
    extra = figure_c(rows, descriptors, curves, fractions, out)
    bias = window_bias(cubes)
    figure_d(bias, out)
    annotations = {name: load_annotations(Path(results) / f'{name}_Filaments{variant_suffix}') for name in cubes}
    review = review_small_loops(rows, annotations)
    figure_e(bias, review, out)
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
        ji_refined=dict(mean_loop_branches=float(np.mean(refined_sizes)), median_loop_branches=float(np.median(refined_sizes)),
                        branches=refined_branches, looped_branches=len(refined_sizes), bridges=refined_bridges,
                        deleted=refine_counts,
                        note='Ji et al. Algorithm 1 thresholds; shorter (not dimmer) branch of short two-branch loops deleted'),
        percolation=dict(threshold_removed_median=float(np.median([d['percolation_threshold_removed'] for d in descriptors])),
                         threshold_removed_range=[float(min(d['percolation_threshold_removed'] for d in descriptors)),
                                                  float(max(d['percolation_threshold_removed'] for d in descriptors))],
                         repeats=PERCOLATION_REPEATS, brain_threshold_removed=list(BRAIN_PERCOLATION_REMOVED)),
        brain_reference=dict(mean_loop_branches=BRAIN_MEAN_LOOP_BRANCHES, se=BRAIN_LOOP_SE, source=BRAIN_REFERENCE_LABEL,
                             citation=CITATION,
                             note='published mean ± SE over 240-µm cubes of a whole-brain graph; not recomputed (no brain data)'),
        small_loops=dict(review,
                         brain_by_eye={name: list(brain_small_fraction(lo, hi)) for name, (lo, hi) in SMALL_LOOP_SETS.items()},
                         note='fractions of ALL branches; prostate values rise with window size (see window_bias), so the '
                              'whole-cube values are lower bounds; brain k=3,4 read by eye from Fig. 3E'),
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
