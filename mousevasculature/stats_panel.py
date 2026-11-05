"""Statistics dock for the explorer: histograms of the current cube against all cubes.

The current cube is drawn as filled bars and every cube with saved results (same mask
variant) as a grey outline, both as probability densities so their shapes compare
directly. The clicked vessel is marked on branch metrics, and a table ranks the cube's
headline numbers among the others.
"""
import json
from pathlib import Path

import numpy as np


def _branches(graph, field, scale=1.):
    stats = [e['statistics'] for e in graph['edges'] if e['statistics'].get(field) is not None]
    return np.array([s[field] * scale for s in stats], float), np.array([s['length_um'] for s in stats], float)


def _resolved(graph, field):
    values = [b[field] for b in graph.get('bifurcations', []) if b.get('resolved') and b.get(field) is not None]
    return np.array(values, float), None


# name -> (values and weights from a branch graph, branch statistic shown for the clicked vessel, log x)
METRICS = {
    'vessel diameter (µm, length-weighted)': (lambda g: _branches(g, 'radius_mean_um', 2.), ('radius_mean_um', 2.), False),
    'branch length (µm)': (lambda g: (_branches(g, 'length_um')[0], None), ('length_um', 1.), True),
    'tortuosity (length / chord)': (lambda g: (_branches(g, 'tortuosity')[0], None), ('tortuosity', 1.), True),
    'loop perimeter (µm)': (lambda g: (np.array([l['length_um'] for l in g.get('loops', [])], float), None), None, True),
    'Murray ratio (resolved junctions)': (lambda g: _resolved(g, 'murray_ratio'), None, False),
    'branching angle (°)': (lambda g: _resolved(g, 'branching_angle_deg'), None, False),
}

# label -> (summary value getter, format)
HEADLINES = [
    ('length density (mm/mm³)', lambda s: s.get('length_density_mm_per_mm3'), '{:.0f}'),
    ('large loops per mm³', lambda s: s.get('large_cycle_density_per_mm3'), '{:.0f}'),
    ('mean diameter (µm)', lambda s: 2 * s['length_weighted_radius_um'] if s.get('length_weighted_radius_um') else None, '{:.1f}'),
    ('vessel volume fraction (%)', lambda s: 100 * s['foreground_fraction'] if s.get('foreground_fraction') else None, '{:.2f}'),
    ('median tissue distance (µm)', lambda s: (s.get('tissue_distance_um') or {}).get('median'), '{:.0f}'),
    ('median Murray ratio', lambda s: (s.get('murray_ratio') or {}).get('median'), '{:.2f}'),
    ('crop-cut / interior ends', lambda s: s.get('crop_endpoints'), '{}'),
]


def load_cohort(folders):
    """name -> (branch graph, summary) for every results folder that holds both files."""
    cohort = {}
    for name, folder in folders.items():
        try:
            graph = json.loads((Path(folder) / 'branch_graph.json').read_text())
            summary = json.loads((Path(folder) / 'statistics.json').read_text())
        except (OSError, ValueError):
            continue
        cohort[name] = (graph, summary)
    return cohort


def cohort_values(cohort):
    """Pre-extract every metric for every cube so switching metric is instant."""
    return {metric: {name: spec[0](graph) for name, (graph, _) in cohort.items()}
            for metric, spec in METRICS.items()}


def weighted_median(values, weights=None):
    if not len(values):
        return None
    if weights is None:
        return float(np.median(values))
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def rank_text(value, others):
    others = [v for v in others if v is not None]
    if value is None or not others:
        return ''
    rank = 1 + sum(v > value for v in others)
    return f'{rank} of {len(others)}'


def add_stats_panel(viewer, name, graph, summary, cohort=None, values=None):
    """Add the statistics dock. `cohort` maps cube name -> (graph, summary), including this cube.

    Returns dict(metric, redraw, mark, cleanup); `mark(branch_ids)` draws the clicked vessels.
    """
    from magicgui.widgets import ComboBox, Container, Label
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    cohort = dict(cohort or {})
    cohort[name] = (graph, summary)
    values = values or cohort_values(cohort)
    values = {m: {**v, name: METRICS[m][0](graph)} for m, v in values.items()}
    state = dict(selected=[])
    by_id = {e['id']: e for e in graph['edges']}

    fg, bg, accent, pooled_colour = '#d8dde6', '#262930', '#4fa3e0', '#9aa0aa'
    figure = Figure(figsize=(4.2, 2.9), facecolor=bg, layout='constrained')
    canvas = FigureCanvasQTAgg(figure)
    canvas.setMinimumHeight(270)
    axes = figure.add_subplot()
    metric = ComboBox(choices=list(METRICS), value=list(METRICS)[0], label='plot')
    caption = Label(value='')
    caption.native.setWordWrap(True)
    table = Label(value='')
    table.native.setWordWrap(True)

    def redraw(_=None):
        chosen = metric.value
        compute, branch_field, log = METRICS[chosen]
        current, weights = values[chosen][name]
        pooled = [values[chosen][n] for n in values[chosen]]
        pooled_v = np.concatenate([v for v, _ in pooled]) if pooled else np.array([])
        pooled_w = (np.concatenate([w for _, w in pooled]) if pooled and all(w is not None for _, w in pooled)
                    else None)
        axes.clear()
        axes.set_facecolor(bg)
        for spine in axes.spines.values():
            spine.set_color(pooled_colour)
        axes.tick_params(colors=fg, labelsize=8)
        if not len(current):
            axes.text(.5, .5, 'no values in this cube', color=fg, ha='center', transform=axes.transAxes)
            caption.value = ''
            canvas.draw_idle()
            return
        everything = pooled_v[np.isfinite(pooled_v) & ((pooled_v > 0) if log else True)]
        low, high = np.percentile(everything, [.5, 99.5])
        count = int(np.clip(2 * np.sqrt(len(current)), 12, 40))  # fewer bins for small samples
        bins = np.geomspace(low, high, count) if log else np.linspace(low, high, count)
        if len(everything) > len(current):
            axes.hist(pooled_v, bins=bins, weights=pooled_w, density=True, histtype='step', color=pooled_colour,
                      linewidth=1.2, label=f'all {len(values[chosen])} cubes')
        axes.hist(current, bins=bins, weights=weights, density=True, color=accent, alpha=.75, label=name)
        median = weighted_median(current, weights)
        axes.axvline(median, color=accent, linestyle='--', linewidth=1)
        pooled_median = weighted_median(pooled_v, pooled_w)
        if branch_field and state['selected']:
            field, factor = branch_field
            marked = [by_id[b]['statistics'].get(field) for b in state['selected']]
            for i, value in enumerate(v * factor for v in marked if v is not None):
                axes.axvline(value, color='yellow', linewidth=2, label='selected' if i == 0 else None)
        if log:
            axes.set_xscale('log')
        axes.set_xlabel(chosen, color=fg, fontsize=9)
        axes.set_ylabel('density' + (' (by length)' if weights is not None else ''), color=fg, fontsize=8)
        axes.set_yticks([])
        legend = axes.legend(fontsize=7, frameon=False, labelcolor=fg)
        caption.value = (f'{name}: median {median:.3g} (dashed), n = {len(current)}'
                         + (f'  ·  all cubes: median {pooled_median:.3g}, n = {len(pooled_v)}' if pooled_median else '')
                         + ('\nClick a vessel to mark it.' if branch_field and not state['selected'] else ''))
        canvas.draw_idle()

    def headline_table():
        rows = []
        for label, get, fmt in HEADLINES:
            value = get(summary)
            if value is None:
                continue
            text = fmt.format(value)
            if label.startswith('crop-cut'):
                text = f"{summary.get('crop_endpoints')} / {summary.get('interior_endpoints')}"
                rows.append(f'<tr><td>{label}</td><td align="right"><b>{text}</b></td><td></td></tr>')
                continue
            rank = rank_text(value, [get(s) for _, s in cohort.values()])
            rows.append(f'<tr><td>{label}</td><td align="right"><b>{text}</b></td>'
                        f'<td style="color:{pooled_colour}">&nbsp;&nbsp;{"rank " + rank if rank else ""}</td></tr>')
        table.value = ('<table cellspacing="2">' + ''.join(rows) + '</table>'
                       f'<span style="color:{pooled_colour}">rank 1 = highest among cubes with saved results</span>')

    def mark(branches):
        state['selected'] = list(branches)
        if METRICS[metric.value][1]:
            redraw()

    metric.changed.connect(redraw)
    panel = Container(widgets=[metric, caption, table])
    panel.native.layout().insertWidget(1, canvas)
    panel.native.setMinimumWidth(400)
    headline_table()
    redraw()
    dock = viewer.window.add_dock_widget(panel, name=f'{name} statistics', area='right', tabify=False)

    def cleanup():
        viewer.window.remove_dock_widget(dock)
        figure.clear()

    return dict(metric=metric, redraw=redraw, mark=mark, cleanup=cleanup, axes=axes, values=values)
