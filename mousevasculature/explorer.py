"""Interactive napari network explorer: click vessels, read statistics, highlight loops."""
import networkx as nx
import numpy as np
from scipy import ndimage as ndi

from .features import segments
from .statistics import SMALL_LOOP_UM

LOOP_FILTERS = {
    f'all loops': lambda loop: True,
    f'large loops (≥{SMALL_LOOP_UM:g} µm)': lambda loop: not loop['small'],
    f'small loops (<{SMALL_LOOP_UM:g} µm)': lambda loop: loop['small'],
    'interior loops (no crop contact)': lambda loop: not loop['touches_boundary'],
    'loops touching crop': lambda loop: loop['touches_boundary'],
}


def centerline_tree(branch_graph, scale):
    """KD-tree of all centerline points in µm, and the branch id of each point."""
    from scipy.spatial import cKDTree
    points, ids = [], []
    for e in branch_graph['edges']:
        points.extend(e['points_zyx'])
        ids.extend([e['id']] * len(e['points_zyx']))
    return cKDTree(np.asarray(points, dtype=float) * scale), np.asarray(ids)


def branch_labels(mask, branch_graph, scale):
    """Label each mask voxel with 1 + the id of the branch whose centerline is nearest.

    Mask components that contain no centerline stay 0. Distances use physical spacing.
    """
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not branch_graph['edges'] or not mask.any():
        return labels
    scale = np.asarray(scale, dtype=float)
    tree, ids = centerline_tree(branch_graph, scale)
    voxels = np.argwhere(mask)
    _, nearest = tree.query(voxels * scale)
    labels[tuple(voxels.T)] = ids[nearest] + 1
    components, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    seeds = np.zeros(mask.shape, dtype=bool)
    for e in branch_graph['edges']:
        seeds[tuple(np.asarray(e['points_zyx'], dtype=int).T)] = True
    seeded = np.unique(components[seeds & mask])
    labels[~np.isin(components, seeded[seeded > 0])] = 0
    return labels


def vessel_surface(mask, branch_graph, scale, sigma=.8):
    """Smoothed isosurface of the mask in voxel coordinates, with each vertex's nearest branch.

    A Gaussian of `sigma` voxels before marching cubes removes voxel terracing. This is for
    display only; it can slightly thicken or merge vessels closer than about a voxel.
    Volumes are padded so vessels cut by the crop are capped.
    """
    from skimage.measure import marching_cubes
    scale = np.asarray(scale, dtype=float)
    smooth = ndi.gaussian_filter(np.pad(np.asarray(mask, dtype=np.float32), 2), sigma)
    vertices, faces, _, _ = marching_cubes(smooth, .5)
    vertices -= 2
    if branch_graph['edges']:
        tree, ids = centerline_tree(branch_graph, scale)
        _, nearest = tree.query(vertices * scale)
        branch = ids[nearest]
    else:
        branch = np.zeros(len(vertices), dtype=int)
    return vertices.astype(np.float32), faces.astype(np.int32), branch


class Network:
    """Branch graph queries used by the explorer."""

    def __init__(self, branch_graph):
        self.edges = {e['id']: e for e in branch_graph['edges']}
        self.stats = {i: e.get('statistics', {}) for i, e in self.edges.items()}
        self.loops = branch_graph.get('loops', [])
        self.bifurcations = {b['node_id']: b for b in branch_graph.get('bifurcations', [])}
        self.g = nx.MultiGraph()
        self.g.add_nodes_from(n['id'] for n in branch_graph['nodes'])
        for i, e in self.edges.items():
            self.g.add_edge(e['source'], e['target'], key=i, length_um=e['length_um'])
        self.loops_of = {}
        for loop in self.loops:
            for b in loop['branch_ids']:
                self.loops_of.setdefault(b, []).append(loop)

    def length(self, branches):
        return sum(self.edges[b]['length_um'] for b in branches)

    def mean_radius(self, branches):
        total = self.length(branches)
        return sum(self.edges[b]['length_um'] * self.stats[b]['radius_mean_um'] for b in branches) / total if total else None

    def _keys(self, graph, nodes):
        return [min(graph[a][b], key=lambda k: graph[a][b][k]['length_um']) for a, b in zip(nodes[:-1], nodes[1:])]

    def shortest_loop(self, branch):
        """Shortest closed route containing `branch`, as branch ids; None for bridges."""
        e = self.edges[branch]
        u, v = e['source'], e['target']
        if u == v:
            return [branch]
        h = self.g.copy()
        h.remove_edge(u, v, key=branch)
        try:
            nodes = nx.shortest_path(h, u, v, weight='length_um')
        except nx.NetworkXNoPath:
            return None
        return [branch] + self._keys(h, nodes)

    def path_between(self, first, last):
        """Shortest route of branches joining two branches, including both; None if disconnected."""
        if first == last:
            return [first]
        best = None
        for u in {self.edges[first]['source'], self.edges[first]['target']}:
            for v in {self.edges[last]['source'], self.edges[last]['target']}:
                try:
                    nodes = nx.shortest_path(self.g, u, v, weight='length_um')
                except nx.NetworkXNoPath:
                    continue
                route = self._keys(self.g, nodes)
                candidate = [first] + [k for k in route if k not in (first, last)] + [last]
                if best is None or self.length(candidate) < self.length(best):
                    best = candidate
        return best

    def component(self, branch):
        nodes = nx.node_connected_component(self.g, self.edges[branch]['source'])
        return sorted(k for *_, k in self.g.edges(nodes, keys=True))

    def describe_branch(self, branch):
        s, e = self.stats[branch], self.edges[branch]
        tortuosity = f"{s['tortuosity']:.2f}" if s.get('tortuosity') is not None else 'n/a (closed)'
        lines = [f"Branch {branch}  ·  component {s['component_id']}",
                 f"length {s['length_um']:.1f} µm, chord {s['chord_um']:.1f} µm, tortuosity {tortuosity}",
                 f"radius mean {s['radius_mean_um']:.2f} µm (min {s['radius_min_um']:.2f}, max {s['radius_max_um']:.2f})"]
        ends = []
        for node in dict.fromkeys((e['source'], e['target'])):
            degree = self.g.degree(node)
            text = f"node {node}: " + ('endpoint' if degree == 1 else f'degree {degree}')
            if node in self.bifurcations:
                b = self.bifurcations[node]
                role = 'parent' if b['parent_branch'] == branch else 'child'
                angle = f", angle {b['branching_angle_deg']:.0f}°" if b['branching_angle_deg'] is not None else ''
                text += f" ({role}; Murray ratio {b['murray_ratio']:.2f}{angle}{'' if b['resolved'] else ', unresolved'})"
            ends.append(text)
        lines.append('ends: ' + '; '.join(ends))
        flags = ['bridge (not on any loop)' if s['bridge'] else 'on a loop']
        flags += [f for f, on in [('touches crop', s['touches_boundary']), ('short', s['short_branch']),
                                  ('self-loop', s['self_loop'])] if on]
        lines.append('flags: ' + ', '.join(flags))
        loops = self.loops_of.get(branch, [])
        if loops:
            lines.append('basis loops: ' + ', '.join(f"#{l['loop_id']} ({l['length_um']:.0f} µm, {l['kind']})" for l in loops))
        shortest = None if s['bridge'] else self.shortest_loop(branch)
        if shortest:
            lines.append(f"shortest loop through it: {self.length(shortest):.0f} µm, {len(shortest)} branches")
        return '\n'.join(lines)

    def describe_loop(self, loop):
        ids = loop['branch_ids']
        flags = [f for f, on in [('touches crop', loop['touches_boundary']),
                                 ('has short branch', loop['has_short_branch'])] if on]
        return '\n'.join([
            f"Loop #{loop['loop_id']}  ·  {loop['kind']}{' (small)' if loop['small'] else ''}",
            f"perimeter {loop['length_um']:.1f} µm, extent {loop['extent_um']:.1f} µm, {len(ids)} branches",
            f"length-weighted radius {self.mean_radius(ids):.2f} µm, min radius {loop['min_radius_um']:.2f} µm",
            'branches: ' + ', '.join(map(str, ids)),
        ] + (['flags: ' + ', '.join(flags)] if flags else []))

    def describe_selection(self, branches):
        if len(branches) == 1:
            return self.describe_branch(branches[0])
        return (f"{len(branches)} branches selected: {', '.join(map(str, branches))}\n"
                f"total length {self.length(branches):.1f} µm, "
                f"length-weighted radius {self.mean_radius(branches):.2f} µm\n"
                f"last clicked:\n" + self.describe_branch(branches[-1]))


SURFACE_COLOURS = {
    'radius (µm)': ('viridis', 'radius_mean_um', False),
    'branch length (µm)': ('viridis', 'length_um', True),
    'tortuosity': ('magma', 'tortuosity', False),
    'smallest loop perimeter (µm)': ('plasma', 'loop_um', True),
    'component': ('tab10', 'component_id', None),
    'loop / bridge / crop': (None, 'kind', None),
    'branch (random colours)': (None, 'branch', None),
    'uniform grey': (None, None, None),
}
NO_VALUE = (.45, .45, .45, 1.)
ANNOTATION_LABELS = ('real', 'artifact', 'unsure')
ANNOTATION_FIELDS = ['loop_id', 'label', 'length_um', 'kind', 'small', 'branch_ids', 'centroid_zyx', 'updated']


def branch_colours(net, choice):
    """RGBA per branch id for a surface colouring choice, and a legend description."""
    from matplotlib import colormaps
    ids = np.arange(max(net.edges) + 1)
    colours = np.tile(NO_VALUE, (len(ids), 1))
    cmap_name, key, log = SURFACE_COLOURS[choice]
    if key is None:
        colours[:] = (.7, .7, .7, 1.)
        return colours, dict(kind='text', text='')
    if key == 'branch':
        rng = np.random.default_rng(0)
        colours[:, :3] = rng.uniform(.25, 1, (len(ids), 3))
        return colours, dict(kind='text', text='Each branch has its own colour (mask split by nearest centerline).')
    if key == 'kind':
        legend = []
        for b, s in net.stats.items():
            colours[b] = (1, .6, .1, 1) if s['touches_boundary'] else (.6, .6, .6, 1) if s['bridge'] else (.15, .6, 1, 1)
        return colours, dict(kind='text', text='blue: on a loop · grey: bridge (tree-like) · orange: touches crop')
    if key == 'component':
        cmap = colormaps['tab10']
        for b, s in net.stats.items():
            colours[b] = cmap(s['component_id'] % 10)
        return colours, dict(kind='text', text='Colour = connected component (tab10, repeats after 10)')
    values = np.full(len(ids), np.nan)
    for b, s in net.stats.items():
        if key == 'loop_um':
            values[b] = min((l['length_um'] for l in net.loops_of.get(b, [])), default=np.nan)
        else:
            values[b] = s[key] if s.get(key) is not None else np.nan
    finite = values[np.isfinite(values)]
    if not len(finite):
        return colours, dict(kind='text', text='No values.')
    shown = np.log10(values) if log else values
    lo, hi = np.percentile(shown[np.isfinite(shown)], [2, 98])
    hi = hi if hi > lo else lo + 1
    cmap = colormaps[cmap_name]
    ok = np.isfinite(shown)
    colours[ok] = cmap(np.clip((shown[ok] - lo) / (hi - lo), 0, 1))
    fmt = (lambda v: f'{10 ** v:.0f}') if log else (lambda v: f'{v:.2f}' if hi - lo < 10 else f'{v:.0f}')
    missing = {'tortuosity': 'closed loops', 'loop_um': 'bridges (no loop)'}.get(key)
    return colours, dict(kind='gradient', cmap=cmap, low=fmt(lo), high=fmt(hi),
                         note=('log scale; ' if log else '') + '2nd–98th percentile' +
                              (f'; grey = {missing}' if missing else ''))


def add_explorer(viewer, mask, branch_graph, scale, name, folder=None, on_select=None):
    """Add the vessel surface, click selection, highlight overlays, slab clipping and loop annotation.

    Returns a dict of actions (for scripting and tests) including `cleanup`, which removes
    the panel, mouse callback and key bindings; the caller removes layers.
    """
    import csv
    import datetime
    from pathlib import Path
    from magicgui.widgets import (ComboBox, Container, FloatSlider, Label, PushButton, SpinBox, TextEdit)
    from qtpy.QtWidgets import QScrollArea

    scale = np.asarray(scale, dtype=float)
    net = Network(branch_graph)
    if not net.edges:
        return None
    labels = branch_labels(mask, branch_graph, scale)
    vessels = viewer.add_labels(labels, scale=scale, name=f'{name} voxel mask (by branch)',
                                rendering='iso_categorical', opacity=.35)
    vessels.visible = False  # hidden after creation: see napari note in analysis.add_analysis_layers
    vessels.editable = False
    vertices, faces, vertex_branch = vessel_surface(mask, branch_graph, scale)
    surface = viewer.add_surface((vertices, faces), scale=scale, name=f'{name} vessel surface',
                                 shading='smooth', opacity=1.)

    overlays = {}

    def show(kind, branches, color, width=1.5):
        """Replace one overlay layer's content with the given branches."""
        data = segments([net.edges[b] for b in branches]) if branches else np.zeros((0, 2, 3))
        layer = overlays.get(kind)
        if layer is None or layer not in viewer.layers:
            layer = viewer.add_vectors(data, ndim=3, scale=scale, vector_style='line', edge_color=color,
                                       edge_width=width, name=f'{name} {kind}', blending='translucent_no_depth')
            layer.editable = False
            overlays[kind] = layer
            apply_slab()
        else:
            layer.data = data
        layer.visible = bool(branches)

    def focus(branches):
        points = np.concatenate([np.asarray(net.edges[b]['points_zyx'], float) for b in branches]) * scale
        centre = points.mean(axis=0)
        extent = 2 * np.linalg.norm(points - centre, axis=1).max()
        viewer.camera.center = tuple(centre)
        viewer.dims.point = tuple(centre)
        canvas = min(getattr(viewer, '_canvas_size', (600, 600)))
        viewer.camera.zoom = canvas / max(1.6 * extent, 80)
        if slab['normal'] is not None:  # move (and if needed widen) an active slab to contain the target
            n = slab['normal']
            along = (points / scale - slab_centre) @ n * np.linalg.norm(n * scale)
            needed = float(along.max() - along.min()) + 20.
            if needed > slab_thickness.value:
                slab_thickness.value = float(min(needed, slab_thickness.max))
            middle = float(along.max() + along.min()) / 2
            slab_position.value = float(np.clip(middle, slab_position.min, slab_position.max))

    # ---- panel text ------------------------------------------------------------------
    state = dict(selected=[], loop_index=-1, loop=None)
    info = TextEdit(value='Click a vessel to select it. Shift-click adds to the selection.\n'
                          'Drag rotates; a click without dragging selects.')
    info.native.setReadOnly(True)
    info.native.setMinimumHeight(220)

    def report(text):
        info.value = text
        viewer.status = text.split('\n')[0]

    # ---- surface colouring -----------------------------------------------------------
    colour_by = ComboBox(choices=list(SURFACE_COLOURS), value='radius (µm)', label='colour by')
    bar = Label(value='')
    bar.native.setFixedHeight(12)
    bar.native.setMinimumWidth(320)
    bar_text = Label(value='')
    bar_text.native.setWordWrap(True)

    def recolour(_=None):
        colours, legend = branch_colours(net, colour_by.value)
        surface.vertex_colors = colours[vertex_branch].astype(np.float32)
        if legend['kind'] == 'gradient':
            stops = ', '.join(f'stop:{t:.2f} rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'
                              for t in np.linspace(0, 1, 9) for r, g, b, _ in [legend['cmap'](t)])
            bar.native.setStyleSheet(f'background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});')
            bar.native.setVisible(True)
            bar_text.value = f"{legend['low']}  →  {legend['high']}   ({legend['note']})"
        else:
            bar.native.setVisible(False)
            bar_text.value = legend['text']

    colour_by.changed.connect(recolour)

    # ---- slab clipping ---------------------------------------------------------------
    slab_centre = (np.asarray(mask.shape) - 1) / 2
    half_diagonal = float(np.linalg.norm(np.asarray(mask.shape) * scale) / 2)
    slab_axis = ComboBox(choices=['off', 'Z', 'Y', 'X', 'view direction'], value='off', label='slab')
    slab_position = FloatSlider(min=-half_diagonal, max=half_diagonal, value=0., label='centre (µm)')
    slab_thickness = FloatSlider(min=5., max=300., value=40., label='thickness (µm)')
    slab = dict(normal=None)

    def slab_normal():
        axis = slab_axis.value
        if axis == 'off':
            return None
        if axis == 'view direction':
            n = np.asarray(viewer.camera.view_direction, dtype=float) / scale
        else:
            n = np.eye(3)[['Z', 'Y', 'X'].index(axis)]
        return n / np.linalg.norm(n)

    def in_slab(points):
        """Boolean mask of data-coordinate points inside the current slab."""
        if slab['normal'] is None:
            return np.ones(len(points), dtype=bool)
        n = slab['normal']
        um_per_unit = np.linalg.norm(n * scale)
        offset = (points - slab_centre) @ n * um_per_unit
        return np.abs(offset - slab_position.value) <= slab_thickness.value / 2

    def apply_slab(_=None):
        slab['normal'] = n = slab_normal()
        planes = []
        if n is not None:
            um_per_unit = np.linalg.norm(n * scale)
            low = slab_centre + n * (slab_position.value - slab_thickness.value / 2) / um_per_unit
            high = slab_centre + n * (slab_position.value + slab_thickness.value / 2) / um_per_unit
            planes = [dict(position=tuple(low), normal=tuple(n), enabled=True),
                      dict(position=tuple(high), normal=tuple(-n), enabled=True)]
        for layer in list(viewer.layers):
            if layer.ndim == 3:
                layer.experimental_clipping_planes = planes

    for widget in (slab_axis, slab_position, slab_thickness):
        widget.changed.connect(apply_slab)

    # ---- picking ---------------------------------------------------------------------
    def pick(position, direction, displayed):
        """First labelled voxel along the view ray that lies inside the slab."""
        if len(displayed) != 3:
            value = vessels.get_value(position, world=True)
            return int(value or 0)
        start, end = vessels.get_ray_intersections(position, direction, displayed, world=True)
        if start is None or end is None:
            return 0
        start, end = np.asarray(start, float), np.asarray(end, float)
        steps = int(np.ceil(np.linalg.norm(end - start) * 2)) + 1
        points = start + (end - start) * np.linspace(0, 1, steps)[:, None]
        points = points[in_slab(points)]
        index = np.clip(np.round(points).astype(int), 0, np.asarray(labels.shape) - 1)
        values = labels[tuple(index.T)]
        hits = values[values > 0]
        return int(hits[0]) if len(hits) else 0

    def select(branch, add=False):
        if add and branch in state['selected']:
            state['selected'].remove(branch)
        elif add:
            state['selected'].append(branch)
        else:
            state['selected'] = [branch]
        show('selection', state['selected'], 'yellow', 2)
        report(net.describe_selection(state['selected']) if state['selected'] else 'Selection cleared.')
        if on_select:
            on_select(list(state['selected']))

    def require_selection():
        if not state['selected']:
            report('Select a vessel first (click it).')
            return None
        return state['selected'][-1]

    def on_click(viewer, event):
        if event.button != 1:
            return
        position, direction, displayed = event.position, event.view_direction, event.dims_displayed
        moves = 0
        yield
        while event.type == 'mouse_move':
            moves += 1
            yield
        if moves > 2:
            return
        value = pick(position, direction, displayed)
        if value:
            select(value - 1, add='Shift' in event.modifiers)

    viewer.mouse_drag_callbacks.append(on_click)

    # ---- branch actions --------------------------------------------------------------
    def loops_through():
        branch = require_selection()
        if branch is None:
            return
        loops = net.loops_of.get(branch, [])
        show('loops through selection', sorted({b for l in loops for b in l['branch_ids']}), 'magenta')
        if not loops:
            report(net.describe_branch(branch) + '\n\nThis branch is a bridge: it lies on no loop.')
        else:
            report(net.describe_branch(branch) + '\n\n' + '\n\n'.join(describe_loop(l) for l in loops))
            focus(sorted({b for l in loops for b in l['branch_ids']}))

    def shortest_loop():
        branch = require_selection()
        if branch is None:
            return
        route = net.shortest_loop(branch)
        show('shortest loop', route or [], 'orange', 2)
        if route:
            report(f"Shortest loop through branch {branch}: {net.length(route):.1f} µm, "
                   f"{len(route)} branches ({', '.join(map(str, route))}), "
                   f"length-weighted radius {net.mean_radius(route):.2f} µm\n\n" + net.describe_branch(branch))
            focus(route)
        else:
            report(net.describe_branch(branch) + '\n\nNo loop: this branch is a bridge.')

    def component():
        branch = require_selection()
        if branch is None:
            return
        members = net.component(branch)
        show('component', members, 'cyan', 1)
        report(f"Component {net.stats[branch]['component_id']}: {len(members)} branches, "
               f"{net.length(members):.0f} µm\n\n" + net.describe_branch(branch))

    def path():
        if len(state['selected']) < 2:
            report('Select two vessels (click, then Shift-click) to find the path between them.')
            return
        first, last = state['selected'][0], state['selected'][-1]
        route = net.path_between(first, last)
        show('path', route or [], 'lime', 2)
        report(f"Shortest path from branch {first} to {last}: {net.length(route):.1f} µm, {len(route)} branches"
               if route else f'Branches {first} and {last} are in different components.')

    def clear():
        state['selected'] = []
        for layer in overlays.values():
            if layer in viewer.layers:
                layer.visible = False
        report('Highlights cleared. Click a vessel to select it.')
        if on_select:
            on_select([])

    # ---- loop annotation -------------------------------------------------------------
    annotation_file = Path(folder) / 'loop_annotations.csv' if folder else None
    annotations = {}
    stale = 0
    if annotation_file and annotation_file.exists():
        by_id = {l['loop_id']: l for l in net.loops}
        with annotation_file.open(newline='') as handle:
            for row in csv.DictReader(handle):
                loop = by_id.get(int(row['loop_id']))
                if loop and row['branch_ids'] == ' '.join(map(str, loop['branch_ids'])):
                    annotations[loop['loop_id']] = row
                else:
                    stale += 1

    def describe_loop(loop):
        text = net.describe_loop(loop)
        label = annotations.get(loop['loop_id'], {}).get('label')
        return text + (f'\nannotation: {label}' if label else '\nannotation: none (press R real, J junk/artifact, U unsure)')

    def annotation_summary():
        counts = {k: sum(r['label'] == k for r in annotations.values()) for k in ANNOTATION_LABELS}
        small = [l for l in net.loops if l['small']]
        done_small = sum(l['loop_id'] in annotations for l in small)
        return (f"Annotated {len(annotations)}/{len(net.loops)} loops ({done_small}/{len(small)} small): "
                + ', '.join(f'{n} {k}' for k, n in counts.items())
                + (f'. {stale} saved annotations no longer match a loop and were ignored.' if stale else ''))

    def save_annotations():
        if annotation_file is None:
            return
        with annotation_file.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(sorted(annotations.values(), key=lambda r: int(r['loop_id'])))

    def annotate(label):
        loop = state['loop']
        if loop is None:
            report('Choose a loop first (Previous/Next or Go to loop), then annotate it.')
            return
        if label is None:
            annotations.pop(loop['loop_id'], None)
        else:
            annotations[loop['loop_id']] = dict(
                loop_id=loop['loop_id'], label=label, length_um=f"{loop['length_um']:.2f}", kind=loop['kind'],
                small=loop['small'], branch_ids=' '.join(map(str, loop['branch_ids'])),
                centroid_zyx=' '.join(f'{v:.1f}' for v in loop['centroid_zyx']),
                updated=datetime.datetime.now().isoformat(timespec='seconds'))
        save_annotations()
        annotation_status.value = annotation_summary()
        report(describe_loop(loop) + '\n\n' + annotation_summary()
               + (f'\nSaved to {annotation_file}' if annotation_file else '\n(not saved: no results folder)'))

    filters = dict(LOOP_FILTERS)
    filters['not yet annotated'] = lambda loop: loop['loop_id'] not in annotations
    filters['small, not yet annotated'] = lambda loop: loop['small'] and loop['loop_id'] not in annotations
    for label in ANNOTATION_LABELS:
        filters[f'annotated {label}'] = (lambda wanted: lambda loop: annotations.get(loop['loop_id'], {}).get('label') == wanted)(label)

    # ---- loop browser ----------------------------------------------------------------
    loop_filter = ComboBox(choices=list(filters), value='all loops', label='loops')
    loop_id = SpinBox(min=0, max=max(len(net.loops) - 1, 0), label='loop #')

    def filtered():
        keep = filters[loop_filter.value]
        return [l for l in net.loops if keep(l)]

    def show_loop(loop):
        state['loop'] = loop
        show('current loop', loop['branch_ids'], 'orange', 2)
        focus(loop['branch_ids'])
        report(describe_loop(loop))

    def step(direction):
        loops = filtered()
        if not loops:
            report(f'No loops match "{loop_filter.value}".')
            return
        state['loop_index'] = (state['loop_index'] + direction) % len(loops)
        loop = loops[state['loop_index']]
        loop_id.value = loop['loop_id']
        show_loop(loop)
        viewer.status = f"{state['loop_index'] + 1}/{len(loops)} {loop_filter.value}: " + viewer.status

    def go_to_loop():
        if net.loops:
            show_loop(net.loops[loop_id.value])

    def on_filter(_=None):
        state['loop_index'] = -1
        loops = filtered()
        show('filtered loops', sorted({b for l in loops for b in l['branch_ids']}), 'magenta', 1)
        report(f'{len(loops)} loops match "{loop_filter.value}" (magenta). Use Previous/Next to visit them.')

    loop_filter.changed.connect(on_filter)
    annotation_status = Label(value=annotation_summary())
    annotation_status.native.setWordWrap(True)

    # ---- layout ----------------------------------------------------------------------
    def button(text, callback):
        b = PushButton(text=text)
        b.changed.connect(lambda: callback())
        return b

    def row(*widgets):
        return Container(layout='horizontal', widgets=list(widgets), labels=False)

    def heading(text):
        label = Label(value=f'<b>{text}</b>')
        return label

    panel = Container(widgets=[
        info,
        row(button('Loops through branch', loops_through), button('Shortest loop', shortest_loop)),
        row(button('Component', component), button('Path between selected', path)),
        button('Clear highlights', clear),
        heading('Surface'), colour_by, bar, bar_text,
        heading('Slab (clip to a thick slice)'), slab_axis, slab_position, slab_thickness,
        heading('Loops'), loop_filter,
        row(button('Previous', lambda: step(-1)), button('Next', lambda: step(1))),
        row(loop_id, button('Go to loop', go_to_loop)),
        row(button('Real (R)', lambda: annotate('real')), button('Artifact (J)', lambda: annotate('artifact')),
            button('Unsure (U)', lambda: annotate('unsure')), button('Clear', lambda: annotate(None))),
        annotation_status,
    ])
    panel.native.setMinimumWidth(400)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel.native)
    scroll.setMinimumWidth(430)
    scroll.setMaximumWidth(500)
    dock = viewer.window.add_dock_widget(scroll, name=f'{name} explorer', area='right')

    keys = {'n': lambda v: step(1), 'Shift-N': lambda v: step(-1),
            # Letters unused by napari 0.5.6 layer keymaps (surface/vectors use 1 and 2, labels 1-7).
            'r': lambda v: annotate('real'), 'j': lambda v: annotate('artifact'), 'u': lambda v: annotate('unsure')}
    for key, action in keys.items():
        viewer.bind_key(key, action, overwrite=True)
    recolour()
    viewer.layers.selection.active = surface

    def cleanup():
        if on_click in viewer.mouse_drag_callbacks:
            viewer.mouse_drag_callbacks.remove(on_click)
        for key in keys:
            viewer.bind_key(key, None, overwrite=True)
        viewer.window.remove_dock_widget(dock)

    return dict(select=select, step=step, loops_through=loops_through, shortest_loop=shortest_loop,
                component=component, path=path, clear=clear, filter=loop_filter, info=info,
                vessels=vessels, surface=surface, network=net, pick=pick, annotate=annotate,
                colour_by=colour_by, slab_axis=slab_axis, slab_position=slab_position,
                slab_thickness=slab_thickness, go_to_loop=go_to_loop, loop_id=loop_id,
                annotations=annotations, cleanup=cleanup)
