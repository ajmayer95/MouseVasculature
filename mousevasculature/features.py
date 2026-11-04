"""napari layers that highlight measured features of a branch graph."""
import numpy as np

HIGHLIGHTS = ('small-loops', 'loops', 'bridges', 'boundary', 'short', 'bifurcations')


def segments(edges):
    """Line-vector data for a list of branch-graph edges, in voxel coordinates."""
    vectors = []
    for edge in edges:
        points = np.asarray(edge['points_zyx'], dtype=float)
        if len(points) > 1:
            vectors.extend(np.stack((points[:-1], np.diff(points, axis=0)), axis=1))
    return np.asarray(vectors).reshape(-1, 2, 3)


def add_lines(viewer, edges, scale, name, color, width=2, visible=False):
    data = segments(edges)
    if not len(data):
        return None
    layer = viewer.add_vectors(data, ndim=3, scale=scale, vector_style='line', edge_color=color,
                               edge_width=width, opacity=1, name=name, visible=visible,
                               blending='translucent_no_depth')  # draw over the mask surface
    layer.editable = False
    return layer


def add_feature_layers(viewer, branch_graph, scale, name, highlight=(), browser=True):
    """Add hidden highlight layers and a small-loop browser. `highlight` selects visible layers."""
    scale = np.asarray(scale, dtype=float)
    edges = {e['id']: e for e in branch_graph['edges']}
    stats = {i: e.get('statistics', {}) for i, e in edges.items()}
    loops = branch_graph.get('loops', [])
    small = [loop for loop in loops if loop['small']]
    small_ids = {b for loop in small for b in loop['branch_ids']}
    loop_ids = {b for loop in loops for b in loop['branch_ids']}
    show = set(highlight)

    add_lines(viewer, [edges[i] for i in sorted(loop_ids - small_ids)], scale,
              f'{name} looped branches', 'deepskyblue', 2, 'loops' in show)
    add_lines(viewer, [e for i, e in edges.items() if stats[i].get('bridge')], scale,
              f'{name} bridges (tree-like)', 'gray', 2, 'bridges' in show)
    add_lines(viewer, [e for i, e in edges.items() if stats[i].get('touches_boundary')], scale,
              f'{name} crop-boundary branches', 'orange', 2, 'boundary' in show)
    add_lines(viewer, [e for i, e in edges.items() if stats[i].get('short_branch')], scale,
              f'{name} short branches', 'yellow', 2, 'short' in show)
    add_lines(viewer, [edges[i] for i in sorted(small_ids)], scale,
              f'{name} small loops (<50 µm)', 'red', 2, 'small-loops' in show)

    junctions = branch_graph.get('bifurcations', [])
    if junctions:
        nodes = {n['id']: n for n in branch_graph['nodes']}
        layer = viewer.add_points(
            np.asarray([nodes[b['node_id']]['point_zyx'] for b in junctions]), ndim=3, scale=scale,
            size=4, name=f'{name} bifurcations (Murray ratio)', visible='bifurcations' in show,
            features=dict(murray_ratio=[np.log2(b['murray_ratio']) for b in junctions],
                          resolved=[b['resolved'] for b in junctions]),
            face_color='murray_ratio', face_colormap='coolwarm', face_contrast_limits=(-1, 1))
        layer.editable = False

    if small:
        markers = viewer.add_points(
            np.asarray([loop['centroid_zyx'] for loop in small]), ndim=3, scale=scale, size=8,
            face_color='transparent', border_color='red', border_width=.15, name=f'{name} small-loop markers',
            visible='small-loops' in show,
            features=dict(loop_id=[l['loop_id'] for l in small], kind=[l['kind'] for l in small],
                          length_um=[l['length_um'] for l in small]),
            text=dict(string='{loop_id}: {kind} {length_um:.0f} µm', size=8, color='red'),
            blending='translucent_no_depth')
        markers.editable = False
        if browser:
            add_loop_browser(viewer, small, edges, scale, name)


def loop_summary(loop):
    flags = [f for f, on in [('crop', loop['touches_boundary']), ('short branch', loop['has_short_branch'])] if on]
    return (f"loop {loop['loop_id']}: {loop['kind']}, {loop['length_um']:.1f} µm around, "
            f"{loop['extent_um']:.1f} µm across, branches {loop['branch_ids']}, "
            f"min radius {loop['min_radius_um']:.1f} µm" + (f" [{', '.join(flags)}]" if flags else ''))


def add_loop_browser(viewer, loops, edges, scale, name):
    """Dock widget and n / Shift-N keys that centre the camera on each small loop in turn."""
    from magicgui.widgets import Container, Label, PushButton
    state = dict(index=-1, layer=None)
    label = Label(value=f'{len(loops)} small loops. Press Next or n.')
    label.native.setWordWrap(True)

    def go(step):
        state['index'] = (state['index'] + step) % len(loops)
        loop = loops[state['index']]
        data = segments([edges[b] for b in loop['branch_ids']])
        if state['layer'] is None or state['layer'] not in viewer.layers:
            state['layer'] = viewer.add_vectors(data, ndim=3, scale=scale, vector_style='line',
                                                edge_color='lime', edge_width=1, name=f'{name} current loop',
                                                blending='translucent_no_depth')
            state['layer'].editable = False
        else:
            state['layer'].data = data
        centre = np.asarray(loop['centroid_zyx']) * scale
        viewer.camera.center = tuple(centre)
        viewer.dims.point = tuple(centre)
        # Show roughly 80 µm of context, or three loop diameters for larger loops.
        canvas = min(getattr(viewer, '_canvas_size', (600, 600)))
        viewer.camera.zoom = canvas / max(3 * loop['extent_um'], 80)
        text = f"{state['index'] + 1}/{len(loops)}  " + loop_summary(loop)
        label.value = text
        viewer.status = text

    previous, following = PushButton(text='Previous'), PushButton(text='Next')
    previous.changed.connect(lambda: go(-1))
    following.changed.connect(lambda: go(1))
    viewer.window.add_dock_widget(Container(widgets=[label, previous, following]),
                                  name=f'{name} small loops', area='right')
    viewer.bind_key('n', lambda v: go(1), overwrite=True)
    viewer.bind_key('Shift-N', lambda v: go(-1), overwrite=True)
