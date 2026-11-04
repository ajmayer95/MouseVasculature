"""Interactive explorer application: switch samples and mask variants without restarting."""
from pathlib import Path

from .pipeline import VARIANTS, load_or_run


def tiff_paths(data_path):
    data_path = Path(data_path)
    if data_path.is_dir():
        return sorted(p for p in data_path.iterdir() if p.suffix.lower() in ('.tif', '.tiff'))
    return [data_path]


def sample_name(path):
    stem = Path(path).stem
    return stem[:-len('_Filaments')] if stem.endswith('_Filaments') else stem


def explore(data_path, output, sample=None, variant='filled+closed', scale=None, close_radius=1, run=True):
    """Open napari with the network explorer. Saved, current results are reused."""
    import napari
    from magicgui.widgets import ComboBox, Container, Label, PushButton

    from .analysis import add_analysis_layers
    from .explorer import add_explorer
    from .features import add_feature_layers
    from .statistics import save_comparison

    paths = tiff_paths(data_path)
    if not paths or any(not p.is_file() for p in paths):
        raise SystemExit(f'No TIFF files found at {data_path}')
    names = [sample_name(p) for p in paths]
    if sample is not None and sample not in names:
        raise SystemExit(f'--sample {sample} not found; choose from {", ".join(names)}')
    viewer = napari.Viewer(title='Mouse prostate vasculature — explorer', ndisplay=3)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    state = dict(explorer=None)

    sample_box = ComboBox(choices=names, value=sample or names[0], label='sample')
    variant_box = ComboBox(choices=list(VARIANTS), value=variant, label='mask')
    status = Label(value='')
    status.native.setWordWrap(True)

    def load():
        if state['explorer']:
            state['explorer']['cleanup']()
            state['explorer'] = None
        viewer.layers.clear()
        name, chosen = sample_box.value, variant_box.value
        path = paths[names.index(name)]
        status.value = f'Loading {name} ({chosen})…'
        data = load_or_run(path, chosen, output, scale, close_radius)
        if data['computed']:
            save_comparison(output, [])
        add_analysis_layers(viewer, data['skeleton'], data['graph'], data['scale'], name, visible=False,
                            color_by='radius')
        add_feature_layers(viewer, data['graph'], data['scale'], name, browser=False)
        state['explorer'] = add_explorer(viewer, data['mask'], data['graph'], data['scale'], name,
                                         folder=data['folder'])
        summary = data['summary'] or {}
        status.value = (f"{name} · {chosen} · {'computed now' if data['computed'] else 'loaded from saved results'}\n"
                        f"{summary.get('branches', '?')} branches, {summary.get('independent_cycles', '?')} loops, "
                        f"{summary.get('components', '?')} components")
        viewer.reset_view()

    load_button = PushButton(text='Load')
    load_button.changed.connect(lambda: load())
    data_panel = Container(widgets=[sample_box, variant_box, load_button, status])
    data_panel.native.setMaximumHeight(190)  # leave the right column to the explorer panel
    viewer.window.add_dock_widget(data_panel, name='Data', area='right')
    state['controls'] = dict(sample=sample_box, variant=variant_box, load=load, status=status)
    load()
    if run:
        napari.run()
    return viewer, state
