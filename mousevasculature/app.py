"""Interactive explorer application: switch samples and mask variants without restarting."""
from pathlib import Path

from .pipeline import VARIANTS, is_current, load_or_run, results_folder


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
    from .stats_panel import add_stats_panel, cohort_values, load_cohort

    paths = tiff_paths(data_path)
    if not paths or any(not p.is_file() for p in paths):
        raise SystemExit(f'No TIFF files found at {data_path}')
    names = [sample_name(p) for p in paths]
    if sample is not None and sample not in names:
        raise SystemExit(f'--sample {sample} not found; choose from {", ".join(names)}')
    viewer = napari.Viewer(title='Mouse prostate vasculature — explorer', ndisplay=3)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    state = dict(explorer=None, stats=None, cohorts={})

    sample_box = ComboBox(choices=names, value=sample or names[0], label='sample')
    variant_box = ComboBox(choices=list(VARIANTS), value=variant, label='mask')
    status = Label(value='')
    status.native.setWordWrap(True)

    def carried_settings():
        """View and explorer settings to keep when switching cubes (all cubes share one size)."""
        ui = state['explorer']
        if ui is None:
            return None
        return dict(camera=dict(center=viewer.camera.center, zoom=viewer.camera.zoom, angles=viewer.camera.angles),
                    widgets={k: ui[k].value for k in ('colour_by', 'slab_axis', 'slab_position', 'slab_thickness')})

    def load(keep_view=False):
        settings = carried_settings() if keep_view else None
        if state['explorer']:
            state['explorer']['cleanup']()
            state['explorer'] = None
        if state['stats']:
            state['stats']['cleanup']()
            state['stats'] = None
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
        summary = data['summary'] or {}
        cohort, values = cohort_for(chosen)
        state['stats'] = stats = add_stats_panel(viewer, name, data['graph'], summary,
                                                 {n: c for n, c in cohort.items() if n != name}, values)
        state['explorer'] = ui = add_explorer(viewer, data['mask'], data['graph'], data['scale'], name,
                                              folder=data['folder'], on_select=stats['mark'])
        status.value = (f"{names.index(name) + 1}/{len(names)}  {name} · {chosen} · "
                        f"{'computed now' if data['computed'] else 'loaded from saved results'}\n"
                        f"{summary.get('branches', '?')} branches, {summary.get('independent_cycles', '?')} loops, "
                        f"{summary.get('components', '?')} components")
        if settings and ui:
            for key, value in settings['widgets'].items():
                if value in getattr(ui[key], 'choices', [value]):
                    ui[key].value = value
            for key, value in settings['camera'].items():
                setattr(viewer.camera, key, value)
        else:
            viewer.reset_view()

    def cohort_for(chosen):
        """Saved results of every cube for one mask variant, read once and cached."""
        if chosen not in state['cohorts']:
            status.value = f'Reading saved results of all cubes ({chosen}) for the statistics panel…'
            folders = {n: results_folder(output, p, chosen) for n, p in zip(names, paths)}
            cohort = load_cohort({n: f for n, f in folders.items() if is_current(f)})
            state['cohorts'][chosen] = (cohort, cohort_values(cohort))
        return state['cohorts'][chosen]

    def cycle(step):
        """Load the previous or next cube, keeping mask variant, view, colouring and slab."""
        sample_box.value = names[(names.index(sample_box.value) + step) % len(names)]
        load(keep_view=True)

    load_button = PushButton(text='Load')
    load_button.changed.connect(lambda: load(keep_view=True))
    previous_button, next_button = PushButton(text='◀ Previous cube (,)'), PushButton(text='Next cube (.) ▶')
    previous_button.changed.connect(lambda: cycle(-1))
    next_button.changed.connect(lambda: cycle(1))
    data_panel = Container(widgets=[sample_box, variant_box, load_button,
                                    Container(layout='horizontal', widgets=[previous_button, next_button], labels=False),
                                    status])
    data_panel.native.setMaximumHeight(280)  # leave the right column to the explorer panel
    viewer.window.add_dock_widget(data_panel, name='Data', area='right')
    # ',' and '.' are unused by napari 0.5.6 viewer and layer keymaps.
    viewer.bind_key(',', lambda v: cycle(-1), overwrite=True)
    viewer.bind_key('.', lambda v: cycle(1), overwrite=True)
    state['controls'] = dict(sample=sample_box, variant=variant_box, load=load, cycle=cycle, status=status)
    load()
    if run:
        napari.run()
    return viewer, state
