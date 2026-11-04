"""Read calibrated ImageJ masks and display separate ROIs in napari."""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import tifffile


CALIBRATION_FILE = Path(__file__).resolve().parent.parent / 'calibration.json'


def calibration_override(path, calibration_file=CALIBRATION_FILE):
    """Voxel size (Z, Y, X µm) and source note from calibration.json for this file, or (None, None)."""
    import fnmatch
    import json
    try:
        entries = json.loads(Path(calibration_file).read_text()).get('files', {})
    except FileNotFoundError:
        return None, None
    for pattern, entry in entries.items():
        if fnmatch.fnmatch(Path(path).name, pattern):
            return tuple(float(v) for v in entry['voxel_size_um_zyx']), entry.get('source', pattern)
    return None, None


def voxel_size(path, scale=None):
    """Voxel size in µm (Z, Y, X): explicit scale, else calibration.json, else ImageJ TIFF metadata."""
    if scale is None:
        scale, _ = calibration_override(path)
    if scale is None:
        with tifffile.TiffFile(path) as tif:
            meta = tif.imagej_metadata or {}
            unit = str(meta.get('unit', '')).replace('\\u00B5', 'µ')
            if unit not in ('µm', 'um', 'micron', 'microns') or 'spacing' not in meta:
                raise ValueError(f'{path}: missing micrometer calibration; supply --scale Z Y X')
            page = tif.pages[0]
            def pixel_size(tag):
                numerator, denominator = page.tags[tag].value
                return denominator / numerator
            scale = (float(meta['spacing']), pixel_size('YResolution'), pixel_size('XResolution'))
    if len(scale) != 3 or not all(np.isfinite(v) and v > 0 for v in scale):
        raise ValueError('Voxel sizes must be three finite positive numbers')
    return tuple(float(v) for v in scale)


def load_mask(path, scale=None):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != 'ZYX' or len(series.shape) != 3:
            raise ValueError(f'{path}: expected ZYX volume, got {series.axes} {series.shape}')
        data = series.asarray()
    values = np.unique(data)
    if len(values) > 2 or not np.all(np.isin(values, [0, 1, 255])):
        raise ValueError(f'{path}: expected binary mask with values 0/1 or 0/255')
    return (data > 0).astype(np.uint8), voxel_size(path, scale)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, nargs='?',
                        default=Path(os.environ.get('MOUSEVASC_DATA', Path.home() / 'Downloads' / 'Filaments')),
                        help='TIFF file or folder (default: MOUSEVASC_DATA or ~/Downloads/Filaments)')
    parser.add_argument('--sample', help='Select a filename stem, e.g. LS025_ROI1')
    parser.add_argument('--analyze', action='store_true', help='Skeletonize, vectorize and display centerlines')
    parser.add_argument('--stats', action='store_true', help='Measure branches and topology; save CSV, JSON and histograms')
    parser.add_argument('--fill-holes', action='store_true', help='Fill enclosed 3D background cavities before analysis; save separately')
    parser.add_argument('--close-tunnels', type=int, nargs='?', const=1, default=None, metavar='RADIUS',
                        help='Plug small mask tunnels (closing radius in voxels, default 1) when this '
                             'removes tunnels without bridging vessels; implies --fill-holes')
    parser.add_argument('--color-by', choices=('length', 'radius', 'component'),
                        help='Color measured branches by a statistic (implies --stats)')
    parser.add_argument('--highlight', nargs='+', default=[], metavar='FEATURE',
                        choices=('small-loops', 'loops', 'bridges', 'boundary', 'short', 'bifurcations'),
                        help='With --stats, show these highlight layers: small-loops, loops, bridges, '
                             'boundary, short, bifurcations (all are added, hidden, otherwise)')
    parser.add_argument('--explore', action='store_true',
                        help='Open the interactive network explorer, reusing saved results when current')
    parser.add_argument('--variant', choices=('original', 'filled', 'filled+closed'),
                        help='Mask variant to open with --explore (default filled+closed); switchable in the app')
    parser.add_argument('--headless', action='store_true', help='Analyze and export without a GUI')
    parser.add_argument('--output', type=Path, default=Path('results'), help='Analysis output directory')
    parser.add_argument('--scale', type=float, nargs=3, metavar=('Z', 'Y', 'X'), help='Override voxel sizes in µm')
    parser.add_argument('--inventory', type=Path, help='Write CSV summary without opening napari')
    parser.add_argument('--grid', action='store_true', help='Show ROIs side by side (uses more GPU memory)')
    args = parser.parse_args(argv)
    if args.explore:
        if args.headless or args.inventory or args.grid:
            parser.error('--explore opens a GUI; do not combine with --headless, --inventory or --grid')
        variant = args.variant or ('filled+closed' if args.close_tunnels or not args.fill_holes else 'filled')
        from .app import explore
        return explore(args.path, args.output, sample=args.sample, variant=variant,
                       scale=args.scale, close_radius=args.close_tunnels or 1)
    if args.color_by or args.highlight:
        args.stats = True
    if args.stats:
        args.analyze = True
    if args.close_tunnels is not None:
        if args.close_tunnels < 1:
            parser.error('--close-tunnels radius must be at least 1 voxel')
        args.fill_holes = True
    if args.fill_holes:
        args.analyze = True
    if args.headless:
        args.analyze = True
    if args.inventory and args.analyze:
        parser.error('--inventory cannot be combined with analysis')
    if args.grid and args.analyze:
        parser.error('Analyze one sample at a time; grid is for masks only')
    paths = sorted(p for p in args.path.iterdir() if p.suffix.lower() in ('.tif', '.tiff')) if args.path.is_dir() else [args.path]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No TIFF files found')
    if args.sample:
        paths = [p for p in paths if p.stem in (args.sample, args.sample + '_Filaments')]
        if len(paths) != 1:
            parser.error('--sample must match exactly one file, e.g. LS025_ROI1')
    viewer = None
    rows = []
    summaries = []
    if not args.inventory and not args.headless:
        import napari
        viewer = napari.Viewer(title='Mouse prostate vasculature', ndisplay=2)
        viewer.dims.axis_labels = ('Z', 'Y', 'X')
    for index, path in enumerate(paths):
        try:
            mask, scale = load_mask(path, args.scale)
        except (ValueError, KeyError) as exc:
            parser.error(str(exc))
        count = int(np.count_nonzero(mask))
        rows.append(dict(file=path.name, z=mask.shape[0], y=mask.shape[1], x=mask.shape[2],
                         voxel_z_um=scale[0], voxel_y_um=scale[1], voxel_x_um=scale[2],
                         foreground_voxels=count, foreground_fraction=count / mask.size))
        print(f'{path.name}: {mask.shape}, voxel µm {scale}, foreground {count:,}', flush=True)
        if viewer is not None:
            layer = viewer.add_labels(mask, name=path.stem, scale=scale, opacity=1,
                                      rendering='iso_categorical')
            layer.visible = args.grid or index == 0  # set after creation (napari 0.5.6 thumbnail bug)
            layer.editable = False
        if args.analyze:
            from .analysis import add_analysis_layers
            from .pipeline import run_analysis
            analysis_mask, skeleton, graph, summary = run_analysis(
                path, mask, scale, args.output, args.fill_holes, args.close_tunnels, args.stats)
            if summary is not None:
                summaries.append(summary)
            if viewer is not None:
                layer.opacity = .2
                layer.visible = False
                if args.fill_holes:
                    filled_layer = viewer.add_labels(analysis_mask, name=f'{path.stem} ' + ('repaired' if args.close_tunnels else 'filled') + ' mask',
                                                      scale=scale, opacity=.2)
                    filled_layer.visible = False
                    filled_layer.editable = False
                add_analysis_layers(viewer, skeleton, graph, scale, path.stem,
                                    visible=index == 0, color_by=args.color_by)
                if args.stats:
                    from .features import add_feature_layers
                    add_feature_layers(viewer, graph, scale, path.stem,
                                       highlight=args.highlight if index == 0 else (), browser=False)
                    if index == 0:
                        from .explorer import add_explorer
                        print('Building clickable vessel labels...', flush=True)
                        stem = path.stem + ('_filled' if args.fill_holes else '') + ('_closed' if args.close_tunnels else '')
                        add_explorer(viewer, analysis_mask, graph, scale, path.stem, folder=args.output / stem)
    if summaries:
        from .statistics import save_comparison
        save_comparison(args.output, summaries)
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        with args.inventory.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif viewer is not None:
        viewer.dims.ndisplay = 3
        viewer.grid.enabled = args.grid
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = 'um'
        viewer.reset_view()
        napari.run()
