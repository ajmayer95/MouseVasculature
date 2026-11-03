"""Read calibrated ImageJ masks and display separate ROIs in napari."""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile


def load_mask(path, scale=None):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != 'ZYX' or len(series.shape) != 3:
            raise ValueError(f'{path}: expected ZYX volume, got {series.axes} {series.shape}')
        data = series.asarray()
        values = np.unique(data)
        if len(values) > 2 or not np.all(np.isin(values, [0, 1, 255])):
            raise ValueError(f'{path}: expected binary mask with values 0/1 or 0/255')
        if scale is None:
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
        return (data > 0).astype(np.uint8), tuple(scale)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='TIFF file or folder of TIFF masks')
    parser.add_argument('--scale', type=float, nargs=3, metavar=('Z', 'Y', 'X'), help='Override voxel sizes in µm')
    parser.add_argument('--inventory', type=Path, help='Write CSV summary without opening napari')
    parser.add_argument('--grid', action='store_true', help='Show ROIs side by side (uses more GPU memory)')
    args = parser.parse_args(argv)
    paths = sorted(p for p in args.path.iterdir() if p.suffix.lower() in ('.tif', '.tiff')) if args.path.is_dir() else [args.path]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No TIFF files found')
    viewer = None
    rows = []
    if not args.inventory:
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
                                      rendering='iso_categorical', visible=args.grid or index == 0)
            layer.editable = False
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        with args.inventory.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        viewer.dims.ndisplay = 3
        viewer.grid.enabled = args.grid
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = 'um'
        viewer.reset_view()
        napari.run()
