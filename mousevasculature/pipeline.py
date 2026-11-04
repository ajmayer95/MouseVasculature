"""Run or reload one sample's preprocessing, skeleton, graph and statistics."""
import json
from pathlib import Path

import numpy as np

VARIANTS = {'original': '', 'filled': '_filled', 'filled+closed': '_filled_closed'}
SCHEMA_VERSION = 4


def variant_options(variant, close_radius=1):
    """(fill_holes, close_tunnels radius or None) for a variant name."""
    return dict(original=(False, None), filled=(True, None), **{'filled+closed': (True, close_radius)})[variant]


def run_analysis(path, mask, scale, output, fill_holes=False, close_tunnels=None, stats=True):
    """Preprocess, skeletonize, vectorize and (optionally) measure one mask; save to output/<stem>.

    Returns (analysis_mask, skeleton, graph, summary). With stats, `graph` is the measured
    branch graph; otherwise the raw voxel graph. `summary` is None without stats.
    """
    from .analysis import analyze, save_analysis
    from .preprocess import close_tunnels as close, fill_enclosed_holes
    path, output = Path(path), Path(output)
    count = int(np.count_nonzero(mask))
    print(f'Skeletonizing and tracing {path.name}...', flush=True)
    analysis_mask, closing = mask, {}
    if fill_holes:
        analysis_mask = fill_enclosed_holes(mask)
    if close_tunnels:
        filled_voxels = int(analysis_mask.sum())
        analysis_mask, closing = close(analysis_mask, close_tunnels)
        # Plugging a tunnel can enclose a cavity; fill again.
        analysis_mask = fill_enclosed_holes(analysis_mask)
        closing['voxels_added_after_fill'] = int(analysis_mask.sum()) - filled_voxels
        print(f"Closed tunnels: {closing['accepted_patches']} patches accepted, "
              f"{closing['rejected_bridging_patches']} bridging patches rejected, "
              f"{closing['voxels_added_after_fill']} voxels added", flush=True)
    analysis_mask = analysis_mask.astype(np.uint8)
    stem = path.stem + ('_filled' if fill_holes else '') + ('_closed' if close_tunnels else '')
    skeleton, graph = analyze(analysis_mask, scale)
    graph['preprocessing'] = dict(fill_enclosed_holes=fill_holes, close_tunnels=bool(close_tunnels), **closing,
                                  original_foreground_voxels=count,
                                  added_voxels=int(analysis_mask.sum()) - count,
                                  background_connectivity=6 if fill_holes else None)
    save_analysis(output, stem, skeleton, graph, path)
    if fill_holes:
        np.savez_compressed(output / stem / 'analysis_mask.npz', mask=analysis_mask, voxel_size_um=scale)
    summary = None
    if stats:
        from .statistics import save_statistics
        summary, graph = save_statistics(output / stem, path, analysis_mask, skeleton, graph)
    return analysis_mask, skeleton, graph, summary


def results_folder(output, path, variant):
    return Path(output) / (Path(path).stem + VARIANTS[variant])


def is_current(folder, scale=None):
    """True when a results folder holds a complete statistics export of the current schema.

    With `scale`, the saved voxel size must also match (so a calibration change forces a re-run).
    """
    folder = Path(folder)
    try:
        summary = json.loads((folder / 'statistics.json').read_text())
    except (OSError, ValueError):
        return False
    needed = ['branch_graph.json', 'skeleton.npz'] + (['analysis_mask.npz'] if summary.get('analysis_variant') != 'original' else [])
    if scale is not None and not np.allclose(summary.get('voxel_size_um', []), scale, rtol=1e-6):
        return False
    return summary.get('schema_version', 0) >= SCHEMA_VERSION and all((folder / f).exists() for f in needed)


def load_or_run(path, variant, output, scale=None, close_radius=1):
    """Reload saved results for a sample and variant, or compute them if missing or outdated.

    Returns dict(mask, skeleton, graph, scale, summary, folder, computed).
    """
    from .viewer import load_mask, voxel_size
    folder = results_folder(output, path, variant)
    if is_current(folder, voxel_size(path, scale)):
        print(f'Loading saved results from {folder}', flush=True)
        graph = json.loads((folder / 'branch_graph.json').read_text())
        if 'loops' in graph:
            summary = json.loads((folder / 'statistics.json').read_text())
            skeleton = np.load(folder / 'skeleton.npz')['skeleton']
            if variant == 'original':
                mask, scale = load_mask(path, scale)
            else:
                saved = np.load(folder / 'analysis_mask.npz')
                mask, scale = saved['mask'], tuple(saved['voxel_size_um'])
            return dict(mask=mask, skeleton=skeleton, graph=graph, scale=tuple(scale), summary=summary,
                        folder=folder, computed=False)
    mask, scale = load_mask(path, scale)
    fill, close = variant_options(variant, close_radius)
    analysis_mask, skeleton, graph, summary = run_analysis(path, mask, scale, output, fill, close)
    return dict(mask=analysis_mask, skeleton=skeleton, graph=graph, scale=tuple(scale), summary=summary,
                folder=folder, computed=True)
