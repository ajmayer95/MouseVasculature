"""Vessel density relative to the tissue where vessels can form, not the whole crop.

Prostate is glandular: lumens and epithelium are vessel-free, so length per crop volume mixes
vessel packing with how much of the crop is gland. Without an independent tissue channel the
vascular domain is estimated from the vessels themselves, with a "rolling ball":

- Local thickness. Every non-vessel voxel gets the radius of the largest vessel-free ball
  that contains it (in steps of RADIUS_STEP_UM). This also measures void sizes directly.
- Domain at radius R = vessels plus every voxel whose largest vessel-free ball is smaller
  than R. This equals a morphological closing of the vessel mask with a ball of radius R:
  gaps narrower than 2R count as tissue between vessels; larger voids are excluded.
  R is a size scale that must be chosen, so results are reported for a sweep of R values.
- Crop faces. Whether a void continues past a face is unknown, so everything is computed
  twice: assuming no vessels beyond the faces (voids can extend out: largest voids) and
  assuming vessels lie just beyond every face (voids stop there: smallest voids).
- External domain. A tissue mask from another channel can replace the estimate later:
  --domain-dir with one <cube>.tif or <cube>.npz (key "mask") per cube, same shape as the
  vessel mask, nonzero = tissue where vessels can form.

Density is vessel centerline length divided by domain volume, per cube and per 240 µm region
(the regions of loop_topology). The spread of region densities is split into between
animals, between cubes of one animal, and within cubes, as shares of the sum of squares.

Run:  python -m mousevasculature.tissue_domain [--results results] [--variant filled+closed]
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .loop_topology import CUBE_UM, REGION_UM, load_cubes, region_index

RADIUS_STEP_UM = 10.
MAX_RADIUS_UM = 400.
# Below ~80 µm the domain shrinks to a shell around each vessel (ordinary inter-vessel gaps are
# wider than 2R), so density there mostly reflects R. 50 µm is kept to show that failure.
SWEEP_R_UM = (50., 80., 100., 125., 150., 200.)
MIN_REGION_DOMAIN = .25   # regions with less domain than this are too unreliable to report
MODES = {'open': False, 'closed': True}  # value: assume vessels just beyond the crop faces


def local_thickness_radius(mask, scale, outside_is_vessel=False, step=RADIUS_STEP_UM, max_radius=MAX_RADIUS_UM):
    """Radius (µm, floored to `step`) of the largest vessel-free ball containing each voxel.

    Vessel voxels get 0. Ball centres must lie inside the crop. With outside_is_vessel the
    crop is padded with one layer of vessel, so no ball can extend past a face.
    """
    mask = np.asarray(mask, bool)
    if outside_is_vessel:
        mask = np.pad(mask, 1, constant_values=True)
    distance = ndi.distance_transform_edt(~mask, sampling=scale)
    radius = np.zeros(mask.shape, np.float32)
    r = step
    while r <= max_radius:
        centres = distance >= r
        if not centres.any():
            break
        covered = ndi.distance_transform_edt(~centres, sampling=scale) <= r
        radius[covered & ~mask] = r
        r += step
    if outside_is_vessel:
        radius = radius[1:-1, 1:-1, 1:-1]
    return radius


def segments_um(graph):
    """Midpoints and lengths (µm) of every centerline step of the branch graph."""
    scale = np.asarray(graph['voxel_size_um'], float)
    mids, lengths = [], []
    for e in graph['edges']:
        p = np.asarray(e['points_zyx'], float) * scale
        if len(p) > 1:
            mids.append((p[1:] + p[:-1]) / 2)
            lengths.append(np.linalg.norm(np.diff(p, axis=0), axis=1))
    if not mids:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(mids), np.concatenate(lengths)


def region_volumes(domain, scale, region_um=REGION_UM):
    """Domain volume (µm³) and total volume (µm³) per 240 µm region, keyed by region index."""
    n = int(CUBE_UM // region_um)
    offset = (CUBE_UM - n * region_um) / 2
    axis_index = [np.floor(((np.arange(size) + .5) * s - offset) / region_um).astype(int)
                  for size, s in zip(domain.shape, scale)]
    voxel = float(np.prod(scale))
    dom, tot = {}, {}
    for key in np.ndindex(n, n, n):
        sel = np.ix_(*[np.flatnonzero(a == k) for a, k in zip(axis_index, key)])
        block = domain[sel]
        dom[key], tot[key] = float(block.sum()) * voxel, block.size * voxel
    return dom, tot


def region_lengths(mids, lengths, region_um=REGION_UM):
    out = {}
    for m, l in zip(mids, lengths):
        key = region_index(m, region_um)
        if key is not None:
            out[key] = out.get(key, 0.) + l
    return out


def void_components(void, scale):
    """Connected vessel-free voids: volumes (µm³) and whether each touches a crop face."""
    labels, n = ndi.label(void, structure=np.ones((3, 3, 3)))
    if not n:
        return []
    volumes = np.bincount(labels.ravel())[1:] * float(np.prod(scale))
    faces = np.zeros(n + 1, bool)
    for axis in range(3):
        for index in (0, -1):
            faces[np.unique(np.take(labels, index, axis=axis))] = True
    return [dict(volume_um3=float(v), touches_face=bool(faces[i + 1])) for i, v in enumerate(volumes)]


def analyse_domain(name, domain, vessel, scale, mids, lengths):
    """Density within one domain mask, per cube and per region."""
    voxel = float(np.prod(scale))
    domain_um3 = domain.sum() * voxel
    total_um3 = domain.size * voxel
    dom_r, tot_r = region_volumes(domain, scale)
    len_r = region_lengths(mids, lengths)
    regions = []
    for key, dv in dom_r.items():
        frac = dv / tot_r[key] if tot_r[key] else 0.
        regions.append(dict(cube=name, region=f'z{key[0]}y{key[1]}x{key[2]}', domain_fraction=frac,
                            density_crop_mm_per_mm3=len_r.get(key, 0.) / tot_r[key] * 1e6 if tot_r[key] else None,
                            density_domain_mm_per_mm3=(len_r.get(key, 0.) / dv * 1e6) if frac >= MIN_REGION_DOMAIN else None))
    return dict(cube=name, domain_fraction=domain_um3 / total_um3,
                vessel_fraction_of_domain=vessel.sum() * voxel / domain_um3 if domain_um3 else None,
                density_crop_mm_per_mm3=lengths.sum() / total_um3 * 1e6,
                density_domain_mm_per_mm3=lengths.sum() / domain_um3 * 1e6 if domain_um3 else None), regions


def sum_of_squares_shares(rows, field):
    """Share of the total sum of squares between animals, between cubes within animals, within cubes."""
    data = [(r['cube'][:5], r['cube'], r[field]) for r in rows if r[field] is not None]
    if len(data) < 3:
        return None
    y = np.array([d[2] for d in data], float)
    grand = y.mean()
    total = np.sum((y - grand) ** 2)
    if total == 0:
        return None
    animal_mean = {a: y[[d[0] == a for d in data]].mean() for a in {d[0] for d in data}}
    cube_mean = {c: y[[d[1] == c for d in data]].mean() for c in {d[1] for d in data}}
    between_animals = sum((animal_mean[d[0]] - grand) ** 2 for d in data)
    between_cubes = sum((cube_mean[d[1]] - animal_mean[d[0]]) ** 2 for d in data)
    within = sum((v - cube_mean[d[1]]) ** 2 for d, v in zip(data, y))
    return dict(animals=between_animals / total, cubes_within_animal=between_cubes / total, within_cube=within / total,
                cv=float(y.std(ddof=1) / y.mean()), n_regions=len(y))


def load_external_domain(directory, name, shape):
    directory = Path(directory)
    for path in (directory / f'{name}.npz', directory / f'{name}.tif', directory / f'{name}.tiff'):
        if path.exists():
            if path.suffix == '.npz':
                domain = np.load(path)['mask']
            else:
                import tifffile
                domain = tifffile.imread(path)
            domain = np.asarray(domain) != 0
            if domain.shape != tuple(shape):
                raise SystemExit(f'{path}: shape {domain.shape} does not match vessel mask {tuple(shape)}')
            return domain
    raise SystemExit(f'No external domain for {name} in {directory} (expected {name}.npz or {name}.tif)')


def analyse_cube(args):
    name, folder, graph, domain_dir = args
    data = np.load(Path(folder) / 'analysis_mask.npz')
    vessel, scale = data['mask'].astype(bool), np.asarray(data['voxel_size_um'], float)
    mids, lengths = segments_um(graph)
    cube_rows, region_rows, voids, thickness = [], [], [], {}
    if domain_dir:
        domain = load_external_domain(domain_dir, name, vessel.shape) | vessel
        row, regions = analyse_domain(name, domain, vessel, scale, mids, lengths)
        cube_rows.append(dict(row, mode='external', r_um=None))
        region_rows += [dict(r, mode='external', r_um=None) for r in regions]
        return cube_rows, region_rows, voids, thickness
    for mode, outside in MODES.items():
        radius = local_thickness_radius(vessel, scale, outside)
        values, counts = np.unique(radius[~vessel], return_counts=True)
        thickness[mode] = {float(v): int(c) for v, c in zip(values, counts)}
        for r in SWEEP_R_UM:
            domain = vessel | (radius < r)
            row, regions = analyse_domain(name, domain, vessel, scale, mids, lengths)
            cube_rows.append(dict(row, mode=mode, r_um=r))
            region_rows += [dict(x, mode=mode, r_um=r) for x in regions]
            voids += [dict(v, cube=name, mode=mode, r_um=r) for v in void_components(~domain, scale)]
        if mode == 'open':
            np.savez_compressed(Path(folder) / 'local_thickness_radius.npz', radius_um=radius.astype(np.float32),
                                step_um=RADIUS_STEP_UM, assumption='no vessels beyond the crop faces')
    return cube_rows, region_rows, voids, thickness


# ---- figures -------------------------------------------------------------------------
def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def figure_sweep(cube_rows, region_rows, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    cubes = sorted({r['cube'] for r in cube_rows})
    colours = {a: c for a, c in zip(sorted({c[:5] for c in cubes}), ('#1b9e77', '#d95f02', '#7570b3', '#e7298a'))}
    ax = axes[0]
    for cube in cubes:
        rows = sorted((r for r in cube_rows if r['cube'] == cube and r['mode'] == 'open'), key=lambda r: r['r_um'])
        closed = sorted((r for r in cube_rows if r['cube'] == cube and r['mode'] == 'closed'), key=lambda r: r['r_um'])
        x = [0] + [r['r_um'] for r in rows]
        ax.plot(x, [rows[0]['density_crop_mm_per_mm3']] + [r['density_domain_mm_per_mm3'] for r in rows],
                color=colours[cube[:5]], marker='o', ms=3, lw=1, alpha=.8)
        ax.plot(x[1:], [r['density_domain_mm_per_mm3'] for r in closed], color=colours[cube[:5]], lw=.6, ls=':', alpha=.8)
    for a, c in colours.items():
        ax.plot([], [], color=c, label=a)
    ax.plot([], [], color='grey', ls=':', label='voids stop at crop faces')
    ax.set_xticks([0, *SWEEP_R_UM], ['whole\ncrop', *[f'{r:.0f}' for r in SWEEP_R_UM]])
    ax.set_xlabel('rolling-ball radius R (µm)'), ax.set_ylabel('length density in domain (mm/mm³)')
    ax.set_title('Density within the vascular domain, per cube', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    ax = axes[1]
    for cube in cubes:
        rows = sorted((r for r in cube_rows if r['cube'] == cube and r['mode'] == 'open'), key=lambda r: r['r_um'])
        closed = sorted((r for r in cube_rows if r['cube'] == cube and r['mode'] == 'closed'), key=lambda r: r['r_um'])
        ax.plot([r['r_um'] for r in rows], [1 - r['domain_fraction'] for r in rows], color=colours[cube[:5]], marker='o', ms=3, lw=1, alpha=.8)
        ax.plot([r['r_um'] for r in closed], [1 - r['domain_fraction'] for r in closed], color=colours[cube[:5]], lw=.6, ls=':', alpha=.8)
    ax.set_xlabel('rolling-ball radius R (µm)'), ax.set_ylabel('fraction of crop outside the domain (voids)')
    ax.set_ylim(0, 1), ax.set_title('Vessel-free voids wider than 2R', fontsize=9)
    _style(ax)

    ax = axes[2]
    base = sum_of_squares_shares([r for r in region_rows if r['mode'] == 'open' and r['r_um'] == SWEEP_R_UM[0]],
                                 'density_crop_mm_per_mm3')
    labels, shares = ['whole crop'], [base]
    for r in SWEEP_R_UM:
        labels.append(f'R={r:.0f}')
        shares.append(sum_of_squares_shares([x for x in region_rows if x['mode'] == 'open' and x['r_um'] == r],
                                            'density_domain_mm_per_mm3'))
    bottom = np.zeros(len(labels))
    for key, colour, label in (('animals', '#08519c', 'between animals'), ('cubes_within_animal', '#6baed6', 'between cubes'),
                               ('within_cube', '#c6dbef', 'within cubes')):
        vals = np.array([s[key] if s else np.nan for s in shares])
        ax.bar(labels, vals, bottom=bottom, color=colour, label=label)
        bottom += np.nan_to_num(vals)
    for i, s in enumerate(shares):
        if s:
            ax.text(i, 1.02, f"CV {s['cv']:.0%}\nn={s['n_regions']}", ha='center', fontsize=6.5)
    ax.set_ylim(0, 1.15), ax.set_ylabel('share of sum of squares, 240 µm regions')
    ax.set_title('Where region-to-region variation comes from', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    _style(ax)
    fig.savefig(Path(out) / 'figA_domain_density_sweep.png', dpi=170)
    fig.savefig(Path(out) / 'figA_domain_density_sweep.pdf')
    plt.close(fig)


def figure_voids(thickness, voids, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    ax = axes[0]
    for mode, style in (('open', '-'), ('closed', ':')):
        pooled = {}
        for per_cube in thickness.values():
            for r, c in per_cube.get(mode, {}).items():
                pooled[r] = pooled.get(r, 0) + c
        radii = np.array(sorted(pooled))
        counts = np.array([pooled[r] for r in radii], float)
        tail = counts[::-1].cumsum()[::-1] / counts.sum()
        ax.step(2 * radii, tail, where='post', color='#08519c', ls=style,
                label='no vessels beyond faces' if mode == 'open' else 'vessels just beyond faces')
    ax.set_xlabel('local vessel-free diameter (µm)'), ax.set_ylabel('fraction of non-vessel volume this wide or wider')
    ax.set_title('Local thickness of vessel-free space (all cubes)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for r, colour in zip(SWEEP_R_UM, plt.get_cmap('Blues')(np.linspace(.3, 1, len(SWEEP_R_UM)))):
        v = [x['volume_um3'] / 1e9 for x in voids if x['mode'] == 'open' and x['r_um'] == r]
        if v:
            v = np.sort(v)[::-1]
            ax.plot(np.arange(1, len(v) + 1), v, color=colour, label=f'R={r:.0f} µm: {len(v)} voids, '
                    f'{np.mean([x["touches_face"] for x in voids if x["mode"] == "open" and x["r_um"] == r]):.0%} touch a face')
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel('void rank'), ax.set_ylabel('void volume (mm³)')
    ax.set_title('Sizes of separate voids (all cubes)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.savefig(Path(out) / 'figB_void_sizes.png', dpi=170)
    fig.savefig(Path(out) / 'figB_void_sizes.pdf')
    plt.close(fig)


def figure_slices(cubes, results, variant_suffix, out, radii=(50., 100.)):
    import matplotlib.pyplot as plt
    names = list(cubes)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10), layout='constrained')
    for ax, name in zip(axes.flat, names):
        folder = Path(results) / f'{name}_Filaments{variant_suffix}'
        vessel = np.load(folder / 'analysis_mask.npz')['mask'].astype(bool)
        radius = np.load(folder / 'local_thickness_radius.npz')['radius_um']
        z = vessel.shape[0] // 2
        im = ax.imshow(2 * radius[z], cmap='Greys', vmin=0, vmax=400)
        ax.imshow(np.ma.masked_where(~vessel[z], vessel[z]), cmap='autumn', interpolation='nearest')
        for r, colour in zip(radii, ('#1f78b4', '#33a02c')):
            ax.contour(radius[z] >= r, levels=[.5], colors=colour, linewidths=.8)
        ax.set_title(name, fontsize=8), ax.set_xticks([]), ax.set_yticks([])
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    fig.colorbar(im, ax=axes, shrink=.5, label='local vessel-free diameter (µm); red = vessel')
    fig.suptitle(f'Middle Z slice. Outlines: voids excluded at R = {radii[0]:.0f} µm (blue) and {radii[1]:.0f} µm (green)', fontsize=10)
    fig.savefig(Path(out) / 'figC_domain_slices.png', dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(results='results', variant_suffix='_filled_closed', out=None, domain_dir=None, workers=None):
    import matplotlib
    matplotlib.use('Agg')
    out = Path(out or Path(results) / ('tissue_domain_external' if domain_dir else 'tissue_domain'))
    out.mkdir(parents=True, exist_ok=True)
    cubes = load_cubes(results, variant_suffix)
    if not cubes:
        raise SystemExit(f'No saved results matching *_Filaments{variant_suffix} in {results}')
    jobs = [(name, Path(results) / f'{name}_Filaments{variant_suffix}', graph, domain_dir)
            for name, (graph, _) in cubes.items()]
    cube_rows, region_rows, voids, thickness = [], [], [], {}
    with ProcessPoolExecutor(max_workers=workers or min(8, len(jobs))) as pool:
        for (name, *_), (c, r, v, t) in zip(jobs, pool.map(analyse_cube, jobs)):
            cube_rows += c
            region_rows += r
            voids += v
            thickness[name] = t
    write_csv(out / 'cubes.csv', cube_rows)
    write_csv(out / 'regions.csv', region_rows)
    write_csv(out / 'voids.csv', voids)
    summary = dict(variant=variant_suffix, external_domain=str(domain_dir) if domain_dir else None,
                   region_um=REGION_UM, min_region_domain_fraction=MIN_REGION_DOMAIN, sweep=[])
    modes = ['external'] if domain_dir else list(MODES)
    for mode in modes:
        for r in ([None] if domain_dir else SWEEP_R_UM):
            rows = [x for x in cube_rows if x['mode'] == mode and x['r_um'] == r]
            regs = [x for x in region_rows if x['mode'] == mode and x['r_um'] == r]
            dens = np.array([x['density_domain_mm_per_mm3'] for x in rows], float)
            summary['sweep'].append(dict(
                mode=mode, r_um=r, domain_fraction_mean=float(np.mean([x['domain_fraction'] for x in rows])),
                density_domain_mean=float(dens.mean()), density_domain_cv_between_cubes=float(dens.std(ddof=1) / dens.mean()),
                regions_crop=sum_of_squares_shares(regs, 'density_crop_mm_per_mm3'),
                regions_domain=sum_of_squares_shares(regs, 'density_domain_mm_per_mm3')))
    first_r = None if domain_dir else SWEEP_R_UM[0]
    crop = np.array([x['density_crop_mm_per_mm3'] for x in cube_rows if x['mode'] == modes[0] and x['r_um'] == first_r])
    summary['density_crop_mean'] = float(crop.mean())
    summary['density_crop_cv_between_cubes'] = float(crop.std(ddof=1) / crop.mean())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not domain_dir:
        figure_sweep(cube_rows, region_rows, out)
        figure_voids(thickness, voids, out)
        figure_slices(cubes, results, variant_suffix, out)
    return summary, out


def main(argv=None):
    from .pipeline import VARIANTS
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--results', default='results')
    parser.add_argument('--variant', default='filled+closed', choices=list(VARIANTS))
    parser.add_argument('--out', default=None)
    parser.add_argument('--domain-dir', default=None,
                        help='folder with one tissue mask per cube (<cube>.npz key "mask", or <cube>.tif); replaces the estimate')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args(argv)
    summary, out = run(args.results, VARIANTS[args.variant], args.out, args.domain_dir, args.workers)
    print(json.dumps(summary, indent=2))
    print(f'Figures and tables in {out}')


if __name__ == '__main__':
    main()
