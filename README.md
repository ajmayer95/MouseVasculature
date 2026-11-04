# Mouse prostate vasculature

View binary filament masks, skeletons and vector centerlines in napari.
Future work: mask quality checks, radius estimation,
and a flow model with explicit inlet/outlet conditions. Flow modeling is not implemented yet.

## Setup on a laptop

Install Python 3.10 or 3.11, then in this repository:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate` instead.
Requirements pin the core packages to the versions used for the initial workflow.

## Open the masks

Copy the Filaments folder onto the laptop separately from Git. From the project root:

```sh
python -m mousevasculature /path/to/Filaments
```

On the original machine:

```sh
python -m mousevasculature /Users/amayer/Downloads/Filaments
```

The first ROI is visible; use layer eye icons to switch ROIs. Drag to rotate and
scroll to zoom. Toggle napari's 2D/3D button to inspect individual slices.
Masks are read-only in this viewer. Zero is background; nonzero voxels become label 1.
No smoothing, segmentation changes, or downsampling are applied.

Use one TIFF path to load only that ROI, or add `--grid` to compare all ROIs side by
side. Grid placement is a display arrangement, not anatomical registration.
All 15 volumes require about 290 MB of mask arrays, plus loading and GPU overhead.

ImageJ spacing and TIFF X/Y resolution are read per file in Z,Y,X order, in µm.
Files without supported calibration fail explicitly. Override with known sizes using
`--scale 1.8 1.8 1.8`. Do not infer physical sizes from array dimensions.

```sh
python -m mousevasculature /path/to/Filaments --inventory reports/masks.csv
python -m unittest discover -s tests
```

The inventory reports mask occupancy, not validated vascular volume or connectivity.
Original masks remain at their source location and are excluded from Git.

## Skeletonize and vectorize a specific filament

From the project folder, launch one ROI with the mask, centerline paths, endpoints,
and junctions. Centerlines render as plain strokes without arrowheads:

```sh
python -m mousevasculature --sample LS025_ROI1 --analyze
```

The default data folder is `~/Downloads/Filaments`. On another laptop, supply the
folder or set the `MOUSEVASC_DATA` environment variable:

```sh
python -m mousevasculature /path/to/Filaments --sample LS027_ROI3 --analyze
python -m mousevasculature /path/to/LS025_ROI1_Filaments.tif --analyze
```

Omit `--analyze` to view just the selected mask. Cyan paths are centerlines,
yellow points are endpoints, and magenta points are junction voxels. Toggle the
skeleton layer to see the voxel skeleton; it starts hidden. The mask and junction
points also start hidden so the centerlines are visible; toggle them for comparison.
All layers share the original physical calibration.

Analysis uses scikit-image's 3D Lee thinning, followed by tracing chains of
degree-two voxels under 26-neighbor connectivity. Redundant diagonal shortcuts are
removed when an intermediate skeleton voxel provides a local route. Paths retain every skeleton voxel
along the chain; there is no smoothing or pruning. Pure loops and isolated voxels
are retained. Adjacent junction voxels remain separate graph nodes, which can create
short edges around junctions. This is an initial voxel graph, not yet a validated
anatomical vessel graph or flow network. Disconnected masks remain disconnected.
Thinning operates in voxel space; physical spacing is applied to path lengths.

Each run exports `results/<filename>/skeleton.npz` and `graph.json`. The NPZ stores
the skeleton and voxel sizes. JSON stores nodes, edges, ZYX voxel coordinates,
calibration in µm, and path lengths in µm. Results are excluded from Git.
Re-running a sample replaces its previous results. To export without opening napari:

```sh
python -m mousevasculature --sample LS025_ROI1 --headless
python -m mousevasculature /path/to/Filaments --headless
```

## Vessel and network statistics

```sh
python -m mousevasculature --sample LS025_ROI1 --stats --color-by radius
python -m mousevasculature --sample LS025_ROI1 --stats --fill-holes --color-by radius
python -m mousevasculature /path/to/Filaments --stats --headless --fill-holes
```

### Plug small mask tunnels

```sh
python -m mousevasculature --sample LS025_ROI1 --close-tunnels --highlight small-loops
```

`--close-tunnels` (implies `--fill-holes`) plugs holes a voxel or two wide that pass through
vessels, keeping only patches that remove a tunnel without bridging vessels. Results go to
`<sample>_filled_closed`. See [statistics methods](docs/statistics.md#plug-small-tunnels).

### Highlight features in napari

```sh
python -m mousevasculature --sample LS025_ROI1 --fill-holes --highlight small-loops
python -m mousevasculature --sample LS025_ROI1 --fill-holes --highlight bridges boundary bifurcations
```

Every `--stats` run adds hidden highlight layers; `--highlight` chooses which start visible:
`small-loops` (red, loops <50 µm, with labelled markers), `loops` (blue, other looped
branches), `bridges` (gray, tree-like branches), `boundary` (orange, crop-touching),
`short` (yellow) and `bifurcations` (points colored by log2 Murray ratio, blue < 0 < red).
Highlights draw over the mask. With small loops present, a **small loops** panel on the
right steps through them (Next/Previous, or `n` / `Shift-N`), centres the camera, outlines
the current loop in green and prints its kind, perimeter, size, branch ids and flags.
Toggle napari's 2D button to see the slice through the loop centre; with
`--highlight small-loops` the mask starts visible so the tunnel can be compared with it.

Exports branch length, EDT radius, tortuosity, components, branching, graph cycles,
bridges, loop sizes, meshedness, bifurcation radius ratios and angles, tissue-to-vessel
distance, volume fraction, density and orientation statistics, plus CSV tables and
histograms. Small enclosed mask cavities can strongly affect skeletons; the optional
hole-filled version is saved separately for comparison. See [statistics methods and
QC guidance](docs/statistics.md) before interpreting values. Knot invariants are not
implemented.

## Share the project

The private repository is https://github.com/ajmayer95/MouseVasculature.
After changes, synchronize with:

```sh
git push
```

On the laptop run `git clone <your-repository-url>` and follow setup above. TIFF data
must be transferred separately.

Rendering uses napari's [3D labels layer](https://napari.org/howtos/layers/labels.html).
