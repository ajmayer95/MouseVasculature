# Mouse prostate vasculature

First milestone: view binary filament masks as calibrated 3D volumes in napari.
Future work: mask quality checks, skeleton and vessel graph extraction, radius estimation,
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

## Share the project

The local Git repository tracks code, documentation, and the inventory. To synchronize
between computers, create an empty private remote repository, then:

```sh
git remote add origin <your-repository-url>
git push -u origin main
```

On the laptop run `git clone <your-repository-url>` and follow setup above. TIFF data
must be transferred separately. No remote repository is configured automatically.

Rendering uses napari's [3D labels layer](https://napari.org/howtos/layers/labels.html).
