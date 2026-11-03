# Exploratory vascular statistics

From the project root, activate the environment and measure one ROI:

```sh
source .venv/bin/activate
python -m mousevasculature --sample LS025_ROI1 --stats --color-by radius
```

Color choices are `radius`, `length`, or `component`. Color maps use the measured
branch mean radius in µm or branch length in µm; component IDs use categorical
colors. Centerlines have no arrowheads and
do not imply a flow direction. Results are computed before the viewer opens.

## Compare original masks with filled cavities

These exports contain small enclosed background cavities. In LS025_ROI1, filling
enclosed holes adds 1,246 voxels (0.65% of the original foreground), but substantially
changes the skeleton: small cavities can produce sheets and large junction clusters.
Use an explicit preprocessing option to examine its effect:

```sh
python -m mousevasculature --sample LS025_ROI1 --stats --fill-holes --color-by radius
```

This fills only 3D background components enclosed by foreground, using 6-neighbor
background connectivity. It does not reconnect separated vessels or fill spaces
connected to the crop boundary. It can remove real enclosed cavities, so keep both
versions for quality control. Original TIFFs are never modified. Processed masks
are saved as `analysis_mask.npz` in a separate `<sample>_filled` results folder.
Both original and filled masks remain available as hidden napari layers for comparison.

## Batch export

```sh
python -m mousevasculature /path/to/Filaments --stats --headless
python -m mousevasculature /path/to/Filaments --stats --headless --fill-holes
```

`results/summary.csv` compares every completed statistics export found in this
output folder. Original and filled variants have separate rows, labeled
`analysis_variant`. `results/overview.png` plots the ROI comparisons. Each sample folder contains:

- `branches.csv`: one row per maximal branch, with length, chord distance,
  tortuosity, radius estimates, component, boundary flag, short-branch flag and bridge flag.
- `components.csv`: component size, total branch length and independent cycle count.
- `statistics.json`: distributions, network metrics, mask measurements, QC messages,
  preprocessing provenance and library versions. Undefined values are JSON null or
  empty CSV cells, not zero.
- `statistics.png`: histograms for branch length, mean radius, tortuosity and node degree.
- `branch_graph.json`: measured graph with centerline coordinates and a per-vertex
  radius profile. `graph.json` retains the original uncontracted voxel graph.
- `skeleton.npz`: binary skeleton and calibrated voxel sizes.

Use `--output some/folder` to keep different experiments separate. Re-running the
same sample and variant replaces its files. Results stay outside Git.

## Definitions and practical limits

**Branches.** Contiguous junction voxels connected by direct voxel edges are grouped
into a junction region. Their internal voxel edges are removed from the branch
network; the raw graph preserves them. Degree-two regions are then merged into
maximal branches. Where a branch passes through such a region, its connector follows
the shortest calibrated path through original junction voxels. Loops, parallel
branches, disconnected components and isolated nodes are retained. Pure cycles have
one anchor and a self-loop. At higher-degree junctions, branches terminate at their
attachment voxels; their lengths exclude the junction interior. Region centroids
shown as points are not necessarily on a branch endpoint. Large regions (>27 voxels)
trigger a QC message. No short branches are automatically pruned.

**Lengths and tortuosity.** Length is the sum of calibrated 3D centerline steps.
Chord is the distance between attachment endpoints. Tortuosity is length/chord;
a straight branch is 1. Closed/self-loop branches have undefined tortuosity and
are excluded from its distribution. Digital steps can bias length and tortuosity;
there is no smoothing. Branches shorter than three times the smallest voxel spacing
are flagged. Any path touching an ROI face is crop-censored and flagged. Interior-only
length distributions are also included. Endpoints are graph endpoints, not identified
anatomical inlets or outlets.

**Radius.** The calibrated Euclidean distance transform estimates distance from a
centerline voxel center to the nearest background voxel center. It assumes a filled
vessel mask and does not measure lumen radius reliably from wall-only masks. Radius
mean is weighted by centerline segment length using trapezoidal integration; radius
min/max/median and per-vertex profiles are also saved. Summary distributions treat
branches equally; `length_weighted_radius_um` weights branches by length. EDT has
voxel discretization error and can overestimate radius near bifurcations. Background
padding gives finite values at ROI faces but biases radius near cropped vessels.
Values near the 1.8 µm voxel size are poorly resolved. No subvoxel correction is applied.

**Topology.** Connected components, endpoints, branching nodes, degree distribution,
isolated nodes, self-loops, bridges, and independent graph cycles are included.
Cycle rank is `edges - nodes + components` on an undirected multigraph, retaining
parallel branches and self-loops. Bridges disconnect a component when removed.
Raw and contracted cycle ranks can differ because collapsing junction regions removes
internal voxel cycles. Mask Euler characteristic is measured separately with
26-neighbor foreground connectivity; in 3D it is components minus tunnels plus enclosed
cavities. Do not equate mask Euler characteristic with graph cycle rank when cavities
or skeleton sheets exist.
For the filled variant, the export also compares graph cycle rank with the mask's
tunnel count (`mask components - Euler characteristic`) and flags discrepancies.

**Other metrics.** Foreground fraction/volume, largest mask-component fraction,
largest graph-component length fraction, length density (mm/mm³), branch density
(1/mm³), and orientation anisotropy are included. Orientation uses an arc-length-weighted
tensor of unit centerline directions; its eigenvalues sum to 1 for a nonempty graph.
Anisotropy is largest minus smallest eigenvalue (0 for isotropic orientation, 1 for
perfect alignment). Total branch length excludes higher-degree junction interiors;
it should not be interpreted as a complete anatomical vascular length before QC.

**Knotting.** A cycle is not evidence of a knot. These statistics do not compute knot
types or linking numbers. That needs explicit closed-curve selection, treatment of
intersections/shared junctions, checks for self-intersections, and a closure policy
for cropped/open curves. Graph cycles are the starting point for that later analysis.

Compare ROI distributions first. Filenames suggest ROIs grouped within specimens;
confirm the specimen/group mapping before statistical comparisons. Multiple ROIs
from one mouse are not independent biological replicates. No group p-values are
computed automatically.

Methods: [SciPy distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html),
[scikit-image skeletonization](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize),
[NetworkX bridges](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bridges.bridges.html).
