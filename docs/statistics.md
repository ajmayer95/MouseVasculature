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

## Plug small tunnels

```sh
python -m mousevasculature --sample LS025_ROI1 --close-tunnels --highlight small-loops
python -m mousevasculature /path/to/Filaments --stats --headless --close-tunnels
```

Even after hole filling, these masks contain small tunnels: holes a few voxels wide
that pass through a vessel. Each becomes a graph cycle (mostly two parallel branches
around the hole, median ~8 µm across). `--close-tunnels [RADIUS]` (default 1 voxel; implies
`--fill-holes`) plugs them before skeletonization:

1. Fill enclosed cavities.
2. A binary closing with a ball of RADIUS voxels proposes candidate patches; each
   26-connected group of added voxels is one patch.
3. A patch is kept only if it raises the mask's 26-connected Euler characteristic
   (components − tunnels + cavities), i.e. it removes a tunnel. Patches that would join
   separate vessel pieces lower it (they add a tunnel or merge components) and are
   rejected. Patches that only smooth the surface leave it unchanged and are also
   rejected, so the mask is otherwise untouched. The Euler change is computed exactly on
   a crop around each patch, against the progressively repaired mask.
4. Fill cavities again, since a plug can enclose one.

Results go to `<sample>_filled_closed` with `analysis_variant` `filled+closed`; the
repaired mask is `analysis_mask.npz`. `statistics.json` → `preprocessing` records candidate,
accepted, bridging-rejected and neutral patches and voxels added.

On these data (radius 1), 10–49 patches per ROI were accepted, adding 40–330 voxels
(≤0.15% of foreground). Small loops fell from 410 to 12 in total while loops ≥50 µm changed
by at most one per ROI. Graph cycle rank matched mask tunnel count in all 15 ROIs.
Branch counts fell 20–45% and total length 1–5%. Radius 2 removed slightly more and added
more voxels. Caveats: this assumes no real vessel loop encloses a hole only one or two
voxels wide; a real loop that narrow cannot be distinguished from a defect at this
resolution. Inspect the remaining small loops and any accepted plugs before relying on
loop counts.

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
- `bifurcations.csv`: one row per three-branch node with parent and child radii, Murray ratio and exponent, child asymmetry, branching angle, and boundary/short/resolved flags.
- `loops.csv`: one row per minimum-basis loop: kind (`cycle`, `parallel`, `self_loop`),
  perimeter, extent, small flag, member branch ids, centroid (ZYX voxels), minimum radius
  and crop/short flags.
- `statistics.json`: distributions, network metrics, mask measurements, QC messages,
  preprocessing provenance and library versions. Undefined values are JSON null or
  empty CSV cells, not zero.
- `statistics.png`: histograms for branch length, mean radius, tortuosity, node degree, minimum-basis loop length and resolved Murray ratio.
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

**Loopiness.** `cycle_density_per_mm3` is cycle rank per ROI volume. `meshedness` is
cycle rank divided by `2N - 5`, where N is the number of degree ≥3 nodes; that is the
maximum for a planar graph, so in 3D use it only to compare ROIs. `looped_length_fraction`
is the share of branch length that is not a bridge, i.e. that lies on at least one cycle.
`loop_length_um` gives perimeters of an exact minimum-weight cycle basis (Horton's
algorithm, solved per biconnected block; parallel branches and self-loops included).
Basis size equals the cycle rank. networkx 3.2's `minimum_cycle_basis` returned
non-minimal bases on these graphs, so it is not used. Loops shorter than 50 µm
(`small_loops`) are often small tunnels through the mask, mostly parallel branch pairs
and self-loops. In these data they are 30–65% of loops even after hole filling.
`large_cycle_density_per_mm3` and `large_loop_length_um` exclude them; prefer these for
comparisons. Loops cut by the crop are not closed and are not counted.

**Hierarchy.** `bifurcations.csv` describes each node with exactly three distinct,
non-loop branches. The widest branch by mean EDT radius is called the parent. This is
a geometric convention, not a flow direction. `murray_ratio` is
parent³ / (child₁³ + child₂³); 1 matches Murray's cube law and 0.5 means all three
radii are equal. `murray_exponent` solves parent^x = child₁^x + child₂^x when it has a
solution. `child_asymmetry` is the smaller/larger child radius, and `branching_angle_deg`
is the angle between the two children's directions over their first 10 µm. A bifurcation
is `resolved` when no branch touches the crop or is flagged short, and the smallest radius
is at least two voxels (3.6 µm here). Murray and asymmetry summaries use resolved
bifurcations only. Angle summaries exclude boundary and short branches.
`radius_length_spearman` is the rank correlation of branch radius and length for interior
branches. Radii are only about two voxels, so EDT discretization dominates radius ratios.
Treat Murray values as descriptive, not as tests of the cube law. Strahler or diameter-defined
orders are not computed: they need a tree or inlet/outlet assignment, which loopy cropped
networks lack.

**Tissue supply.** `tissue_distance_um` is the distribution of distance from every background
voxel to the nearest foreground voxel. A vessel just outside the crop could be closer,
so each value is an upper bound. `tissue_distance_exact_fraction` is the share of voxels
whose distance is no larger than their distance past the ROI face, so their value is exact.
This is about half of voxels in these ROIs, which mostly affects the upper percentiles and
the maximum. Filled-variant cavities count as vessel.

**Knotting and entanglement.** Not implemented yet. A cycle is not evidence of a knot. These statistics do not compute knot
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
