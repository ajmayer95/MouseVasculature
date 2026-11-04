"""Optional, topology-checked mask repairs. Original TIFFs are never modified."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.morphology import ball


def fill_enclosed_holes(mask):
    """Fill background components enclosed by foreground (6-connected background)."""
    return ndi.binary_fill_holes(mask)


def close_tunnels(mask, radius=1):
    """Plug small tunnels through the mask, rejecting patches that would bridge vessels.

    A binary closing with a ball of `radius` voxels proposes candidate patches (each
    26-connected group of added voxels). Each patch is accepted only if it raises the
    mask's 26-connected Euler characteristic. Euler characteristic is components -
    tunnels + cavities, so an accepted patch removes tunnels without merging components.
    Patches that connect separate vessel pieces (which would add a tunnel or merge
    components) lower it and are rejected; patches that only smooth the surface leave
    it unchanged and are also rejected. The Euler change is local, so it is computed
    exactly on a crop around each patch. Patches are tested in order against the
    progressively repaired mask.
    """
    mask = np.asarray(mask, dtype=bool)
    pad = radius + 1
    closed = ndi.binary_closing(np.pad(mask, pad), structure=ball(radius))
    closed = closed[pad:-pad, pad:-pad, pad:-pad]
    candidates, count = ndi.label(closed & ~mask, structure=np.ones((3, 3, 3)))
    repaired = mask.copy()
    accepted = rejected_bridge = rejected_neutral = added = tunnels_removed = 0
    for label, box in enumerate(ndi.find_objects(candidates), start=1):
        box = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n)) for s, n in zip(box, mask.shape))
        before = repaired[box]
        patch = candidates[box] == label
        change = euler_number(before | patch, connectivity=3) - euler_number(before, connectivity=3)
        if change > 0:
            repaired[box] |= patch
            accepted += 1
            added += int(patch.sum())
            tunnels_removed += int(change)
        elif change < 0:
            rejected_bridge += 1
        else:
            rejected_neutral += 1
    report = dict(close_tunnels_radius_voxels=radius, candidate_patches=int(count),
                  accepted_patches=accepted, rejected_bridging_patches=rejected_bridge,
                  rejected_neutral_patches=rejected_neutral, tunnel_voxels_added=added,
                  euler_increase=tunnels_removed)
    return repaired, report
