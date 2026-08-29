"""Fixed-IF spatial analysis for the cleaved TMEM106B / LAMP1 / MAP2 / DAPI dataset.

Single-timepoint 4-channel z-stack → max-project → segment nuclei (Cellpose) →
per-cell feature extraction. Channel lookup is by wavelength name from ND2
metadata, never by index — handles the known D20_F1 channel-order swap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Channel name constants matching ND2 metadata wavelength labels for this dataset
CH_MAP2 = "488nm"
CH_LAMP1 = "640nm"
CH_TMEM = "561nm"
CH_DAPI = "405nm"

# Nucleus diameter estimate: ~12 µm / 0.108 µm per px ≈ 111 px
# ponytail: fixed estimate; pass diameter=None to let Cellpose auto-detect (slower)
_NUCLEUS_DIAMETER_PX = 111

# Fixed per-channel display LUTs (raw uint16 DN) for the d7 fixed-IF dataset.
# lo = p1 (camera/background floor ~105 DN), hi = p99.9, pooled over 18 FOVs across
# the plate (see reports/if_segmentation_pilot/lut_ranges.md). Use these instead of
# per-image percentile stretch so brightness is COMPARABLE across conditions and
# doesn't drift image-to-image. Note TMEM/561 is very dim/sparse (hi≈1800) — that's
# the biology, not a display error.
DISPLAY_LUT = {
    CH_MAP2: (110, 21000),
    CH_LAMP1: (120, 12000),
    CH_TMEM: (105, 1800),
    CH_DAPI: (130, 21000),
}


def apply_display_lut(img_yx: np.ndarray, channel: str) -> np.ndarray:
    """Scale a raw channel to [0, 1] float for display using the fixed per-channel LUT.

    `channel` is a CH_* wavelength key. Falls back to per-image p1/p99.5 for any
    channel without a fixed LUT, so the helper is always total.
    """
    img = np.asarray(img_yx, dtype=np.float32)
    lo, hi = DISPLAY_LUT.get(channel, tuple(np.percentile(img, [1, 99.5])))
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def load_fov(nd2_path: str | Path) -> dict[str, np.ndarray]:
    """Load one ND2 FOV; max-project Z; return {channel_name: yx_uint16_array}.

    Channel names come from ND2 metadata (e.g. '488nm', '561nm') so files with
    a channel-order swap are handled correctly without special-casing.
    """
    try:
        import nd2
    except ImportError as exc:
        raise ImportError("Install nd2 support: pip install -e '.[nd2]'") from exc

    with nd2.ND2File(Path(nd2_path)) as f:
        channel_names = [
            str(getattr(getattr(ch, "channel", ch), "name", None) or f"ch{i}")
            for i, ch in enumerate(f.metadata.channels)
        ]
        arr = f.asarray()  # ZCYX for z-stacks, CYX for single-Z

    if arr.ndim == 4:
        arr = arr.max(axis=0)  # ZCYX → CYX
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected ND2 shape after load: {arr.shape}")

    if arr.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: metadata={len(channel_names)}, array C={arr.shape[0]}"
        )

    return {name: arr[i] for i, name in enumerate(channel_names)}


def segment_nuclei(
    dapi_yx: np.ndarray,
    diameter: float | None = _NUCLEUS_DIAMETER_PX,
    model_name: str = "cpsam",
    gpu: bool = True,
) -> np.ndarray:
    """Return integer label array (0 = background) from Cellpose-SAM nuclei model.

    Defaults to cpsam (SAM2 backbone, best accuracy) with gpu=True (uses MPS on
    Apple Silicon). Falls back gracefully if MPS is unavailable.
    Requires cellpose: pip install cellpose
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError("Install cellpose: pip install cellpose") from exc

    model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
    img = np.asarray(dapi_yx, dtype=np.float32)
    masks, _, _ = model.eval(img, diameter=diameter, channels=[0, 0])
    return masks.astype(np.int32)


# Fine-tuned MAP2 cell-body model (cpsam, HITL-trained 2026-08-27 on 9 corrected
# FOVs). Lives with the data, not in git. AP@0.5 0.58 vs expansion 0.54.
_CELLBODY_MODEL = (
    "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821"
    "/hitl_map2_train/models/map2_cellbody_cpsam"
)


def segment_cell_bodies(
    map2_yx: np.ndarray,
    model_path: str = _CELLBODY_MODEL,
    gpu: bool = True,
) -> np.ndarray:
    """Segment MAP2 cell bodies with the HITL fine-tuned cpsam model.

    Feeds the same fixed-LUT uint8 the model was trained on (apply_display_lut).
    Returns int32 label array (0 = background).
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError("Install cellpose: pip install cellpose") from exc

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Cell-body model not found at {model_path}. It lives with the data, not git."
        )

    model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
    img_u8 = (apply_display_lut(map2_yx, CH_MAP2) * 255).astype(np.uint8)
    masks, _, _ = model.eval(img_u8)
    return masks.astype(np.int32)


# ~12 µm max soma radius / 0.108 µm per px ≈ 110 px. Sweep (reports/if_segmentation
# _pilot/map2_expansion_sweep.png) showed 90/110/140 near-identical — mesh connectivity,
# not this cap, bounds the bodies. ponytail: 110 keeps bodies soma-focused.
_MAX_EXPANSION_PX = 110

# Fill interior holes up to ~soma area (punctate MAP2 leaves sub-threshold gaps
# inside somata). ~14 µm soma → ~130 px → ~13000 px² area.
_SOMA_HOLE_PX = 13000


def cell_foreground_mask(
    map2_yx: np.ndarray,
    close_radius: int = 3,
) -> np.ndarray:
    """Binary MAP2 foreground: where any neuron signal exists (Otsu + morphology).

    Growth in expand_to_cell_bodies is clipped to this mask so labels never bleed
    into background. close_radius fills pinholes; set 0 to skip.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, disk, remove_small_holes

    img = np.asarray(map2_yx, dtype=np.float32)
    fg = img > threshold_otsu(img)
    if close_radius > 0:
        fg = closing(fg, disk(close_radius))
    # Fill sub-threshold gaps inside somata so cell bodies are solid, not speckled.
    fg = remove_small_holes(fg, max_size=_SOMA_HOLE_PX)
    return fg


def expand_to_cell_bodies(
    nuclei_masks: np.ndarray,
    map2_yx: np.ndarray,
    max_distance: int = _MAX_EXPANSION_PX,
    close_radius: int = 3,
) -> np.ndarray:
    """Grow each nucleus label outward into MAP2 foreground → cell-body labels.

    One cell body per nucleus. Labels are preserved (cell N's body keeps label N),
    grown by watershed on the distance transform so touching cells split at the
    midline. Growth is clipped to the MAP2 foreground mask and to max_distance px
    from the nucleus, so bodies stay soma-sized and never fill background.

    Returns an int32 label array aligned to nuclei_masks.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    seeds = np.asarray(nuclei_masks, dtype=np.int32)
    fg = cell_foreground_mask(map2_yx, close_radius=close_radius)
    # A nucleus can sit just outside thresholded MAP2 (soma dimmer than neurites);
    # union so every seed is inside the region it's allowed to grow into.
    fg = fg | (seeds > 0)

    # Watershed splits touching cells at the midline; -distance so basins sit on
    # foreground and flood outward from the seeds. Restricted to fg by mask.
    distance = ndi.distance_transform_edt(fg)
    grown = watershed(-distance, markers=seeds, mask=fg)

    # Cap expansion: zero any pixel more than max_distance px from any nucleus,
    # so bodies stay soma-sized instead of flooding down a neurite.
    dist_from_seed = ndi.distance_transform_edt(seeds == 0)
    grown[dist_from_seed > max_distance] = 0
    return grown.astype(np.int32)
