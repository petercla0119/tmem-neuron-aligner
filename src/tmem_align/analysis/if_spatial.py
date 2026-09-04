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

# IC / corrected-domain policy (validated 2026-09-02, scripts/validate_ic_flatfield.py):
#   - The RELATIVE puncta path (analysis/mcherry_metrics.py: percentile bg + DoG + robust-MAD)
#     is invariant to the IC transform to ~1st order (flat-only Δlocal-contrast <0.001) — nothing to re-derive.
#   - BUT the coloc path (analysis/if_coloc.py) uses ABSOLUTE-DN thresholds
#     (tmem 1763.5, lamp1 3765) calibrated on RAW. Those DO depend on the DN scale: run coloc detection on
#     RAW (don't pass ic_fields to run_if_coloc), or re-run the surveys to re-derive them on the corrected domain.
#   - These fixed display LUTs are calibrated on RAW DN. They stay valid for the FLAT-ONLY corrected domain
#     (corrected=raw/flat, flat mean≈1 → DN scale preserved). If flat+DARK is ever adopted instead,
#     every `lo` must drop by that channel's darkfield (488≈96, 561≈91, 640≈100, 405≈98 DN) or it clips to black.
#   - The MAP2 cell-body model (segment_cell_bodies) was trained on RAW-DN MAP2 through this LUT.
#     Feed it RAW MAP2 only — do NOT route IC-corrected 488 into it without a retrain.


def apply_display_lut(img_yx: np.ndarray, channel: str) -> np.ndarray:
    """Scale a raw channel to [0, 1] float for display using the fixed per-channel LUT.

    `channel` is a CH_* wavelength key. Falls back to per-image p1/p99.5 for any
    channel without a fixed LUT, so the helper is always total.
    """
    img = np.asarray(img_yx, dtype=np.float32)
    lo, hi = DISPLAY_LUT.get(channel, tuple(np.percentile(img, [1, 99.5])))
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


# IC field type: {channel_name: 2D_field} or {channel_name: (2D_field, darkfield_scalar)}.
# Load from a pre-computed .npz via load_ic_fields().
ICFields = dict[str, "np.ndarray | tuple[np.ndarray, float]"]


def load_ic_fields(npz_path: str | Path) -> ICFields:
    """Load per-channel IC fields from a .npz produced by calculate_ic_fields_by_channel.

    Keys are channel names (e.g. '488nm'); values are (field_2d, darkfield_scalar)
    when a matching '<key>_darkfield' entry exists, else just field_2d.
    """
    data = np.load(npz_path)
    result: ICFields = {}
    for key in data.files:
        if key.endswith("_darkfield"):
            continue
        dark_key = f"{key}_darkfield"
        result[key] = (data[key], float(data[dark_key])) if dark_key in data.files else data[key]
    return result


def _apply_ic_zcyx(
    arr: np.ndarray,
    channel_names: list[str],
    ic_fields: ICFields,
) -> np.ndarray:
    """Dark-subtract and flat-divide a ZCYX uint16 array in-place on a copy.

    IC order: (raw - darkfield) / flatfield, clamped to [0, 65535] uint16.
    Only channels present in ic_fields are corrected; others pass through unchanged.
    """
    out = arr.astype(np.float32)
    for ci, ch in enumerate(channel_names):
        if ch not in ic_fields:
            continue
        entry = ic_fields[ch]
        flat, dark = entry if isinstance(entry, tuple) else (entry, 0.0)
        # IC_FIELD_FLOOR=0.1: same guard as preprocess._apply_ic_field_float
        flat_c = np.clip(np.asarray(flat, dtype=np.float32), 0.1, None)
        out[:, ci] = np.clip(out[:, ci] - dark, 0.0, None) / flat_c[np.newaxis]
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def load_fov(
    nd2_path: str | Path,
    ic_fields: ICFields | None = None,
) -> dict[str, np.ndarray]:
    """Load one ND2 FOV; optionally IC-correct; max-project Z; return {channel_name: yx_uint16}.

    IC is applied to the full ZCYX stack before max-projection (dark → flat → max-project),
    so the MIP reflects corrected intensities. Pass ic_fields=load_ic_fields(npz) to enable;
    None (default) is a no-op for backward compatibility.

    Channel names come from ND2 metadata so files with a channel-order swap are handled
    correctly without special-casing.
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
        if ic_fields:
            arr = _apply_ic_zcyx(arr, channel_names, ic_fields)
        arr = arr.max(axis=0)  # ZCYX → CYX (after IC)
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected ND2 shape after load: {arr.shape}")

    if arr.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: metadata={len(channel_names)}, array C={arr.shape[0]}"
        )

    return {name: arr[i] for i, name in enumerate(channel_names)}


def load_fov_3d(
    nd2_path: str | Path,
    ic_fields: ICFields | None = None,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    """Load one ND2 FOV WITHOUT max-projection: {channel_name: zyx_uint16}, (z,y,x) µm.

    The 3D counterpart of load_fov, for coloc/detection that needs the z-stack.
    IC is applied to the ZCYX stack before channel splitting when ic_fields is provided.
    Voxel size (µm) comes from ND2 metadata per FOV (this dataset uses adaptive Z,
    so z-spacing is not constant across files). Single-Z FOVs come back with a
    length-1 z axis.
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
        vx = f.voxel_size()  # VoxelSize(x, y, z) in µm

    if arr.ndim == 3:  # CYX — add singleton Z
        arr = arr[np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected ND2 shape after load: {arr.shape}")

    if ic_fields:
        arr = _apply_ic_zcyx(arr, channel_names, ic_fields)  # ZCYX in, ZCYX out

    arr = np.moveaxis(arr, 1, 0)  # ZCYX → CZYX
    if arr.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: metadata={len(channel_names)}, array C={arr.shape[0]}"
        )
    channels = {name: arr[i] for i, name in enumerate(channel_names)}
    return channels, (float(vx.z), float(vx.y), float(vx.x))


def segment_nuclei(
    dapi_yx: np.ndarray,
    diameter: float | None = None,
    model_name: str = "cpsam",
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 1.0,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Return integer label array (0 = background) from Cellpose-SAM nuclei model.

    Defaults to cpsam (SAM2 backbone, best accuracy) with gpu=True (uses MPS on
    Apple Silicon). Falls back gracefully if MPS is unavailable.
    Requires cellpose: pip install cellpose

    Endorsed defaults (tuned on d7/d14/d28 cleaved-TMEM FOVs, 2026-09-03):
      diameter=None, flow_threshold=0.4, cellprob_threshold=+1.0

    diameter: None = auto (Cellpose estimates from the image). A fixed value (e.g.
    111 px) rescales all objects to that size before the network runs; blebs and
    pyknotic nuclei smaller than the fixed prior are shrunk below the detection floor
    and silently missed. Use None for max recall; tune cellprob_threshold for precision.

    flow_threshold: maximum allowed flow-field error for a mask to survive. HIGHER =
    more permissive = more detections (Cellpose convention, opposite of a p-value).
    Default 0.4 (Cellpose default). Do NOT raise this when diameter=None — at 0.8
    with auto-diameter it floods the background with garbage-flow masks (804 vs 54
    objects on a control FOV). Use cellprob_threshold instead to adjust recall.

    cellprob_threshold: detection-probability floor. +1.0 cuts background debris
    while preserving real dim/dying nuclei (which are flagged downstream by
    nuclear_health_stats). Note: object count is non-monotonic — sweeping lower does
    not guarantee more detections. Tune on a QC well; see nuclear_health_qc.ipynb.

    Pass cache_path (a .npy file) to skip Cellpose on subsequent calls — loads
    from disk if the file exists, otherwise runs Cellpose and saves the result.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            return np.load(cache).astype(np.int32)

    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError("Install cellpose: pip install cellpose") from exc

    model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
    img = np.asarray(dapi_yx, dtype=np.float32)
    masks, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    result = masks.astype(np.int32)

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, result)
    return result


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


# HITL fine-tuned cpsam cell-body model (trained 2026-08-27, 9 corrected FOVs).
# Not committed — 1.2 GB, lives beside the image data. See
# 03_contexts/2026_tmem/Fixed-IF Spatial Analysis.md for training/eval provenance.
_CELLBODY_MODEL = (
    "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821"
    "/hitl_map2_train/models/map2_cellbody_cpsam"
)


def segment_cell_bodies(
    map2_yx: np.ndarray,
    model_path: str | Path = _CELLBODY_MODEL,
    gpu: bool = True,
    cache_path: str | Path | None = None,
) -> np.ndarray:
    """Segment MAP2 cell bodies with the HITL fine-tuned cpsam model.

    Primary cell-body method (replaces expand_to_cell_bodies, kept as fallback).
    The model learned soma-vs-neurite, so outlines stop at the cell body instead
    of following bright neurites the way seeded expansion does.

    INVARIANT: feed the same fixed-LUT uint8 the model trained on (apply_display_lut
    on the raw MAP2 max-projection), NOT raw 16-bit — Cellpose's per-image
    normalize99 on raw DN would reintroduce the brightness drift the LUT removes.

    Known ceiling: ~15-20% under-detection (faint + saturated somata; the fixed LUT
    likely clips the brightest to flat white, removing texture the model keys on).
    Fine for detection/area-grade features; if the per-cell spatial features later
    prove boundary-sensitive, the fix is ~6-9 more corrected FOVs + retrain (the
    map2_cellbody_cpsam model retrains in ~5 min).

    Pass cache_path (a .npy file) to skip Cellpose on subsequent calls — loads
    from disk if the file exists, otherwise runs Cellpose and saves the result.

    Returns an int32 label array (0 = background).
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            return np.load(cache).astype(np.int32)

    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError("Install cellpose: pip install cellpose") from exc

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Fine-tuned cell-body model not found: {model_path}. "
            "It is not committed (1.2 GB); retrain via notebooks/ or use "
            "expand_to_cell_bodies() as the fallback."
        )

    model = models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))
    img_u8 = (apply_display_lut(map2_yx, CH_MAP2) * 255).astype(np.uint8)
    masks, _, _ = model.eval(img_u8)
    result = masks.astype(np.int32)

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, result)
    return result
