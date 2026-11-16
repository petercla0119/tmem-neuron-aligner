"""Count segmented nuclei and measure longitudinal fluorescence in camera units."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops


def prepare_channels(raw: np.ndarray, scale: int):
    """Block average CYX channels; trim remainders and exclude zero-padded blocks.

    No percentile normalization is applied to quantitative red data. Cellpose
    normalizes its separate green input. The caller records the trimmed extent.
    """
    if raw.ndim != 3 or raw.shape[0] != 2 or scale < 1 or int(scale) != scale:
        raise ValueError("Expected two CYX channels and positive integer scale")
    h, w = raw.shape[1] // scale, raw.shape[2] // scale
    if min(h, w) == 0:
        raise ValueError("Image smaller than scale")
    blocks = raw[:, : h * scale, : w * scale].reshape(2, h, scale, w, scale)
    averaged = blocks.mean(axis=(2, 4), dtype=np.float64).astype(np.float32)
    valid = (blocks > 0).all(axis=(0, 2, 4))
    return averaged[0], averaged[1], valid


def quantify_frame(red, labels, valid, pixel_um, scale, background):
    """Measurements at segmentation resolution with exact block-area scaling.

    Nuclear fluorescence uses block-average pixels at mask edges; integrated
    field measurements are equivalent to summing original pixels of valid blocks.
    """
    if red.shape != labels.shape or red.shape != valid.shape:
        raise ValueError("Image, labels and valid mask must have the same shape")
    if not valid.any():
        raise ValueError("No valid pixels")
    if np.any(labels[~valid]):
        raise ValueError("Labels extend into invalid pixels")
    vals = red[valid].astype(np.float64)
    px_area = scale**2
    area_mm2 = vals.size * (pixel_um * scale) ** 2 / 1e6
    # Boundary diagnostic includes the perimeter of the usable field.
    boundary = valid & ~ndi.binary_erosion(valid)
    edge_labels = set(np.unique(labels[boundary])) - {0}
    cells = []
    for p in regionprops(labels, intensity_image=red):
        cells.append(
            {
                "label": int(p.label),
                "centroid_y_px": float(p.centroid[0] * scale + (scale - 1) / 2),
                "centroid_x_px": float(p.centroid[1] * scale + (scale - 1) / 2),
                "area_um2": float(p.area * (pixel_um * scale) ** 2),
                "nuclear_mcherry_mean_raw": float(p.intensity_mean),
                "nuclear_mcherry_mean_bg_corrected": float(p.intensity_mean - background),
                "touches_field_boundary": p.label in edge_labels,
            }
        )
    n = len(cells)
    raw_total = float(vals.sum() * px_area)
    corrected_total = float((vals.sum() - background * vals.size) * px_area)
    metrics = {
        "nuclei_count": n,
        "nuclei_touching_boundary": len(edge_labels),
        "nuclei_interior": n - len(edge_labels),
        "area_mm2": float(area_mm2),
        "nuclei_per_mm2": n / area_mm2,
        "valid_pixels_original": int(vals.size * px_area),
        "mcherry_background": float(background),
        "mcherry_mean_raw": float(vals.mean()),
        "mcherry_integrated_raw": raw_total,
        "mcherry_mean_bg_corrected": float(vals.mean() - background),
        "mcherry_integrated_bg_corrected": corrected_total,
        "mcherry_integrated_per_nucleus": corrected_total / n if n else np.nan,
        "median_nuclear_area_um2": float(np.median([c["area_um2"] for c in cells]))
        if n
        else np.nan,
    }
    return metrics, cells


def add_baseline_changes(frame: pd.DataFrame):
    """First observed day per well is baseline; nonpositive denominators yield NaN."""
    frame = frame.sort_values(["well", "day"]).copy()
    metrics = [
        "nuclei_count",
        "mcherry_mean_raw",
        "mcherry_mean_bg_corrected",
        "mcherry_integrated_per_nucleus",
    ]
    for _, group in frame.groupby("well"):
        baseline = group.iloc[0]
        frame.loc[group.index, "baseline_day"] = baseline.day
        for key in metrics:
            frame.loc[group.index, key + "_change_baseline"] = group[key] - baseline[key]
            frame.loc[group.index, key + "_fold_baseline"] = (
                group[key] / baseline[key] if baseline[key] > 0 else np.nan
            )
    return frame


def segment_nuclei(model, green, parameters):
    """Protect camera-unit preprocessing data from Cellpose's in-place normalization."""
    masks, _, _ = model.eval(green.copy(), **parameters)
    return masks.astype(np.uint32)
