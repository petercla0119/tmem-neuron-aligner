from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .io import read_image


def quantify_puncta_vs_diffuse(
    timeseries_path: str | Path,
    phenotype_channel_index: int | None = None,
    min_size_pixels: int = 6,
    background_percentile: float = 20,
    threshold_method: str = "otsu",
) -> pd.DataFrame:
    """Quantify punctate/diffuse signal over a neuron-centered time stack.

    Expected input shapes:
    - TYX for a single-channel stack
    - TCYX for a multichannel stack
    - TCZYX will be max-projected over Z for this first-pass analysis
    """
    arr = np.asarray(read_image(timeseries_path))
    img = _extract_phenotype(arr, phenotype_channel_index)

    rows = []
    for t in range(img.shape[0]):
        frame = img[t].astype(np.float32)
        background = np.percentile(frame, background_percentile)
        bg_corrected = np.clip(frame - background, 0, None)
        smooth = filters.gaussian(bg_corrected, sigma=1.0, preserve_range=True)
        threshold = _threshold(smooth, threshold_method)
        puncta_mask = smooth > threshold
        puncta_mask = morphology.remove_small_objects(puncta_mask, min_size=min_size_pixels)
        labels = measure.label(puncta_mask)
        props = measure.regionprops(labels, intensity_image=bg_corrected)

        punctate_sum = float(bg_corrected[puncta_mask].sum())
        diffuse_mask = ~puncta_mask
        diffuse_sum = float(bg_corrected[diffuse_mask].sum())
        punctate_mean = float(bg_corrected[puncta_mask].mean()) if puncta_mask.any() else 0.0
        diffuse_mean = float(bg_corrected[diffuse_mask].mean()) if diffuse_mask.any() else 0.0
        rupture_like_score = diffuse_mean / (punctate_mean + 1e-6)

        rows.append(
            {
                "time_index": t,
                "puncta_count": int(len(props)),
                "punctate_sum": punctate_sum,
                "diffuse_sum": diffuse_sum,
                "punctate_mean": punctate_mean,
                "diffuse_mean": diffuse_mean,
                "rupture_like_score": rupture_like_score,
                "mean_puncta_area_pixels": float(np.mean([p.area for p in props])) if props else 0.0,
                "max_puncta_intensity": float(max([p.max_intensity for p in props])) if props else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _extract_phenotype(arr: np.ndarray, channel_index: int | None) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 3:  # TYX
        return arr
    if arr.ndim == 4:  # TCYX
        if channel_index is None:
            channel_index = 0
        return arr[:, channel_index, :, :]
    if arr.ndim == 5:  # TCZYX
        if channel_index is None:
            channel_index = 0
        return arr[:, channel_index].max(axis=1)
    raise ValueError(f"Unsupported timeseries shape: {arr.shape}")


def _threshold(frame: np.ndarray, method: str) -> float:
    if method == "otsu":
        return float(filters.threshold_otsu(frame))
    if method == "yen":
        return float(filters.threshold_yen(frame))
    if method == "triangle":
        return float(filters.threshold_triangle(frame))
    raise ValueError(f"Unknown threshold method: {method}")
