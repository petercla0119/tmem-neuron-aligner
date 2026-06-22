from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    - YX for a single image frame
    - TYX for a single-channel stack
    - TCYX for a multichannel stack
    - TCZYX will be max-projected over Z for this first-pass analysis
    """
    arr = np.asarray(read_image(timeseries_path))
    img = _extract_phenotype(arr, phenotype_channel_index)

    rows = []
    for t in range(img.shape[0]):
        rows.append(
            quantify_puncta_vs_diffuse_frame(
                img[t],
                time_index=t,
                min_size_pixels=min_size_pixels,
                background_percentile=background_percentile,
                threshold_method=threshold_method,
            )
        )
    return pd.DataFrame(rows)


def quantify_puncta_vs_diffuse_roi(
    timeseries_path: str | Path,
    phenotype_channel_index: int | None = None,
    min_size_pixels: int = 6,
    background_percentile: float = 20,
    threshold_method: str = "otsu",
    roi_min_size_pixels: int = 128,
    roi_dilation_pixels: int = 3,
    foreground_percentile: float = 70,
) -> pd.DataFrame:
    """Quantify punctate/diffuse signal inside a conservative foreground ROI mask.

    This is a first-pass cell/neuron-enriched metric. The ROI is derived from the same phenotype
    channel by selecting smoothed foreground signal, removing small objects, and dilating slightly.
    It should be reviewed visually before being treated as a final segmentation strategy.
    """
    arr = np.asarray(read_image(timeseries_path))
    img = _extract_phenotype(arr, phenotype_channel_index)

    rows = []
    for t in range(img.shape[0]):
        frame = img[t].astype(np.float32)
        bg_corrected = background_correct_frame(frame, background_percentile)
        roi_mask = foreground_roi_mask(
            bg_corrected,
            min_size_pixels=roi_min_size_pixels,
            dilation_pixels=roi_dilation_pixels,
            foreground_percentile=foreground_percentile,
        )
        row = quantify_puncta_vs_diffuse_frame(
            frame,
            time_index=t,
            min_size_pixels=min_size_pixels,
            background_percentile=background_percentile,
            threshold_method=threshold_method,
            roi_mask=roi_mask,
        )
        row["roi_area_pixels"] = int(roi_mask.sum())
        row["roi_fraction"] = float(roi_mask.mean())
        row["roi_min_size_pixels"] = int(roi_min_size_pixels)
        row["roi_dilation_pixels"] = int(roi_dilation_pixels)
        row["foreground_percentile"] = float(foreground_percentile)
        rows.append(row)
    return pd.DataFrame(rows)


def quantify_puncta_vs_diffuse_frame(
    frame: np.ndarray,
    *,
    time_index: int = 0,
    min_size_pixels: int = 6,
    background_percentile: float = 20,
    threshold_method: str = "otsu",
    roi_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    frame = frame.astype(np.float32)
    bg_corrected = background_correct_frame(frame, background_percentile)
    analysis_mask = np.ones(bg_corrected.shape, dtype=bool) if roi_mask is None else roi_mask.astype(bool)

    if not analysis_mask.any():
        return _empty_metric_row(time_index)

    smooth = filters.gaussian(bg_corrected, sigma=1.0, preserve_range=True)
    threshold = _threshold(smooth[analysis_mask], threshold_method)
    puncta_mask = (smooth > threshold) & analysis_mask
    puncta_mask = _remove_small_objects(puncta_mask, min_size_pixels)
    labels = measure.label(puncta_mask)
    props = measure.regionprops(labels, intensity_image=bg_corrected)

    punctate_sum = float(bg_corrected[puncta_mask].sum())
    diffuse_mask = analysis_mask & ~puncta_mask
    diffuse_sum = float(bg_corrected[diffuse_mask].sum())
    punctate_mean = float(bg_corrected[puncta_mask].mean()) if puncta_mask.any() else 0.0
    diffuse_mean = float(bg_corrected[diffuse_mask].mean()) if diffuse_mask.any() else 0.0
    rupture_like_score = diffuse_mean / (punctate_mean + 1e-6)

    return {
        "time_index": time_index,
        "puncta_count": int(len(props)),
        "punctate_sum": punctate_sum,
        "diffuse_sum": diffuse_sum,
        "punctate_mean": punctate_mean,
        "diffuse_mean": diffuse_mean,
        "rupture_like_score": rupture_like_score,
        "mean_puncta_area_pixels": float(np.mean([p.area for p in props])) if props else 0.0,
        "max_puncta_intensity": _max_region_intensity(props),
        "analysis_area_pixels": int(analysis_mask.sum()),
        "analysis_area_fraction": float(analysis_mask.mean()),
    }


def background_correct_frame(frame: np.ndarray, background_percentile: float) -> np.ndarray:
    background = np.percentile(frame, background_percentile)
    return np.clip(frame.astype(np.float32) - background, 0, None)


def foreground_roi_mask(
    bg_corrected: np.ndarray,
    *,
    min_size_pixels: int = 128,
    dilation_pixels: int = 3,
    foreground_percentile: float = 70,
) -> np.ndarray:
    positive = bg_corrected[bg_corrected > 0]
    if positive.size == 0:
        return np.zeros(bg_corrected.shape, dtype=bool)

    smooth = filters.gaussian(bg_corrected, sigma=2.0, preserve_range=True)
    threshold = np.percentile(positive, foreground_percentile)
    mask = smooth > threshold
    mask = _remove_small_objects(mask, min_size_pixels)
    if dilation_pixels > 0:
        mask = morphology.dilation(mask, morphology.disk(dilation_pixels))
    return mask.astype(bool)


def _extract_phenotype(arr: np.ndarray, channel_index: int | None) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:  # YX
        return arr[np.newaxis, :, :]
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
    if frame.size == 0:
        return 0.0
    if method == "otsu":
        return float(filters.threshold_otsu(frame))
    if method == "yen":
        return float(filters.threshold_yen(frame))
    if method == "triangle":
        return float(filters.threshold_triangle(frame))
    raise ValueError(f"Unknown threshold method: {method}")


def _remove_small_objects(mask: np.ndarray, min_size_pixels: int) -> np.ndarray:
    try:
        return morphology.remove_small_objects(mask, max_size=min_size_pixels - 1)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=min_size_pixels)


def _max_region_intensity(props: list[Any]) -> float:
    if not props:
        return 0.0
    values = []
    for prop in props:
        if hasattr(prop, "intensity_max"):
            values.append(prop.intensity_max)
        else:
            values.append(prop.max_intensity)
    return float(max(values))


def _empty_metric_row(time_index: int) -> dict[str, Any]:
    return {
        "time_index": time_index,
        "puncta_count": 0,
        "punctate_sum": 0.0,
        "diffuse_sum": 0.0,
        "punctate_mean": 0.0,
        "diffuse_mean": 0.0,
        "rupture_like_score": 0.0,
        "mean_puncta_area_pixels": 0.0,
        "max_puncta_intensity": 0.0,
        "analysis_area_pixels": 0,
        "analysis_area_fraction": 0.0,
    }
