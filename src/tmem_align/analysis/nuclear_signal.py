"""Quantify mNG in saved nuclear masks without normalizing quantitative pixels."""

import numpy as np


def quantify_nuclear_mng(green, labels, valid, scale):
    if green.shape != labels.shape or green.shape != valid.shape:
        raise ValueError("Green, labels and valid mask must have matching shapes")
    if np.any(labels[~valid]):
        raise ValueError("Nuclear labels extend into invalid pixels")
    if not np.isfinite(green).all():
        raise ValueError("Non-finite quantitative green pixels")
    areas = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=green.ravel(), minlength=len(areas))
    present = np.flatnonzero(areas[1:]) + 1
    means = sums[present] / areas[present]
    count = len(present)
    total = float(sums[present].sum())
    pixels = int(areas[present].sum())
    return {
        "mng_nuclei_count_check": count,
        "mng_mean_per_nucleus_raw": float(means.mean()) if count else np.nan,
        "mng_median_per_nucleus_raw": float(np.median(means)) if count else np.nan,
        "mng_mean_nuclear_pixels_raw": total / pixels if pixels else np.nan,
        "mng_integrated_nuclear_raw": total * scale**2,
        "nuclear_pixels_original": pixels * scale**2,
    }
