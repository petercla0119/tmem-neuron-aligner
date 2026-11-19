"""Descriptive spatial measurements from existing quantitative images and masks."""

import numpy as np
import pandas as pd


def relative_cv(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / x.mean()) if len(x) >= 2 and x.mean() > 0 else np.nan


def spatial_tiles(green, red, valid, labels, cells, pixel_um, scale):
    if not green.shape == red.shape == valid.shape == labels.shape:
        raise ValueError("Array shapes differ")
    if np.any(labels[~valid]):
        raise ValueError("Masks extend into invalid area")
    h, w = labels.shape
    ys, xs = np.linspace(0, h, 4, dtype=int), np.linspace(0, w, 4, dtype=int)
    area = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=green.ravel(), minlength=len(area))
    ids = cells.label.to_numpy(dtype=int)
    if set(ids) != set(np.flatnonzero(area[1:]) + 1) or len(set(ids)) != len(ids):
        raise ValueError("Nuclear table and labels disagree")
    means = sums[ids] / area[ids]
    cy = cells.centroid_y_px.to_numpy(dtype=float) / scale
    cx = cells.centroid_x_px.to_numpy(dtype=float) / scale
    if np.any((cy < 0) | (cy >= h) | (cx < 0) | (cx >= w)):
        raise ValueError("Nuclear centroid outside field")
    rows = []
    for iy in range(3):
        for ix in range(3):
            y0, y1, x0, x1 = ys[iy], ys[iy + 1], xs[ix], xs[ix + 1]
            keep = (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
            tile_valid = valid[y0:y1, x0:x1]
            values = red[y0:y1, x0:x1][tile_valid]
            n = int(keep.sum())
            pixels = int(tile_valid.sum())
            mm2 = pixels * (pixel_um * scale) ** 2 / 1e6
            rows.append(
                {
                    "tile": iy * 3 + ix,
                    "tile_row": iy,
                    "tile_column": ix,
                    "y0": int(y0),
                    "y1": int(y1),
                    "x0": int(x0),
                    "x1": int(x1),
                    "valid_pixels": pixels,
                    "valid_fraction": pixels / tile_valid.size,
                    "valid_area_mm2": mm2,
                    "nuclei_count": n,
                    "nuclei_per_mm2": n / mm2 if mm2 else np.nan,
                    "mcherry_mean_raw": float(values.mean(dtype=np.float64)) if pixels else np.nan,
                    "mng_mean_per_nucleus_raw": float(means[keep].mean()) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)
