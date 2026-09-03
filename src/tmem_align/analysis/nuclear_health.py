"""Nuclear health QC for fixed-IF DAPI channel.

Per-nucleus intensity statistics + soft health flag from a Cellpose label mask
and a raw DAPI max-projection. Intended as a pre-filter step in the fixed-IF
pipeline: segment_nuclei() → nuclear_health_stats() → filter is_healthy → downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def nuclear_health_stats(
    nuclei_masks: np.ndarray,
    dapi_yx: np.ndarray,
    *,
    min_mean_intensity: float = 300.0,
    max_skewness: float = 2.0,
    min_nucleus_area: int = 50,
    necrotic_area_max: int = 200,
) -> pd.DataFrame:
    """Per-nucleus DAPI intensity stats and soft health flag.

    nuclei_masks: int32 label array from segment_nuclei() — 0 = background.
    dapi_yx: raw uint16 DAPI max-projection, same spatial shape as nuclei_masks.
    Nuclei with area_px < min_nucleus_area are dropped (true segmentation debris).
    Nuclei with min_nucleus_area <= area_px < necrotic_area_max are classified as
    "skewed" (pyknotic/necrotic — shrunken chromatin), not dropped.
    Returns one row per nucleus; health_flag ∈ {"healthy", "low_signal", "skewed"}.
    Thresholds are placeholders — tune empirically on a QC pilot well first.
    """
    from scipy.stats import skew
    from skimage.measure import regionprops

    masks = np.asarray(nuclei_masks, dtype=np.int32)
    img = np.asarray(dapi_yx, dtype=np.float32)
    if masks.shape != img.shape:
        raise ValueError(
            f"nuclei_masks shape {masks.shape} does not match dapi_yx shape {img.shape}"
        )

    rows: list[dict[str, Any]] = []
    for prop in regionprops(masks, intensity_image=img):
        if prop.area < min_nucleus_area:
            continue
        values = prop.image_intensity[prop.image]
        mean_i = float(values.mean())
        skew_val = float(skew(values))
        rows.append(
            {
                "nucleus_label": int(prop.label),
                "area_px": int(prop.area),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "mean_intensity": mean_i,
                "median_intensity": float(np.median(values)),
                "std_intensity": float(values.std()),
                "skewness": skew_val,
                "health_flag": _classify_nucleus(
                    mean_i, skew_val, prop.area, min_mean_intensity, max_skewness, necrotic_area_max
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["is_healthy"] = df["health_flag"] == "healthy"
    else:
        df["is_healthy"] = pd.Series(dtype=bool)
    return df


def _classify_nucleus(
    mean_intensity: float,
    skewness: float,
    area: int,
    min_mean_intensity: float,
    max_skewness: float,
    necrotic_area_max: int,
) -> str:
    # Small nuclei first: pyknosis (chromatin condensation) shrinks the nucleus.
    # Catching by size is more robust than skewness alone since condensed chromatin
    # can appear either uniform-bright or splotchy depending on the stage.
    if area < necrotic_area_max:
        return "skewed"
    # low_signal checked before skewness: dim nucleus is most actionable (dead/dying cell)
    if mean_intensity < min_mean_intensity:
        return "low_signal"
    if abs(skewness) > max_skewness:
        return "skewed"
    return "healthy"


def plot_nuclear_health(
    stats: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
) -> Path:
    """Violin + jitter strip of per-nucleus mean_intensity, colored by health_flag.

    Writes a PNG to output_path and returns the path.
    stats is the DataFrame from nuclear_health_stats().
    """
    import matplotlib.pyplot as plt

    # CVD-safe categorical palette (from dataviz skill defaults)
    _COLORS = {
        "healthy": "#888888",
        "low_signal": "#4477AA",
        "skewed": "#EE6677",
    }
    _ORDER = ["healthy", "low_signal", "skewed"]

    fig, ax = plt.subplots(figsize=(6, 4))

    present = [f for f in _ORDER if f in stats["health_flag"].values]
    positions = list(range(1, len(present) + 1))

    violin_data = [stats.loc[stats["health_flag"] == f, "mean_intensity"].values for f in present]
    if any(len(d) > 0 for d in violin_data):
        parts = ax.violinplot(
            [d for d in violin_data if len(d) > 0],
            positions=[p for p, d in zip(positions, violin_data) if len(d) > 0],
            showmedians=True,
            widths=0.6,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#dddddd")
            pc.set_alpha(0.6)
        for part in ("cbars", "cmins", "cmaxes", "cmedians"):
            if part in parts:
                parts[part].set_color("#555555")

    # jitter strip colored by flag
    rng = np.random.default_rng(0)
    for pos, flag in zip(positions, present):
        vals = stats.loc[stats["health_flag"] == flag, "mean_intensity"].values
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=_COLORS[flag],
            s=12,
            alpha=0.7,
            linewidths=0,
            label=flag,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(present)
    ax.set_ylabel("Mean DAPI intensity (raw DN)")
    ax.set_xlabel("Health flag")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
