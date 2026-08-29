#!/usr/bin/env python3
"""Recreate the Arm D puncta masks for the mentee-vs-mentor masking comparison.

Companion to explain_arm_d_cyan_foreground.py. That script explains the cyan *foreground*
outline (488 stable channel). This one renders the *puncta* masks — the mCherry (561)
puncta detected inside the 488 foreground, using the same canonical `detect_puncta` the
pipeline uses. Reuses the same 8 cells / input CSVs built by build_arm_d_inputs.py.

Run (after build_arm_d_inputs.py):
  source .venv/bin/activate
  python notebooks/explain_arm_d_puncta.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure, morphology

from tmem_align.analysis.mcherry_metrics import (
    MCherryMetricConfig,
    background_subtract,
    detect_puncta,
)

# reuse the foreground script's paths, loaders, and crop helpers
from explain_arm_d_cyan_foreground import (
    BASE,
    MANIFEST,
    METRICS,
    ROOT,
    SOURCE,
    _channel_index,
    _load_cyx,
    build_foreground_stages,
    center_crop,
    normalize,
)

CFG = MCherryMetricConfig()
OUT = BASE / "arm_d_puncta_masks"
PANELS = OUT / "individual_panels"
SIZE = 160


def overlay_rgb(
    mcherry_crop: np.ndarray,
    *,
    foreground: np.ndarray | None = None,
    puncta: np.ndarray | None = None,
) -> np.ndarray:
    """Grayscale mCherry with optional cyan foreground contour + red puncta fill."""
    gray = normalize(mcherry_crop)
    rgb = np.repeat(gray[..., None], 3, axis=2) * 0.7
    if foreground is not None:
        boundary = morphology.dilation(foreground) ^ morphology.erosion(foreground)
        rgb[boundary] = np.array([0.0, 0.95, 1.0])  # cyan = 488 foreground edge
    if puncta is not None:
        rgb[puncta] = np.array([1.0, 0.15, 0.2])  # red = puncta mask
    return np.clip(rgb, 0, 1)


def write_panel(
    path: Path,
    *,
    example_id: str,
    well: str,
    day: int,
    mcherry: np.ndarray,
    corrected: np.ndarray,
    dog: np.ndarray,
    foreground: np.ndarray,
    puncta: np.ndarray,
    cy: float,
    cx: float,
) -> dict[str, object]:
    """One 'here are the puncta masks' panel; returns per-cell measurements."""
    raw_c, y0, x0 = center_crop(mcherry, cy, cx, SIZE)
    dog_c, _, _ = center_crop(dog, cy, cx, SIZE)
    fg_c, _, _ = center_crop(foreground, cy, cx, SIZE)
    pun_c, _, _ = center_crop(puncta, cy, cx, SIZE)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(normalize(raw_c), cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(
        plt.Circle((SIZE / 2, SIZE / 2), 26, edgecolor="#57FF7A", facecolor="none", linestyle="--")
    )
    axes[0].set_title("Raw mCherry (561) crop\ngreen = focal candidate")
    axes[1].imshow(normalize(dog_c), cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("DoG bandpass (σ=1−σ=3)\nwhat puncta detection sees")
    axes[2].imshow(overlay_rgb(raw_c, puncta=pun_c))
    axes[2].set_title(f"Puncta mask (red)\n{int(pun_c.sum())} px in crop")
    axes[3].imshow(overlay_rgb(raw_c, foreground=fg_c, puncta=pun_c))
    axes[3].set_title("Post-mask overlay\ncyan = 488 foreground | red = puncta")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{example_id}. {well} Day {day} | Arm D puncta mask",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.04,
        "Puncta = mCherry (561) blobs passing the DoG + robust-threshold test INSIDE the 488 foreground. "
        "Detected on the full ND2 frame; the 160 px crop only limits what is shown.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    crop_labels = measure.label(pun_c)
    return {
        "example_id": example_id,
        "well": well,
        "day": day,
        "full_frame_foreground_pixels": int(foreground.sum()),
        "full_frame_puncta_pixels": int(puncta.sum()),
        "full_frame_puncta_count": int(measure.label(puncta).max()),
        "crop_puncta_pixels": int(pun_c.sum()),
        "crop_puncta_count": int(crop_labels.max()),
        "crop_foreground_pixels": int(fg_c.sum()),
        "punctate_fraction_of_foreground": float(puncta.sum() / max(foreground.sum(), 1)),
    }


def write_contact_sheet(paths: list[Path]) -> None:
    fig, axes = plt.subplots(8, 1, figsize=(15, 34), constrained_layout=True)
    for ax, path in zip(axes.flat, paths, strict=True):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.suptitle("Arm D puncta masks (8 neuron-candidate crops)", fontsize=18, fontweight="bold")
    fig.savefig(OUT / "arm_d_puncta_mask_contact_sheet.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    selected = pd.read_csv(METRICS, dtype={"example_id": str})

    rows: list[dict[str, object]] = []
    panel_paths: list[Path] = []
    for _, row in selected.sort_values("example_id").iterrows():
        example_id = str(row["example_id"]).zfill(2)
        well, day = str(row["well"]), int(row["day"])
        src = manifest[(manifest["well"] == well) & (manifest["day"] == day)].iloc[0]

        arr = _load_cyx(Path(str(src["file_path"])))
        chan = str(src["channel_names"])
        mcherry = arr[_channel_index(chan, "561")]
        stable = arr[_channel_index(chan, "488")]

        foreground = np.asarray(build_foreground_stages(stable)["final"])
        corrected = background_subtract(mcherry, percentile=CFG.background_percentile)
        puncta = detect_puncta(corrected, foreground, config=CFG)

        from skimage import filters

        small = filters.gaussian(corrected, sigma=CFG.puncta_sigma_small, preserve_range=True)
        large = filters.gaussian(corrected, sigma=CFG.puncta_sigma_large, preserve_range=True)
        dog = np.clip(small - large, 0, None)

        panel_path = PANELS / f"{example_id}_{well}_day{day}_arm_d_puncta.png"
        rows.append(
            write_panel(
                panel_path,
                example_id=example_id,
                well=well,
                day=day,
                mcherry=mcherry,
                corrected=corrected,
                dog=dog,
                foreground=foreground,
                puncta=puncta,
                cy=float(row["candidate_centroid_y"]),
                cx=float(row["candidate_centroid_x"]),
            )
        )
        panel_paths.append(panel_path)
        print(f"  {example_id} {well} day{day}: {rows[-1]['full_frame_puncta_count']} puncta")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "arm_d_puncta_measurements.csv", index=False)
    write_contact_sheet(panel_paths)
    print(f"\nWrote {len(frame)} Arm D puncta-mask panels to {OUT}")


if __name__ == "__main__":
    main()
