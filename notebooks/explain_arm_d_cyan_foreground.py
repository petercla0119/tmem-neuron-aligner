#!/usr/bin/env python3
"""Explain where the cyan Arm D foreground outline comes from.

This is intentionally a visualization/debug script, not a new analysis method. It reads the
same eight ND2 frames and uses the same Arm D foreground settings as the comparison panels.

To run:
  1. Set ROOT below to wherever Makenna's run directory landed on your machine.
  2. Ensure MANIFEST and SOURCE paths resolve (ask Makenna for the two CSVs if missing).
  3. source .venv/bin/activate && python notebooks/explain_arm_d_cyan_foreground.py
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from tmem_align.analysis.mcherry_metrics import MCherryMetricConfig, background_subtract

# ── configure to match your local layout ────────────────────────────────────
ROOT = Path("/Users/pmihack/claire/tmem_2026/tmem-neuron-aligner/reports/arm_d_cyan_local")
SOURCE = ROOT / "08_collaborator_report/neuron_level_post_mask_A_vs_D_8_examples"
# ARM_D_VARIANT selects which candidate set to render: "v488_centered" (488 foreground)
# or "v561_puncta_centered" (brightest 561 puncta cluster). Empty = legacy flat layout.
VARIANT = os.environ.get("ARM_D_VARIANT", "")
BASE = SOURCE / VARIANT if VARIANT else SOURCE
METRICS = BASE / "neuron_post_mask_metrics.csv"
OUT = BASE / "arm_d_cyan_foreground_diagnostics"
PANELS = OUT / "individual_panels"
MANIFEST = ROOT / "05_stratified_24_ad_bridge/selection_manifest.csv"
WORKTREE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
# ────────────────────────────────────────────────────────────────────────────


def _load_cyx(path: Path) -> np.ndarray:
    """Load an ND2 file as a (C, Y, X) array, squeezing any singleton T/Z axes."""
    import nd2

    arr = nd2.imread(path)  # shape is (..., C, Y, X) or (Y, X)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim > 3:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return arr


def _channel_index(channel_names_raw: str, target: str) -> int:
    """Return the index of target in the channel_names manifest column."""
    try:
        names = ast.literal_eval(channel_names_raw)
    except Exception:
        names = [n.strip() for n in str(channel_names_raw).split(",")]
    for i, name in enumerate(names):
        if target.lower() in str(name).lower():
            return i
    raise ValueError(f"Channel '{target}' not found in {names}")


CFG = MCherryMetricConfig()


def center_crop(array: np.ndarray, cy: float, cx: float, size: int) -> tuple[np.ndarray, int, int]:
    """Grab a square around the chosen 488-positive candidate.

    The little clamp here is just guardrail stuff: if a candidate sits near an image edge,
    we slide the crop inward instead of returning a smaller, awkwardly shaped image.
    """

    half = size // 2
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half
    y0 = max(0, min(y0, array.shape[-2] - size))
    x0 = max(0, min(x0, array.shape[-1] - size))
    return array[..., y0 : y0 + size, x0 : x0 + size], y0, x0


def normalize(frame: np.ndarray) -> np.ndarray:
    """Simple display normalization; none of this affects the mask math."""

    arr = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99.5])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def build_foreground_stages(stable: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Reproduce the Arm D stable-channel foreground mask one step at a time."""

    # First, Arm D removes a broad p20 background estimate from the 488 image.
    # This is why dim haze mostly disappears before the foreground decision happens.
    corrected = background_subtract(stable, percentile=CFG.background_percentile)
    positive = corrected[corrected > 0]
    if positive.size == 0:
        raise ValueError("488 frame has no positive pixels after p20 subtraction")

    # The sigma=2 blur is modest, but it matters: nearby bright pixels become a smooth occupied
    # region instead of a salt-and-pepper mask. That makes the cyan outline look cell-sized.
    smooth = filters.gaussian(corrected, sigma=2.0, preserve_range=True)

    # Slightly quirky but faithful detail: p65 is taken from positive *corrected* pixels,
    # then that value is applied to the smoothed image.
    threshold = float(np.percentile(positive, CFG.foreground_percentile))
    initial = smooth > threshold

    # Tiny islands are not useful foreground, so components smaller than 128 px are dropped.
    cleaned = morphology.remove_small_objects(initial, max_size=CFG.foreground_min_area - 1)

    # Filling holes makes a bright ring count as one occupied region instead of a cyan donut.
    filled = ndi.binary_fill_holes(cleaned)

    # This two-pixel dilation is the main reason cyan sits a little outside the bright 488 body.
    # It is deliberate breathing room, not a second neuron boundary.
    final = morphology.dilation(filled, morphology.disk(CFG.foreground_dilation))

    # The normal cyan line is just a one-ish-pixel morphological boundary around that final mask.
    full_boundary = morphology.dilation(final) ^ morphology.erosion(final)

    return {
        "corrected": corrected,
        "smooth": smooth,
        "initial": initial,
        "cleaned": cleaned,
        "filled": filled,
        "final": final,
        "full_boundary": full_boundary,
        "threshold": threshold,
        "initial_pixels": int(initial.sum()),
        "cleaned_pixels": int(cleaned.sum()),
        "filled_pixels": int(filled.sum()),
        "final_pixels": int(final.sum()),
    }


def edge_flags(mask: np.ndarray) -> str:
    """Say which sides of the display crop cut through foreground."""

    flags = []
    if mask[0].any():
        flags.append("top")
    if mask[-1].any():
        flags.append("bottom")
    if mask[:, 0].any():
        flags.append("left")
    if mask[:, -1].any():
        flags.append("right")
    return "|".join(flags) if flags else "none"


def boundary_rgb(
    raw: np.ndarray,
    *,
    true_boundary: np.ndarray,
    cropped_first_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Color boundary pixels so crop artifacts cannot hide in plain sight."""

    gray = normalize(raw)
    rgb = np.repeat(gray[..., None], 3, axis=2) * 0.65
    if cropped_first_boundary is None:
        rgb[true_boundary] = np.array([0.0, 0.95, 1.0])
        return rgb

    shared = true_boundary & cropped_first_boundary
    full_only = true_boundary & ~cropped_first_boundary
    crop_only = cropped_first_boundary & ~true_boundary
    rgb[shared] = np.array([1.0, 0.9, 0.0])  # yellow = both ways agree
    rgb[full_only] = np.array([0.0, 0.95, 1.0])  # cyan = real full-frame edge only
    rgb[crop_only] = np.array([1.0, 0.2, 0.25])  # red = display-crop-only edge
    return rgb


def write_panel(
    path: Path,
    *,
    example_id: str,
    well: str,
    day: int,
    stable: np.ndarray,
    cy: float,
    cx: float,
    stages: dict[str, np.ndarray | float | int],
) -> dict[str, object]:
    """Build one why-is-it-cyan explainer panel and return its measurements."""

    size = 160
    raw_crop, y0, x0 = center_crop(stable, cy, cx, size)
    smooth_crop, _, _ = center_crop(np.asarray(stages["smooth"]), cy, cx, size)
    initial_crop, _, _ = center_crop(np.asarray(stages["initial"]), cy, cx, size)
    filled_crop, _, _ = center_crop(np.asarray(stages["filled"]), cy, cx, size)
    final_crop, _, _ = center_crop(np.asarray(stages["final"]), cy, cx, size)
    true_boundary_crop, _, _ = center_crop(np.asarray(stages["full_boundary"]), cy, cx, size)

    # Important visual-debug detail: this is how the original overlay drew cyan. It cropped the
    # final mask first, then asked for a boundary. If foreground hits a crop edge, that operation
    # can create extra cyan on the edge that was not a boundary in the full 2868x2868 frame.
    cropped_first_boundary = morphology.dilation(final_crop) ^ morphology.erosion(final_crop)
    crop_only = cropped_first_boundary & ~true_boundary_crop
    full_only = true_boundary_crop & ~cropped_first_boundary
    shared = true_boundary_crop & cropped_first_boundary

    # Show three crop sizes. The mask itself is identical; only how much context survives changes.
    size_views: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for crop_size in (96, 160, 256):
        raw_view, _, _ = center_crop(stable, cy, cx, crop_size)
        boundary_view, _, _ = center_crop(np.asarray(stages["full_boundary"]), cy, cx, crop_size)
        size_views[crop_size] = (raw_view, boundary_view)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8))
    axes[0, 0].imshow(normalize(raw_crop), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].add_patch(
        plt.Circle((size / 2, size / 2), 26, edgecolor="#57FF7A", facecolor="none", linestyle="--")
    )
    axes[0, 0].set_title("Raw 488 crop\ngreen=focal candidate")
    axes[0, 1].imshow(normalize(smooth_crop), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Sigma=2 smoothed 488\np65 threshold={float(stages['threshold']):.1f}")
    axes[0, 2].imshow(initial_crop, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Above-threshold pixels\nbefore size cleanup")

    dilation_added = final_crop & ~filled_crop
    dilation_rgb = np.zeros((*final_crop.shape, 3), dtype=float)
    dilation_rgb[filled_crop] = np.array([0.72, 0.72, 0.72])
    dilation_rgb[dilation_added] = np.array([0.0, 0.95, 1.0])
    axes[0, 3].imshow(dilation_rgb)
    axes[0, 3].set_title("Final foreground\ngray=filled | cyan=2 px dilation")

    for column, crop_size in enumerate((96, 160, 256)):
        raw_view, boundary_view = size_views[crop_size]
        axes[1, column].imshow(boundary_rgb(raw_view, true_boundary=boundary_view))
        axes[1, column].set_title(f"True full-frame boundary\nshown in {crop_size} px crop")

    axes[1, 3].imshow(
        boundary_rgb(
            raw_crop,
            true_boundary=true_boundary_crop,
            cropped_first_boundary=cropped_first_boundary,
        )
    )
    axes[1, 3].set_title(
        "Boundary source check\nyellow=shared | cyan=full-only | red=crop-only"
    )

    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(
        f"{example_id}. {well} Day {day} | Why Arm D foreground is cyan",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "Cyan is a display boundary around the 488-defined foreground. Mask sizing happens on the full ND2 frame; "
        "the 160 px crop only limits what is visible.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10, hspace=0.22, wspace=0.08)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    labels = measure.label(final_crop)
    edge_component_ids = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    edge_component_ids = edge_component_ids[edge_component_ids > 0]
    return {
        "example_id": example_id,
        "well": well,
        "day": day,
        "crop_size_pixels": size,
        "crop_x_start": x0,
        "crop_y_start": y0,
        "full_frame_foreground_pixels": int(np.asarray(stages["final"]).sum()),
        "crop_foreground_pixels": int(final_crop.sum()),
        "crop_foreground_fraction": float(final_crop.mean()),
        "crop_edges_touched": edge_flags(final_crop),
        "edge_touching_components": int(len(edge_component_ids)),
        "filled_pixels_in_crop": int(filled_crop.sum()),
        "dilation_added_pixels_in_crop": int(dilation_added.sum()),
        "true_full_frame_boundary_pixels_in_crop": int(true_boundary_crop.sum()),
        "cropped_first_boundary_pixels": int(cropped_first_boundary.sum()),
        "shared_boundary_pixels": int(shared.sum()),
        "full_frame_only_boundary_pixels": int(full_only.sum()),
        "crop_only_boundary_pixels": int(crop_only.sum()),
        "crop_only_boundary_fraction": float(crop_only.sum() / max(cropped_first_boundary.sum(), 1)),
        "interpretation": (
            "no extra crop-only cyan observed; crop size changes visible context, not the Arm D mask"
            if not crop_only.any()
            else "crop-only cyan is a rendering boundary effect; it does not add pixels to the Arm D mask"
        ),
    }


def write_contact_sheet(paths: list[Path]) -> None:
    """The contact sheet is just for quick scanning; full-size panels remain the real deliverable."""

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), constrained_layout=True)
    for axis, path in zip(axes.flat, paths, strict=True):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    fig.suptitle("Arm D cyan foreground diagnostics (8 neuron-candidate crops)", fontsize=19, fontweight="bold")
    fig.savefig(OUT / "arm_d_cyan_foreground_diagnostic_contact_sheet.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_readme(measurements: pd.DataFrame) -> None:
    """Leave the explanation beside the figures so the cyan line does not become lore."""

    crop_only_total = int(measurements["crop_only_boundary_pixels"].sum())
    displayed_total = int(measurements["cropped_first_boundary_pixels"].sum())
    crop_only_fraction = crop_only_total / max(displayed_total, 1)
    full_only_total = int(measurements["full_frame_only_boundary_pixels"].sum())
    true_total = int(measurements["true_full_frame_boundary_pixels_in_crop"].sum())
    text = f"""# Why the Arm D Foreground Is Cyan

The cyan line is a visualization boundary around Arm D's stable-488 foreground mask. Cyan is not a biological class and is not included as extra mCherry signal.

## Mask construction

1. Subtract the 20th-percentile background from the full 488 ND2 frame.
2. Smooth with Gaussian sigma 2.
3. Threshold using the 65th percentile of positive corrected 488 pixels.
4. Remove foreground components smaller than 128 pixels.
5. Fill holes.
6. Dilate the foreground by a disk of radius 2 pixels.
7. Draw the final mask boundary in cyan for the figure.

## Sizing and cropping

- The mask is computed on the full 2868 x 2868 frame, before the 160 x 160 neuron-candidate crop.
- Changing the display crop from 96 to 160 to 256 pixels changes context, not the underlying mask.
- In principle, recomputing a boundary after cropping can make a display-only crop edge. The source-check panel tests that directly.
- Across these eight panels, crop-only boundary pixels were {crop_only_total}/{displayed_total} ({crop_only_fraction:.1%}) of cropped-first boundary pixels.
- Cropping therefore added no cyan in this set; it omitted {full_only_total}/{true_total} true full-frame boundary pixels at cut edges.
- Even when display-boundary differences occur, they do not change the saved Arm D foreground or D:P calculation.

## Outputs

- `arm_d_cyan_foreground_diagnostic_contact_sheet.png`
- `individual_panels/*.png`
- `arm_d_cyan_foreground_measurements.csv`

## Local code sources

- `{WORKTREE / 'src/tmem_align/analysis/mcherry_metrics.py'}`
- `{SCRIPT_DIR / 'generate_neuron_post_mask_comparisons.py'}`
- `{SCRIPT_DIR / 'explain_arm_d_cyan_foreground.py'}`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    selected = pd.read_csv(METRICS, dtype={"example_id": str})
    measurements: list[dict[str, object]] = []
    panel_paths: list[Path] = []

    for _, row in selected.sort_values("example_id").iterrows():
        example_id = str(row["example_id"]).zfill(2)
        well = str(row["well"])
        day = int(row["day"])
        source = manifest[(manifest["well"] == well) & (manifest["day"] == day)].iloc[0]

        # Same deal as the comparison generator: ND2 in, no TIFF/PNG method input anywhere.
        array = _load_cyx(Path(str(source["file_path"])))
        stable = array[_channel_index(str(source["channel_names"]), "488")]
        stages = build_foreground_stages(stable)
        panel_path = PANELS / f"{example_id}_{well}_day{day}_arm_d_cyan_explainer.png"
        measurements.append(
            write_panel(
                panel_path,
                example_id=example_id,
                well=well,
                day=day,
                stable=stable,
                cy=float(row["candidate_centroid_y"]),
                cx=float(row["candidate_centroid_x"]),
                stages=stages,
            )
        )
        panel_paths.append(panel_path)

    frame = pd.DataFrame(measurements)
    frame.to_csv(OUT / "arm_d_cyan_foreground_measurements.csv", index=False)
    write_contact_sheet(panel_paths)
    write_readme(frame)
    print(f"Wrote {len(frame)} Arm D cyan diagnostics to {OUT}")


if __name__ == "__main__":
    main()
