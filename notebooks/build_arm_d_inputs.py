#!/usr/bin/env python3
"""Build the input CSVs the Arm D figure scripts need — two candidate variants.

Writes one shared selection_manifest.csv, plus a neuron_post_mask_metrics.csv for
each variant:
  - v488_centered        : brightest neuron-sized 488 foreground blob (foreground-driven)
  - v561_puncta_centered : densest cluster of 561 mCherry puncta (puncta-driven)

Both figure scripts pick a variant via the ARM_D_VARIANT env var.

Run once before the figure scripts:
  source .venv/bin/activate
  python notebooks/build_arm_d_inputs.py
  for v in v488_centered v561_puncta_centered; do
    ARM_D_VARIANT=$v python notebooks/explain_arm_d_cyan_foreground.py
    ARM_D_VARIANT=$v python notebooks/explain_arm_d_puncta.py
  done
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tmem_align.analysis.mcherry_metrics import (
    MCherryMetricConfig,
    background_subtract,
    detect_puncta,
)

from explain_arm_d_cyan_foreground import (
    MANIFEST,
    ROOT,
    SOURCE,
    _channel_index,
    _load_cyx,
    build_foreground_stages,
)

CFG = MCherryMetricConfig()
VARIANTS = ("v488_centered", "v561_puncta_centered")

DATA = Path(
    "/Users/pmihack/claire/tmem_2026/data"
    "/260213_Feb16recopy_HYdiff_landingpadlines_survival_384well1"
)

# 8 fluorescence acquisition dirs (brightfield-only dir 20260216_094406_352 excluded)
DAY_DIRS = [
    "20260216_084435_959",
    "20260220_085237_196",
    "20260224_090320_759",
    "20260228_090815_318",
    "20260305_171612_406",
    "20260309_090353_210",
    "20260312_083635_497",
    "20260316_093139_667",
]
# mCherry-valid rows, rotating wells for variety
WELLS = ["F05", "J05", "N05", "E05", "I05", "M05", "F06", "J06"]


def channel_repr(path: Path) -> str:
    with nd2.ND2File(path) as f:
        names = [c.channel.name for c in f.metadata.channels]
    return repr(names)


def best_centroid(stable: np.ndarray) -> tuple[float, float]:
    """Brightest neuron-sized 488 foreground blob, biased toward frame center.

    Score = mean 488 intensity (on the background-subtracted image the mask is built
    from) divided by a centrality penalty. Picks a bright, cell-like blob the 160 px
    crop can show, instead of whatever tiny/dim blob is merely closest to center.
    """
    stages = build_foreground_stages(stable)
    final = stages["final"]
    corrected = np.asarray(stages["corrected"])
    labels = measure.label(final)
    props = measure.regionprops(labels, intensity_image=corrected)
    if not props:
        raise ValueError("No foreground found")
    cy_c, cx_c = final.shape[0] / 2, final.shape[1] / 2
    scale = final.shape[0] / 4  # centrality falloff; ~quarter-frame
    cands = [p for p in props if 128 <= p.area <= 8000] or props

    def score(p):
        dist = ((p.centroid[0] - cy_c) ** 2 + (p.centroid[1] - cx_c) ** 2) ** 0.5
        return p.intensity_mean / (1 + dist / scale)

    best = max(cands, key=score)
    return float(best.centroid[0]), float(best.centroid[1])


def best_puncta_centroid(
    mcherry: np.ndarray, foreground: np.ndarray, stable: np.ndarray
) -> tuple[float, float]:
    """Center of the densest cluster of 561 mCherry puncta.

    Detect puncta with the canonical pipeline (inside the 488 foreground), blur the
    puncta mask into a density map, and take the peak — i.e. where puncta cluster most
    tightly. Falls back to the 488 foreground pick when a frame has no puncta.
    """
    corrected = background_subtract(mcherry, percentile=CFG.background_percentile)
    puncta = detect_puncta(corrected, foreground, config=CFG)
    if not puncta.any():
        return best_centroid(stable)  # no puncta → fall back to foreground center
    # Intensity-weighted puncta density. mode="constant" so the border isn't inflated by
    # reflection; sigma ~= half the 160 px crop so the peak marks a crop-fillable cluster.
    weighted = np.where(puncta, corrected, 0).astype(np.float32)
    density = ndi.gaussian_filter(weighted, sigma=40.0, mode="constant")
    # Exclude a half-crop (80 px) border so the returned point is always a valid crop center.
    margin = 80
    density[:margin] = density[-margin:] = 0
    density[:, :margin] = density[:, -margin:] = 0
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    return float(cy), float(cx)


def main() -> None:
    manifest_rows: list[dict] = []
    # one metrics-row list per variant
    metrics_rows: dict[str, list[dict]] = {v: [] for v in VARIANTS}

    for i, (day_dir, well) in enumerate(zip(DAY_DIRS, WELLS)):
        acq = DATA / day_dir
        matches = sorted(acq.glob(f"*Well{well}_*.nd2"))
        if not matches:
            raise FileNotFoundError(f"No ND2 for well {well} in {day_dir}")
        path = matches[0]

        day_match = re.search(r"[Dd]ay(\d+)", path.name)
        if not day_match:
            raise ValueError(f"Cannot parse day from {path.name}")
        day = int(day_match.group(1))

        chan_repr = channel_repr(path)
        arr = _load_cyx(path)
        stable = arr[_channel_index(chan_repr, "488")]
        mcherry = arr[_channel_index(chan_repr, "561")]

        cy488, cx488 = best_centroid(stable)
        foreground = np.asarray(build_foreground_stages(stable)["final"])
        cy561, cx561 = best_puncta_centroid(mcherry, foreground, stable)

        print(
            f"  {i+1:02d} {well} day{day}: 488 ({cy488:.0f},{cx488:.0f}) "
            f"561 ({cy561:.0f},{cx561:.0f})  {path.name}"
        )
        manifest_rows.append(
            {"well": well, "day": day, "file_path": str(path), "channel_names": chan_repr}
        )
        base = {"example_id": str(i + 1).zfill(2), "well": well, "day": day}
        metrics_rows["v488_centered"].append(
            {**base, "candidate_centroid_y": cy488, "candidate_centroid_x": cx488}
        )
        metrics_rows["v561_puncta_centered"].append(
            {**base, "candidate_centroid_y": cy561, "candidate_centroid_x": cx561}
        )

    (ROOT / "05_stratified_24_ad_bridge").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST, index=False)
    print(f"\nWrote {MANIFEST}")

    mf = pd.read_csv(MANIFEST)
    assert len(mf) == 8, "Expected 8 manifest rows"
    for variant in VARIANTS:
        out_dir = SOURCE / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "neuron_post_mask_metrics.csv"
        pd.DataFrame(metrics_rows[variant]).to_csv(metrics_path, index=False)

        mt = pd.read_csv(metrics_path)
        assert len(mt) == 8, f"Expected 8 rows for {variant}"
        for _, row in mt.iterrows():
            match = mf[(mf["well"] == row["well"]) & (mf["day"] == row["day"])]
            assert len(match) == 1, f"No manifest match for {row['well']} day{row['day']}"
            assert Path(match.iloc[0]["file_path"]).exists()
        print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
