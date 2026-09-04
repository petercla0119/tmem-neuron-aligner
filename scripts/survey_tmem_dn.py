"""Pick an absolute DN detection threshold for the 3D TMEM-LAMP1 coloc pipeline.

Mirror of survey_lamp1_otsu.py, with three deliberate differences:
  - operates on the 3D stack per-voxel (coloc thresholds the raw ZYX stack via
    detect_tmem_puncta_3d, NOT the MIP), so a MIP-derived DN would not transfer,
  - restricts to voxels inside the MAP2 cell-body masks (extruded across z),
  - references TMEM_KO as the null (no real cleaved-TMEM => it is the
    false-positive floor), NOT Control+KI as the LAMP1 survey does.

The DN we want is the level that floors out KO detection while Control/KI keep
real puncta. Two passes:
  1. per-FOV intra-cell 561 percentiles/Otsu -> candidate DNs from KO aggregates,
  2. re-run detect_tmem_puncta_3d at each candidate, report mean puncta/cell per
     condition, so you pick the smallest DN where KO floors out.

    PYTHONPATH=src python scripts/survey_tmem_dn.py configs/if_coloc_d7_smoke.yaml [max_per_cond]

Writes <output_dir>/<timepoint>_tmem_dn_survey.csv (per-FOV stats) and prints the
candidate DNs + the puncta/cell table. Ignores cfg.max_fovs (surveys all FOVs
unless the optional max_per_cond cap is given).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tmem_align.analysis.if_coloc import ColocConfig, detect_tmem_puncta_3d  # noqa: E402
from tmem_align.analysis.if_features import parse_fov_metadata  # noqa: E402
from tmem_align.analysis.if_spatial import load_fov_3d  # noqa: E402

from run_if_coloc import _cell_masks  # noqa: E402  (reuse the pipeline's mask loader)


def _cap_by_condition(paths: list[Path], max_per_cond: int | None) -> list[Path]:
    if max_per_cond is None:
        return paths
    seen: dict[str, int] = defaultdict(int)
    out: list[Path] = []
    for p in paths:
        c = parse_fov_metadata(p)["condition"]
        if seen[c] < max_per_cond:
            out.append(p)
            seen[c] += 1
    return out


def _puncta_per_cell(tmem_zyx: np.ndarray, cells_yx: np.ndarray, thr: float, min_vox: int) -> float:
    """Mean 561 puncta assigned to a cell, at absolute DN `thr` (matches the pipeline)."""
    _, centroids = detect_tmem_puncta_3d(tmem_zyx, threshold=thr, min_size_vox=min_vox)
    n_cells = int(np.unique(cells_yx).size - 1)
    if n_cells == 0:
        return float("nan")
    if centroids.shape[0] == 0:
        return 0.0
    iy = np.clip(np.rint(centroids[:, 1]).astype(int), 0, cells_yx.shape[0] - 1)
    ix = np.clip(np.rint(centroids[:, 2]).astype(int), 0, cells_yx.shape[1] - 1)
    return float(np.count_nonzero(cells_yx[iy, ix] > 0)) / n_cells


def main() -> None:
    cfg = ColocConfig.from_yaml(sys.argv[1])
    max_per_cond = int(sys.argv[2]) if len(sys.argv) > 2 else None

    root = Path(cfg.data_root) / cfg.timepoint
    paths = _cap_by_condition(sorted(root.rglob("*.nd2")), max_per_cond)
    print(
        f"tmem-dn survey: {len(paths)} FOVs under {root}"
        f"{'' if max_per_cond is None else f' (<= {max_per_cond}/condition)'}",
        flush=True,
    )

    # ---- pass 1: per-FOV intra-cell 561 percentiles + raw-stack Otsu ----
    from skimage.filters import threshold_otsu

    # Don't cache 3D stacks (72 FOVs would be ~15 GB); pass 2 reloads from disk.
    stats: list[dict] = []
    ok_paths: list[Path] = []
    for i, p in enumerate(paths):
        try:
            cells = _cell_masks(p, cfg)
            channels, _ = load_fov_3d(p)
            tmem = np.asarray(channels[cfg.tmem_channel], dtype=np.float32)  # ZYX
            cond = parse_fov_metadata(p)["condition"]
            mask3d = np.broadcast_to(cells > 0, tmem.shape)
            vox = tmem[mask3d]
            if vox.size == 0:
                print(f"  SKIP {p.name}: no in-cell voxels", flush=True)
                continue
            stats.append(
                {
                    "condition": cond,
                    "file": p.name,
                    "p99": float(np.percentile(vox, 99)),
                    "p99.9": float(np.percentile(vox, 99.9)),
                    "p99.99": float(np.percentile(vox, 99.99)),
                    "otsu_raw": float(threshold_otsu(vox)),
                    "n_cells": int(np.unique(cells).size - 1),
                }
            )
            ok_paths.append(p)
            print(f"  [{i + 1}/{len(paths)}] {p.name} ({cond})", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"  SKIP {p.name}: {exc}", flush=True)

    if not stats:
        print("no FOVs produced stats")
        return
    df = pd.DataFrame(stats)
    out = Path(cfg.output_dir) / f"{cfg.timepoint}_tmem_dn_survey.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n=== per-condition intra-cell 561 percentiles (median across FOVs) ===")
    print(
        df.groupby("condition")[["p99", "p99.9", "p99.99", "otsu_raw"]]
        .median()
        .round(1)
        .to_string()
    )

    # ---- candidate DNs from the KO null: median of KO per-FOV high percentiles ----
    # parse_fov_metadata labels the knockout condition "KO" (dir is TMEM_KO).
    ko = df[df["condition"] == "KO"]
    if ko.empty:
        print(
            f"\nNO KO FOVs found (conditions seen: {sorted(df['condition'].unique())}) -- "
            "cannot set a KO-referenced floor."
        )
        return
    candidates = {
        "KO_median_p99.9": float(ko["p99.9"].median()),
        "KO_median_p99.99": float(ko["p99.99"].median()),
        "KO_median_otsu": float(ko["otsu_raw"].median()),
    }
    print("\n=== candidate DNs (from KO null) ===")
    for k, v in candidates.items():
        print(f"  {k}: {v:.1f} DN")

    # ---- pass 2: validate puncta/cell per condition at each candidate DN ----
    # Reload each stack once, evaluate all candidate DNs on it (avoids re-reading per DN).
    print(
        "\n=== mean puncta/cell by condition at each candidate DN "
        "(want KO ~0, Control/KI retained) ==="
    )
    per_cand: dict[str, dict[str, list[float]]] = {n: defaultdict(list) for n in candidates}
    for p in ok_paths:
        try:
            cells = _cell_masks(p, cfg)
            channels, _ = load_fov_3d(p)
            tmem = np.asarray(channels[cfg.tmem_channel], dtype=np.float32)
            cond = parse_fov_metadata(p)["condition"]
            for name, thr in candidates.items():
                per_cand[name][cond].append(
                    _puncta_per_cell(tmem, cells, thr, cfg.min_puncta_size_vox)
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP (pass2) {p.name}: {exc}", flush=True)
    rows = []
    for name, thr in candidates.items():
        row = {"candidate": name, "DN": round(thr, 1)}
        for cond, vals in per_cand[name].items():
            row[cond] = round(float(np.nanmean(vals)), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nPick the smallest DN where TMEM_KO floors out (~0 puncta/cell) while "
        "Control/KI keep puncta, then set `tmem_threshold: <DN>` in the coloc config."
    )


if __name__ == "__main__":
    main()
