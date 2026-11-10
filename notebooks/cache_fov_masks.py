#!/usr/bin/env python3
"""Build per-FOV segmentation cache for the visual QC viewer.

Runs Cellpose (nuclei + cell bodies) and lysosome detection on every ND2
across d7/d14/d28, saving compact NPZ files so the QC viewer never re-runs
Cellpose. Resume-safe: skips FOVs whose NPZ already exists.

IMPORTANT: lyso detection params MUST match the fixed-threshold analysis run:
  bg_floor=120 (LAMP1 LUT floor), threshold=3765 (median Control+KI Otsu)

Usage:
    python notebooks/cache_fov_masks.py [--refresh]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tmem_align.analysis.if_features import detect_lysosomes  # noqa: E402
from tmem_align.analysis.if_spatial import (  # noqa: E402
    CH_DAPI,
    CH_LAMP1,
    CH_MAP2,
    load_fov,
    segment_cell_bodies,
    segment_nuclei,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
TIMEPOINTS = ["d7", "d14", "d28"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "reports/if_segmentation_pilot/fov_cache"

# Must match the fixed-threshold analysis run (d7_percell_fixedthr.csv)
BG_FLOOR: float = 120.0
LYSO_THRESHOLD: float = 3765.0


def cache_path(nd2_path: Path) -> Path:
    return CACHE_DIR / f"{nd2_path.stem}.npz"


def cache_fov(nd2_path: Path) -> None:
    cp = cache_path(nd2_path)
    ch = load_fov(nd2_path)
    nuclei = segment_nuclei(ch[CH_DAPI])
    cells = segment_cell_bodies(ch[CH_MAP2])
    lyso, lamp1_corr = detect_lysosomes(
        ch[CH_LAMP1], bg_floor=BG_FLOOR, threshold=LYSO_THRESHOLD
    )
    np.savez_compressed(
        cp,
        nuclei=nuclei,
        cells=cells,
        lyso_labels=lyso,
        lamp1_corr=lamp1_corr,
    )


def main() -> None:
    refresh = "--refresh" in sys.argv
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for tp in TIMEPOINTS for p in (DATA / tp).rglob("*.nd2"))
    print(f"Found {len(paths)} ND2 files across {TIMEPOINTS}")

    done, skipped, failed = 0, 0, 0
    for i, p in enumerate(paths, 1):
        cp = cache_path(p)
        if cp.exists() and not refresh:
            skipped += 1
            continue
        try:
            cache_fov(p)
            done += 1
            print(f"  [{i:3d}/{len(paths)}] cached {p.parent.name}/{p.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [{i:3d}/{len(paths)}] FAILED {p.name}: {exc}", flush=True)

    print(f"\nDone: {done} cached, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
