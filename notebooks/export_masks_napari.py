#!/usr/bin/env python3
"""Export segmentation masks as Napari-compatible Label TIFFs.

Reads per-FOV NPZ caches built by cache_fov_masks.py and writes one TIFF per
mask type per FOV, organised so Napari can open a whole condition folder as a
stack of Label layers.

Output tree:
  reports/if_segmentation_pilot/napari_labels/
    {timepoint}/
      {condition}/
        {fov_stem}/
          nuclei.tif        -- int32 label array: one integer per nucleus
          cells.tif         -- int32 label array: one integer per cell body
          lysosomes.tif     -- int32 label array: one integer per lysosome punctum
          lamp1_corr.tif    -- float32 background-subtracted LAMP1 (reference channel)

Usage:
    python notebooks/export_masks_napari.py [--refresh]

Napari:
    File -> Open Folder -> napari_labels/d7/TMEM_KO/
    Each .tif opens as an Image or Labels layer. For Labels: drag the file into
    Napari or use File -> Open -> select .tif -> Layer type: Labels.
    lamp1_corr.tif opens as Image for reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as e:
    raise ImportError("pip install tifffile") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tmem_align.analysis.if_features import CONDITION_DIRS, parse_fov_metadata  # noqa: E402

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
TIMEPOINTS = ["d7", "d14", "d28"]
CACHE_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports/if_segmentation_pilot/fov_cache"
)
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports/if_segmentation_pilot/napari_labels"
)

# Maps condition-dir name -> short label (same as CONDITION_DIRS)
_COND = CONDITION_DIRS  # {"TMEM_KO": "KO", ...}


def export_fov(nd2_path: Path, refresh: bool = False) -> bool:
    """Export masks for one FOV. Returns True if written, False if skipped."""
    stem = nd2_path.stem
    cache = CACHE_DIR / f"{stem}.npz"
    if not cache.exists():
        print(f"  SKIP (no cache): {stem}", flush=True)
        return False

    meta = parse_fov_metadata(nd2_path)
    tp = meta.get("timepoint") or nd2_path.parent.parent.name
    cond_dir = nd2_path.parent.name
    cond = _COND.get(cond_dir, cond_dir)

    fov_out = OUT_DIR / tp / cond / stem
    sentinel = fov_out / "cells.tif"
    if sentinel.exists() and not refresh:
        return False

    fov_out.mkdir(parents=True, exist_ok=True)
    z = np.load(cache)

    tifffile.imwrite(fov_out / "nuclei.tif",    z["nuclei"].astype(np.int32),    photometric="minisblack")
    tifffile.imwrite(fov_out / "cells.tif",     z["cells"].astype(np.int32),     photometric="minisblack")
    tifffile.imwrite(fov_out / "lysosomes.tif", z["lyso_labels"].astype(np.int32), photometric="minisblack")
    tifffile.imwrite(fov_out / "lamp1_corr.tif", z["lamp1_corr"].astype(np.float32), photometric="minisblack")
    return True


def main() -> None:
    refresh = "--refresh" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for tp in TIMEPOINTS for p in (DATA / tp).rglob("*.nd2"))
    print(f"Exporting masks for {len(paths)} FOVs -> {OUT_DIR}", flush=True)

    written, skipped, no_cache = 0, 0, 0
    for i, p in enumerate(paths, 1):
        result = export_fov(p, refresh=refresh)
        if result:
            written += 1
            if written % 10 == 0 or written == 1:
                print(f"  [{i:3d}/{len(paths)}] {p.parent.name}/{p.stem}", flush=True)
        else:
            cache = CACHE_DIR / f"{p.stem}.npz"
            if not cache.exists():
                no_cache += 1
            else:
                skipped += 1

    print(f"\nDone: {written} exported, {skipped} skipped (already exist), {no_cache} missing cache.")
    print(f"Open in Napari: File -> Open Folder -> {OUT_DIR}/d7/KO/")
    print("Set layer type to 'Labels' for nuclei/cells/lysosomes.tif; 'Image' for lamp1_corr.tif.")


if __name__ == "__main__":
    main()
