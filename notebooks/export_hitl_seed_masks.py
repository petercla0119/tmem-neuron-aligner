"""Export starter cell-body masks for Cellpose human-in-the-loop (HITL) training.

The shortcut for the MAP2 cell-body model: instead of drawing ~100-200 ROI from
scratch, seed the Cellpose GUI with masks we already have from
`expand_to_cell_bodies()`, then *correct* them. This writes, per FOV:

  <name>.tif       MAP2 max-projection, fixed-LUT uint8 (the exact display the
                   annotator sees; also what segment_cell_bodies() feeds at
                   inference, so training/inference input stay consistent)
  <name>_seg.npy   Cellpose seg dict {masks, outlines, ...} the GUI auto-loads

Open the output folder in the GUI (`python -m cellpose`) and each image loads
with its starter masks ready to correct.

Run:  python notebooks/export_hitl_seed_masks.py
"""
from pathlib import Path

import numpy as np
import tifffile
from cellpose.utils import masks_to_outlines

from tmem_align.analysis.if_spatial import (
    CH_DAPI,
    CH_MAP2,
    apply_display_lut,
    expand_to_cell_bodies,
    load_fov,
    segment_nuclei,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
# Training data is derived image data → keep it out of git, next to the raw data.
OUT = DATA.parent / "hitl_map2_train"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
N_PER_CONDITION = 4  # 4 x 3 = 12 FOVs; correct these -> ~100-200 ROI target


def pick_fovs(cond_dir: Path, n: int) -> list[Path]:
    """Evenly spaced across the condition's FOVs, to spread over wells/fields."""
    files = sorted(cond_dir.glob("*.nd2"))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in dict.fromkeys(idx)]  # dedup, keep order


def write_seg(stem: str, map2_u8: np.ndarray, masks: np.ndarray) -> None:
    """Write <stem>.tif + <stem>_seg.npy in the format the Cellpose GUI loads."""
    tif = OUT / f"{stem}.tif"
    tifffile.imwrite(tif, map2_u8)
    outlines = (masks * masks_to_outlines(masks)).astype(np.uint16)
    dat = {
        "masks": masks.astype(np.uint16),
        "outlines": outlines,
        "chan_choose": [0, 0],
        "ismanual": np.zeros(int(masks.max()), bool),
        "filename": str(tif),
        "img": map2_u8,  # fallback if the GUI can't find the .tif
        "flows": [],  # GUI recomputes; empty is tolerated (wrapped in try/except)
        "diameter": np.nan,
    }
    np.save(OUT / f"{stem}_seg.npy", dat)


def main() -> None:
    total = 0
    for cond in CONDITIONS:
        for nd2 in pick_fovs(DATA / cond, N_PER_CONDITION):
            chs = load_fov(nd2)
            nuclei = segment_nuclei(chs[CH_DAPI])
            map2 = chs[CH_MAP2].astype(np.float32)
            bodies = expand_to_cell_bodies(nuclei, map2)
            map2_u8 = (apply_display_lut(map2, CH_MAP2) * 255).astype(np.uint8)
            stem = f"{cond}__{nd2.stem}"
            write_seg(stem, map2_u8, bodies)
            print(f"{stem}: {int(bodies.max())} starter ROI")
            total += 1
    print(f"\nWrote {total} FOVs to {OUT}")
    print(f"Next: python -m cellpose  ->  File > Load folder  ->  {OUT}")


if __name__ == "__main__":
    main()
