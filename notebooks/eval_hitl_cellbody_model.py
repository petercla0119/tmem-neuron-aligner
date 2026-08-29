"""Evaluate the fine-tuned MAP2 cell-body model on held-out FOVs.

Compares, per test FOV, against the human-corrected masks (ground truth):
  1. fine-tuned cpsam model  (the thing we just trained)
  2. seeded expansion        (the stopgap it's meant to replace)

Reports Average Precision at IoU {0.5, 0.75, 0.9} and writes a 3-col overlay
figure (GT | fine-tuned | seeded) per FOV so the numbers have a visual.

Run:  python notebooks/eval_hitl_cellbody_model.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cellpose import models
from cellpose.metrics import average_precision
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_spatial import (
    CH_DAPI,
    CH_MAP2,
    apply_display_lut,
    expand_to_cell_bodies,
    load_fov,
    segment_nuclei,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
TEST = DATA / "hitl_map2_test"
MODEL = DATA / "hitl_map2_train" / "models" / "map2_cellbody_cpsam"
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.5, 0.75, 0.9]


def nd2_for(stem: str) -> Path:
    """<cond>__<nd2stem> -> DATA/d7/<cond>/<nd2stem>.nd2"""
    cond, rest = stem.split("__", 1)
    return DATA / "d7" / cond / f"{rest}.nd2"


def main() -> None:
    model = models.CellposeModel(gpu=True, pretrained_model=str(MODEL))
    segs = sorted(TEST.glob("*_seg.npy"))
    fig, axes = plt.subplots(len(segs), 3, figsize=(18, 6 * len(segs)))
    rows = []
    for i, seg in enumerate(segs):
        stem = seg.name.replace("_seg.npy", "")
        gt = np.load(seg, allow_pickle=True).item()["masks"].astype(np.int32)
        map2_u8 = tifffile.imread(TEST / f"{stem}.tif")

        pred, _, _ = model.eval(map2_u8, batch_size=1)
        pred = pred.astype(np.int32)

        chs = load_fov(nd2_for(stem))
        base = expand_to_cell_bodies(segment_nuclei(chs[CH_DAPI]), chs[CH_MAP2].astype(np.float32))

        ap_ft = average_precision(gt, pred, threshold=THRESHOLDS)[0]
        ap_bl = average_precision(gt, base, threshold=THRESHOLDS)[0]
        rows.append((stem, gt.max(), pred.max(), base.max(), ap_ft, ap_bl))

        gray = apply_display_lut(chs[CH_MAP2].astype(np.float32), CH_MAP2)
        for col, (masks, title) in enumerate(
            [(gt, "GT (corrected)"), (pred, "fine-tuned cpsam"), (base, "seeded expansion")]
        ):
            rgb = np.stack([gray] * 3, axis=-1)
            edges = binary_dilation(find_boundaries(masks, mode="outer"), iterations=2)
            rgb[edges] = [0, 1, 1]
            ax = axes[i, col] if len(segs) > 1 else axes[col]
            ax.imshow(rgb)
            ax.set_title(f"{stem.split('__')[0]} — {title} (n={int(masks.max())})", fontsize=10)
            ax.axis("off")

    plt.suptitle("Held-out eval: fine-tuned cpsam vs seeded expansion vs corrected GT", fontsize=14)
    plt.tight_layout()
    out = REPORT / "hitl_cellbody_eval.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")

    print(f"\n{'FOV':<50}{'GT':>4}{'FT':>4}{'BL':>4}  AP@[.5/.75/.9] fine-tuned | seeded")
    for stem, ng, nf, nb, ap_ft, ap_bl in rows:
        ft = "/".join(f"{a:.2f}" for a in ap_ft)
        bl = "/".join(f"{a:.2f}" for a in ap_bl)
        print(f"{stem[:48]:<50}{ng:>4}{nf:>4}{nb:>4}  {ft}  |  {bl}")
    m_ft = np.mean([r[4] for r in rows], axis=0)
    m_bl = np.mean([r[5] for r in rows], axis=0)
    print(f"\nMEAN AP  fine-tuned: {'/'.join(f'{a:.2f}' for a in m_ft)}"
          f"   seeded: {'/'.join(f'{a:.2f}' for a in m_bl)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
