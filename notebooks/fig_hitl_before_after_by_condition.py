"""Arrayed before/after HITL figure across conditions.

Per FOV (rows), three columns:
  MAP2 (fixed-LUT)  |  BEFORE = stock cpsam  |  AFTER = fine-tuned map2_cellbody_cpsam

Both models run on the SAME fixed-LUT uint8 MAP2 the fine-tuned model trained on,
so the only variable is the fine-tuning. FOVs are held out (not in train or test),
2 per condition, spaced across the available wells.

Run:  python notebooks/fig_hitl_before_after_by_condition.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_spatial import CH_MAP2, apply_display_lut, load_fov

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
D7 = DATA / "d7"
MODEL = DATA / "hitl_map2_train" / "models" / "map2_cellbody_cpsam"
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)

# FOVs already seen during HITL (train + held-out test) — exclude so this figure
# only shows generalization to unseen FOVs.
SEEN = {
    "C8_F1", "F8_F4", "H8_F4", "E8_F1",
    "C17_F1", "F17_F4", "H17_F4", "E17_F1",
    "C20_F1", "F20_F4", "H20_F4", "E20_F1",
}
N_PER_COND = 2


def well_fov(nd2: Path) -> str:
    """...MAP2488DAPI405_D8_F2.nd2 -> 'D8_F2'"""
    return "_".join(nd2.stem.split("_")[-2:])


def pick_fovs(cond_dir: Path) -> list[Path]:
    unseen = [p for p in sorted(cond_dir.glob("*.nd2")) if well_fov(p) not in SEEN]
    if len(unseen) <= N_PER_COND:
        return unseen
    # spaced across the list so we don't just grab adjacent wells
    idx = np.linspace(0, len(unseen) - 1, N_PER_COND).round().astype(int)
    return [unseen[i] for i in idx]


def overlay(ax, gray, masks, title):
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[find_boundaries(masks, mode="outer")] = [0, 1, 1]
    ax.imshow(rgb)
    ax.set_title(f"{title} (n={int(masks.max())})", fontsize=10)
    ax.axis("off")


def main() -> None:
    stock = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    tuned = models.CellposeModel(gpu=True, pretrained_model=str(MODEL))

    rows = []
    for cond_dir in sorted(p for p in D7.iterdir() if p.is_dir()):
        for nd2 in pick_fovs(cond_dir):
            rows.append((cond_dir.name, nd2))

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)))
    for i, (cond, nd2) in enumerate(rows):
        map2 = load_fov(nd2)[CH_MAP2].astype(np.float32)
        gray = apply_display_lut(map2, CH_MAP2)
        img_u8 = (gray * 255).astype(np.uint8)

        before, _, _ = stock.eval(img_u8)
        after, _, _ = tuned.eval(img_u8)

        tag = f"{cond} {well_fov(nd2)}"
        r = axes[i]
        r[0].imshow(gray, cmap="gray")
        r[0].set_title(f"{tag} — MAP2", fontsize=10)
        r[0].axis("off")
        overlay(r[1], gray, before.astype(np.int32), f"{tag} — BEFORE (stock cpsam)")
        overlay(r[2], gray, after.astype(np.int32), f"{tag} — AFTER (fine-tuned)")
        print(f"{tag:<40} before n={int(before.max()):>3}  after n={int(after.max()):>3}")

    plt.suptitle(
        "MAP2 cell bodies before vs after HITL fine-tuning (unseen FOVs, by condition)",
        fontsize=15,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPORT / "hitl_before_after_by_condition.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
