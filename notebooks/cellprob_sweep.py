"""Does lowering cellprob_threshold recover real faint somata, or noise?

For the fine-tuned map2_cellbody_cpsam model, run each FOV twice:
  DEFAULT (cellprob_threshold=0.0)  vs  LOWERED (cellprob_threshold=-2.0)
across all 3 days x 3 conditions, several FOVs each, so we can eyeball whether
the *extra* masks at -2.0 are genuine dim cell bodies or hallucinated blobs.

Same fixed-LUT uint8 MAP2 the model trained on -> the only variable is the
threshold. One figure per day; the per-FOV count delta prints to stdout.

Run:  python notebooks/cellprob_sweep.py            # d7,d14,d28
      python notebooks/cellprob_sweep.py d14         # one day
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from tmem_align.analysis.if_spatial import CH_MAP2, apply_display_lut, load_fov

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
MODEL = DATA / "hitl_map2_train" / "models" / "map2_cellbody_cpsam"
REPORT = Path(__file__).parent.parent / "reports" / "if_segmentation_pilot"
REPORT.mkdir(parents=True, exist_ok=True)

DAYS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["d7", "d14", "d28"]
CP_LOW = -2.0
N_PER_COND = 4


def well_fov(nd2: Path) -> str:
    return "_".join(nd2.stem.split("_")[-2:])


def pick_fovs(cond_dir: Path, n: int) -> list[Path]:
    files = sorted(cond_dir.glob("*.nd2"))
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in dict.fromkeys(idx)]


def overlay(ax, gray, masks, title):
    rgb = np.stack([gray] * 3, axis=-1)
    edges = binary_dilation(find_boundaries(masks, mode="outer"), iterations=2)
    rgb[edges] = [0, 1, 1]
    ax.imshow(rgb)
    ax.set_title(f"{title} (n={int(masks.max())})", fontsize=10)
    ax.axis("off")


def run_day(model, day: str) -> None:
    day_dir = DATA / day
    rows = []
    for cond_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
        for nd2 in pick_fovs(cond_dir, N_PER_COND):
            rows.append((cond_dir.name, nd2))
    if not rows:
        print(f"[{day}] no FOVs found, skipping")
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (cond, nd2) in enumerate(rows):
        map2 = load_fov(nd2)[CH_MAP2].astype(np.float32)
        gray = apply_display_lut(map2, CH_MAP2)
        img_u8 = (gray * 255).astype(np.uint8)

        default, _, _ = model.eval(img_u8, cellprob_threshold=0.0)
        low, _, _ = model.eval(img_u8, cellprob_threshold=CP_LOW)
        default, low = default.astype(np.int32), low.astype(np.int32)

        tag = f"{day} {cond} {well_fov(nd2)}"
        r = axes[i]
        r[0].imshow(gray, cmap="gray")
        r[0].set_title(f"{tag} — MAP2", fontsize=10)
        r[0].axis("off")
        overlay(r[1], gray, default, f"{tag} — cellprob=0.0")
        overlay(r[2], gray, low, f"{tag} — cellprob={CP_LOW}")
        d0, d1 = int(default.max()), int(low.max())
        print(f"{tag:<42} n {d0:>3} -> {d1:>3}  (+{d1 - d0})")

    plt.suptitle(
        f"MAP2 cell bodies: cellprob 0.0 vs {CP_LOW} (fine-tuned model) — {day}",
        fontsize=15,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPORT / f"cellprob_sweep_{day}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}\n")


def main() -> None:
    model = models.CellposeModel(gpu=True, pretrained_model=str(MODEL))
    for day in DAYS:
        run_day(model, day)


if __name__ == "__main__":
    main()
