#!/usr/bin/env python3
"""Publication figures: progressive walk-through of the LAMP1 quantification pipeline.

Three figures on ONE representative Control FOV (d7):
  fig_pipeline_perchannel.png  — per-channel raw -> processed steps (rows=channels)
  fig_pipeline_merged.png      — 4-channel merge with progressive overlays
  fig_pipeline_singlecell.png  — single-cell crops, all channels + features

Uses the real pipeline (if_spatial + if_features); segmentation is cached to an
.npz so layout tweaks don't re-run Cellpose. Re-run with --refresh to re-segment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.color import label2rgb
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tmem_align.analysis.if_features import (  # noqa: E402
    detect_lysosomes,
    per_cell_features,
    qc_filter_cells,
)
from tmem_align.analysis.if_spatial import (  # noqa: E402
    CH_DAPI,
    CH_LAMP1,
    CH_MAP2,
    CH_TMEM,
    apply_display_lut,
    cell_foreground_mask,
    load_fov,
    segment_cell_bodies,
    segment_nuclei,
)

FOV = (
    "/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7"
    "/Z59_PLD_Control/PLD3Control_Plate1_d7_TMEM561LAMP1640MAP2488DAPI405_G17_F2.nd2"
)
OUT = Path(__file__).resolve().parent.parent / "reports/if_segmentation_pilot"
CACHE = OUT / "_fig_cache_G17_F2.npz"
PX_UM = 0.108

# CVD-aware fluorescence merge: blue / green / magenta / yellow (no pure red+green pair)
MERGE_COLORS = {
    CH_DAPI: ("DAPI (nuclei)", (0.0, 0.0, 1.0)),
    CH_MAP2: ("MAP2 (cytoskeleton)", (0.0, 1.0, 0.0)),
    CH_LAMP1: ("LAMP1 (lysosome)", (1.0, 0.0, 1.0)),
    CH_TMEM: ("TMEM106B (protein)", (1.0, 1.0, 0.0)),
}
TITLE_KW = dict(fontsize=9, pad=3)
PANEL_KW = dict(fontsize=13, fontweight="bold", color="white")


# --------------------------------------------------------------------------- #
# pipeline (cached)
# --------------------------------------------------------------------------- #
def run_pipeline(refresh: bool = False) -> dict:
    if CACHE.exists() and not refresh:
        print(f"Loading cached segmentation: {CACHE.name}")
        z = np.load(CACHE, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["feats"] = d["feats"].item()  # dict of arrays
        return d

    print("Segmenting (Cellpose, one-time) ...")
    ch = load_fov(FOV)
    nuclei = segment_nuclei(ch[CH_DAPI])
    cells = segment_cell_bodies(ch[CH_MAP2])
    fg = cell_foreground_mask(ch[CH_MAP2])
    lyso_labels, lamp1_corr = detect_lysosomes(ch[CH_LAMP1])

    feats = per_cell_features(cells, nuclei, ch, lyso_labels, lamp1_corr)
    qc = qc_filter_cells(cells, nuclei, ch[CH_DAPI])
    tab = feats.merge(qc, on="cell_id", how="left")

    d = {
        "dapi": ch[CH_DAPI], "map2": ch[CH_MAP2], "lamp1": ch[CH_LAMP1], "tmem": ch[CH_TMEM],
        "nuclei": nuclei, "cells": cells, "fg": fg,
        "lyso": lyso_labels, "lamp1_corr": lamp1_corr,
        "feats": {c: tab[c].to_numpy() for c in tab.columns},
    }
    np.savez_compressed(CACHE, **{k: v for k, v in d.items() if k != "feats"}, feats=d["feats"])
    print(f"Cached -> {CACHE.name}")
    return d


# --------------------------------------------------------------------------- #
# small display helpers
# --------------------------------------------------------------------------- #
def disp(img, ch_key):
    return apply_display_lut(img, ch_key)


def gray_rgb(img01):
    return np.dstack([img01] * 3)


def show(ax, img):
    """imshow, clipping float RGB to [0,1] (label2rgb can emit ~1.0000001)."""
    ax.imshow(np.clip(img, 0, 1) if np.issubdtype(np.asarray(img).dtype, np.floating) else img)


def tint(img01, color):
    return np.dstack([img01 * c for c in color])


def merge_rgb(d, keys=(CH_DAPI, CH_MAP2, CH_LAMP1, CH_TMEM)):
    key2arr = {CH_DAPI: d["dapi"], CH_MAP2: d["map2"], CH_LAMP1: d["lamp1"], CH_TMEM: d["tmem"]}
    rgb = np.zeros((*d["dapi"].shape, 3), np.float32)
    for k in keys:
        rgb += tint(disp(key2arr[k], k), MERGE_COLORS[k][1])
    return np.clip(rgb, 0, 1)


def overlay_outlines(rgb, labels, color, width=2):
    b = find_boundaries(labels, mode="outer")
    if width > 1:
        b = dilation(b, disk(width - 1))
    out = rgb.copy()
    out[b] = color
    return out


def add_scalebar(ax, img_shape, length_um=20, color="white"):
    h, w = img_shape[:2]
    lp = length_um / PX_UM
    x0, y0 = w * 0.97 - lp, h * 0.95
    ax.add_patch(Rectangle((x0, y0), lp, max(h, w) * 0.008, color=color, ec="black", lw=0.5))
    ax.text(x0 + lp / 2, y0 - h * 0.012, f"{length_um} µm",
            color=color, ha="center", va="bottom", fontsize=8, fontweight="bold")


def panel_letter(ax, letter):
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, va="top", ha="left", **PANEL_KW,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))


def blank(ax):
    ax.axis("off")


def add_figure_title(fig, grid, n_top_cols, text, fontsize=14, pad_frac=0.01):
    """Place the figure title just above the top-row axes titles by measuring
    their actual rendered position — reliable across all figure sizes and fonts.
    Replacing suptitle(y=...) avoids the ImageGrid-overlaps-suptitle problem
    where axes fill rect=111 and push their titles above y=1.0.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    tops = [
        grid[c].title.get_window_extent(renderer).y1
        for c in range(n_top_cols)
        if grid[c].title.get_text()
    ]
    y = (max(tops) if tops else fig_h_px) / fig_h_px + pad_frac
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# --------------------------------------------------------------------------- #
# Figure 1 — per-channel walk-through
# --------------------------------------------------------------------------- #
def figure1(d):
    # rows = channel, each a list of (title, rgb_image); padded to 3 cols
    puncta_disp = dilation(d["lyso"] > 0, disk(2))
    lamp1c = d["lamp1_corr"].astype(np.float32)
    lamp1c01 = np.clip(lamp1c / (np.percentile(lamp1c, 99.9) + 1e-6), 0, 1)
    lamp_punc = gray_rgb(lamp1c01)
    lamp_punc[puncta_disp] = (1.0, 0.0, 1.0)

    tmem01 = disp(d["tmem"], CH_TMEM)
    from skimage.filters import threshold_otsu
    tmem_mask = d["tmem"] > threshold_otsu(d["tmem"].astype(np.float32))
    tmem_over = tint(tmem01, MERGE_COLORS[CH_TMEM][1])
    tmem_over[dilation(tmem_mask, disk(1))] = (1.0, 1.0, 1.0)

    rows = [
        ("DAPI 405nm", [
            ("raw\nload_fov", gray_rgb(disp(d["dapi"], CH_DAPI))),
            ("nuclei labels\nsegment_nuclei", label2rgb(d["nuclei"], image=disp(d["dapi"], CH_DAPI),
                                                        bg_label=0, alpha=0.45, image_alpha=1.0)),
            None,
        ]),
        ("MAP2 488nm", [
            ("raw\nload_fov", gray_rgb(disp(d["map2"], CH_MAP2))),
            ("foreground\ncell_foreground_mask", tint(d["fg"].astype(float), (0.0, 1.0, 0.0))),
            ("cell bodies\nsegment_cell_bodies", label2rgb(d["cells"], image=disp(d["map2"], CH_MAP2),
                                                           bg_label=0, alpha=0.45, image_alpha=1.0)),
        ]),
        ("LAMP1 640nm", [
            ("raw\nload_fov", gray_rgb(disp(d["lamp1"], CH_LAMP1))),
            ("bg-subtracted\ndetect_lysosomes", gray_rgb(lamp1c01)),
            ("puncta\ndetect_lysosomes", lamp_punc),
        ]),
        ("TMEM 561nm", [
            ("raw\nload_fov", tint(tmem01, MERGE_COLORS[CH_TMEM][1])),
            ("coloc mask (Otsu)\nper_cell_features", tmem_over),
            None,
        ]),
    ]

    fig = plt.figure(figsize=(13, 17))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 3), axes_pad=(0.15, 0.55), share_all=True)
    letters = iter("ABCDEFGHIJKL")
    for r, (ch_name, panels) in enumerate(rows):
        for c in range(3):
            ax = grid[r * 3 + c]
            if panels[c] is None:
                blank(ax)
                continue
            title, img = panels[c]
            show(ax, img)
            ax.set_title(title, **TITLE_KW)
            ax.set_xticks([])
            ax.set_yticks([])
            panel_letter(ax, next(letters))
            if c == 0:
                ax.set_ylabel(ch_name, fontsize=12, fontweight="bold")
            if r == 0 and c == 0:
                add_scalebar(ax, img.shape)
    add_figure_title(fig, grid, 3,
                     "LAMP1 pipeline — per-channel walk-through (Control d7, well G17 FOV F2)",
                     fontsize=15)
    fig.savefig(OUT / "fig_pipeline_perchannel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pipeline_perchannel.png")


# --------------------------------------------------------------------------- #
# Figure 2 — merged 4-channel walk-through
# --------------------------------------------------------------------------- #
def figure2(d):
    base = merge_rgb(d)
    step_nuc = overlay_outlines(base, d["nuclei"], (0.0, 1.0, 1.0), width=2)
    step_cell = overlay_outlines(step_nuc, d["cells"], (1.0, 1.0, 1.0), width=2)
    puncta = dilation(d["lyso"] > 0, disk(2))
    step_lyso = step_cell.copy()
    step_lyso[puncta] = (1.0, 0.4, 0.0)  # orange puncta pop against magenta LAMP1

    # final QC-annotated: green pass / red fail
    f = d["feats"]
    pass_ids = set(f["cell_id"][f["qc_pass"].astype(bool)])
    final = base.copy()
    final = overlay_outlines(final, np.where(np.isin(d["cells"], list(pass_ids)), d["cells"], 0),
                             (0.0, 1.0, 0.0), width=2)
    fail_ids = [i for i in f["cell_id"] if i not in pass_ids]
    final = overlay_outlines(final, np.where(np.isin(d["cells"], fail_ids), d["cells"], 0),
                             (1.0, 0.0, 0.0), width=2)

    steps = [
        ("A  raw 4-channel merge", base),
        ("B  + nuclei outlines", step_nuc),
        ("C  + cell-body outlines", step_cell),
        ("D  + lysosome puncta", step_lyso),
        ("E  QC: pass=green fail=red", final),
    ]
    fig = plt.figure(figsize=(24, 5.6))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 5), axes_pad=0.12, share_all=True)
    for ax, (title, img) in zip(grid, steps):
        show(ax, img)
        ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    add_scalebar(grid[0], base.shape)

    # channel color legend + QC counts on last panel
    handles = [Patch(facecolor=col, label=name) for name, col in MERGE_COLORS.values()]
    grid[0].legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.7,
                   handlelength=1.0, borderpad=0.4)
    n_pass, n_fail = len(pass_ids), len(fail_ids)
    grid[4].text(0.02, 0.02, f"pass={n_pass}  fail={n_fail}", transform=grid[4].transAxes,
                 fontsize=9, color="white", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))
    add_figure_title(fig, grid, 5,
                     "LAMP1 pipeline — merged 4-channel FOV walk-through (Control d7, G17 F2)",
                     fontsize=14)
    fig.savefig(OUT / "fig_pipeline_merged.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pipeline_merged.png")


# --------------------------------------------------------------------------- #
# Figure 3 — single-cell crops
# --------------------------------------------------------------------------- #
def _pick_cells(d, n=3, half=130):
    """QC-pass cells spanning low/med/high lysosome count, as fixed square crops.

    Each crop is a fixed side=2*half px window centered on the cell centroid so
    all panels are uniform squares (ImageGrid forces square axes — variable-size
    bboxes leave white bars). Only cells whose centroid sits far enough from the
    edge to fit the full window are eligible.
    """
    f = d["feats"]
    h, w = d["cells"].shape
    ok = f["qc_pass"].astype(bool)
    ids = f["cell_id"][ok]
    nl = f["n_lysosomes"][ok]

    # centroid + interior eligibility per candidate
    elig = []  # (cid, n_lyso, y0, x0)
    for cid, n_lyso in zip(ids, nl):
        ys, xs = np.nonzero(d["cells"] == cid)
        cy, cx = int(ys.mean()), int(xs.mean())
        if half <= cy < h - half and half <= cx < w - half:
            elig.append((int(cid), int(n_lyso), cy - half, cx - half))
    if not elig:
        return []
    elig.sort(key=lambda e: e[1])  # by lysosome count

    picks = []
    for frac in (0.2, 0.55, 0.9):  # low / med / high
        cid, _, y0, x0 = elig[int(frac * (len(elig) - 1))]
        if cid not in [p[0] for p in picks]:
            picks.append((cid, (y0, y0 + 2 * half, x0, x0 + 2 * half)))
        if len(picks) == n:
            break
    return picks


def figure3(d):
    picks = _pick_cells(d)
    f = d["feats"]
    id2row = {c: i for i, c in enumerate(f["cell_id"])}
    ncol = 7
    fig = plt.figure(figsize=(2.05 * ncol, 2.3 * len(picks) + 0.4))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(picks), ncol), axes_pad=(0.08, 0.35))
    col_titles = ["DAPI", "MAP2", "LAMP1", "TMEM", "merge", "cell-body\nmask", "lysosome\npuncta"]

    for ri, (cid, (y0, y1, x0, x1)) in enumerate(picks):
        sl = (slice(y0, y1), slice(x0, x1))
        cropch = {
            "DAPI": tint(disp(d["dapi"][sl], CH_DAPI), MERGE_COLORS[CH_DAPI][1]),
            "MAP2": tint(disp(d["map2"][sl], CH_MAP2), MERGE_COLORS[CH_MAP2][1]),
            "LAMP1": tint(disp(d["lamp1"][sl], CH_LAMP1), MERGE_COLORS[CH_LAMP1][1]),
            "TMEM": tint(disp(d["tmem"][sl], CH_TMEM), MERGE_COLORS[CH_TMEM][1]),
        }
        cellmask = d["cells"][sl] == cid
        merge = merge_rgb(d)[sl]
        cb = merge.copy()
        cb = overlay_outlines(cb, cellmask.astype(int), (1.0, 1.0, 1.0), width=1)
        # puncta belonging to this cell only
        lyso_here = np.where(cellmask, d["lyso"][sl], 0)
        pun = gray_rgb(np.clip(disp(d["lamp1"][sl], CH_LAMP1), 0, 1))
        pun[dilation(lyso_here > 0, disk(1))] = (1.0, 0.4, 0.0)
        pun = overlay_outlines(pun, cellmask.astype(int), (0.2, 0.8, 1.0), width=1)

        imgs = [cropch["DAPI"], cropch["MAP2"], cropch["LAMP1"], cropch["TMEM"], merge, cb, pun]
        for ci, img in enumerate(imgs):
            ax = grid[ri * ncol + ci]
            show(ax, img)
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], **TITLE_KW)
        # feature annotation on the puncta panel
        row = id2row[cid]
        txt = (f"cell {cid}\n"
               f"n_lyso={int(f['n_lysosomes'][row])}\n"
               f"size={f['lyso_mean_size_um2'][row]:.2f} µm²\n"
               f"LAMP1={f['lamp1_mean'][row]:.0f}\n"
               f"M1={f['tmem_manders_m1'][row]:.3f}")
        grid[ri * ncol].set_ylabel(f"cell {cid}", fontsize=10, fontweight="bold")
        grid[ri * ncol + ncol - 1].text(
            1.03, 0.5, txt, transform=grid[ri * ncol + ncol - 1].transAxes,
            fontsize=8, va="center", ha="left", family="monospace")
        add_scalebar(grid[ri * ncol], merge.shape, length_um=5)

    add_figure_title(fig, grid, ncol,
                     "LAMP1 pipeline — single-cell crops with extracted features (Control d7, G17 F2)",
                     fontsize=13)
    fig.savefig(OUT / "fig_pipeline_singlecell.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pipeline_singlecell.png")


def main():
    refresh = "--refresh" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    d = run_pipeline(refresh=refresh)
    f = d["feats"]
    print(f"cells={len(f['cell_id'])}  qc_pass={int(f['qc_pass'].astype(bool).sum())}")
    figure1(d)
    figure2(d)
    figure3(d)
    print("done")


if __name__ == "__main__":
    main()
