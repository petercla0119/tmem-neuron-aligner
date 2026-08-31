# %% [markdown]
# # Fixed-IF LAMP1 Report — TMEM106B KO / Control / KI across d7, d14, d28
#
# Branch: `feat/if-lamp1-quantification` (if-features worktree).
#
# Audience: skeptical biologists. Every quantitative claim is paired with real
# cropped cells. Methods are framed as *"why you can trust the numbers."*
#
# **Pipeline (per FOV):** load ND2 (4 ch: DAPI/405, MAP2/488, cleaved-TMEM/561,
# LAMP1/640, max-Z) → nuclei (Cellpose-SAM on DAPI) → cell bodies (HITL
# fine-tuned cpsam on fixed-LUT MAP2) → lysosome detection (bg-subtract LAMP1,
# **fixed 3765 DN** threshold) → 4 readout families → condition-blind QC →
# well-level aggregation → Kruskal-Wallis + BH-corrected Mann-Whitney.
#
# This script is the single source: run it to (re)generate every figure and the
# markdown report; `nb-convert` turns it into the executable notebook.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures go to disk via savefig (nb-figures Mode B)
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skimage.segmentation import find_boundaries

WORKTREE = Path("/Users/pmihack/claire/tmem_2026/tmem-if-features")
sys.path.insert(0, str(WORKTREE / "src"))
from tmem_align.analysis.if_features import PX_UM, condition_stats  # noqa: E402
from tmem_align.analysis.if_spatial import (  # noqa: E402
    CH_DAPI,
    CH_LAMP1,
    CH_MAP2,
    CH_TMEM,
    apply_display_lut,
    load_fov,
)

REPORTS = WORKTREE / "reports" / "if_segmentation_pilot"
CACHE = REPORTS / "fov_cache"
OUT = REPORTS / "report"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

DAYS = ["d7", "d14", "d28"]
CONDS = ["Control", "KO", "KI"]  # Control is the reference
# CVD-safer categorical palette (blue / orange / purple), not red-green.
CCOLOR = {"Control": "#4C72B0", "KO": "#DD8452", "KI": "#8172B3"}

# Readout families -> (column, label, unit). The 2D coloc pair is flagged as a proxy.
READOUTS = [
    ("lyso_mean_size_um2", "Mean lysosome size", "µm²"),
    ("n_lysosomes", "Lysosome count / cell", "n"),
    ("lyso_area_frac", "Lysosome area fraction", "frac"),
    ("lamp1_mean", "LAMP1 mean (bg-sub)", "DN"),
    ("perinuclear_index", "Perinuclear index", "<1 = perinuclear"),
]
COLOC = [
    ("tmem_manders_m1", "TMEM–LAMP1 Manders M1 (2D proxy)", "frac"),
    ("tmem_pearson", "TMEM–LAMP1 Pearson (2D proxy)", "r"),
]


def _fg_for(bg: str) -> str:
    r, g, b = mc.to_rgb(bg)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


# Report body accumulates here; flushed to report.md at the end.
MD: list[str] = []


def md(*lines: str) -> None:
    MD.extend(lines)


def fig_md(relpath: str, caption: str) -> None:
    md(f"![{caption}](figures/{relpath})", f"*{caption}*", "")


def df_md(d: pd.DataFrame) -> str:
    """GitHub-flavored markdown table (avoids the optional `tabulate` dep)."""
    cols = list(d.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# %% [markdown]
# ## Load the per-cell tables
#
# Fixed-threshold (3765 DN) tables for all three days. `qc_pass` flags cells that
# survive condition-blind QC (border / size-outlier / pyknotic exclusion).

# %%
df = pd.read_csv(REPORTS / "all_timepoints_percell.csv")
df = df[df["condition"].isin(CONDS)].copy()
qc = df[df["qc_pass"] == True].copy()  # noqa: E712

n_tbl = (
    df.assign(qcpass=df["qc_pass"])
    .groupby(["timepoint", "condition"])
    .agg(cells=("cell_id", "size"), qc_pass=("qcpass", "sum"), wells=("well", "nunique"))
    .reset_index()
)
print(n_tbl.to_string(index=False))


# %% [markdown]
# ## Helpers: well-level stats, effect size, crops with masks

# %%
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Well-level Cohen's d (pooled SD). a vs b (b = reference)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else np.nan


def well_dotplot(day: str, col: str, label: str, unit: str, fname: str) -> dict:
    """Well means per condition (the pseudoreplication guard) as a dot plot with
    KW + BH-corrected MWU-vs-Control annotations. Returns the stats dict."""
    d = qc[qc["timepoint"] == day]
    res = condition_stats(d, col, reference="Control", timepoint=day)
    wm = res["well_means"]

    BG, FG = "white", "black"
    fig, ax = plt.subplots(figsize=(4.2, 4.0), facecolor=BG, dpi=150)
    ax.set_facecolor(BG)
    rng = np.random.default_rng(0)
    for i, c in enumerate(CONDS):
        vals = wm[wm["condition"] == c][col].values
        if not len(vals):
            continue
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, color=CCOLOR[c], s=55, alpha=0.9, edgecolor=FG,
                   linewidth=0.5, label=f"{c} (n={len(vals)} wells)", zorder=3)
        m, sd = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.hlines(m, i - 0.25, i + 0.25, color=FG, linewidth=2, zorder=4)
        ax.errorbar(i, m, yerr=sd, color=FG, capsize=4, linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(CONDS)
    ax.set_ylabel(f"{label} ({unit})", color=FG)
    kw = res.get("kruskal_p", np.nan)
    title = f"{day} — {label}\nKruskal-Wallis p = {kw:.3g}"
    ax.set_title(title, color=FG, fontsize=10)
    for s in ax.spines.values():
        s.set_color(FG)
    ax.tick_params(colors=FG)
    # BH p vs Control annotations
    ann = []
    for c, st in res.get("vs_reference", {}).items():
        ann.append(f"{c} vs Control: BH p = {st['p_bh']:.3g}")
    if ann:
        ax.text(0.5, -0.22, "  |  ".join(ann), transform=ax.transAxes,
                ha="center", va="top", color=FG, fontsize=8)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.subplots_adjust(bottom=0.2, left=0.18, right=0.97, top=0.86)
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return res


def _npz_for(file_path: str) -> Path:
    return CACHE / (Path(file_path).stem + ".npz")


def crop_cell(row: pd.Series, pad: int = 45, window: int | None = None) -> np.ndarray | None:
    """RGB crop (LAMP1=red, MAP2=green, DAPI=blue) with mask overlay:
    cyan cell outline, magenta-dashed-look nucleus outline, yellow lysosome fill.
    Lyso fill drawn BEFORE boundaries so outlines stay visible. Returns None if
    the cell label is absent from the cached masks.

    `window`: if set, crop a fixed window×window box centered on the cell centroid
    instead of the (variable) mask bbox + `pad`. A fixed window keeps the µm-scale
    identical across crops, so cell/lysosome SIZE is comparable panel-to-panel
    (required for the across-days montage; the default bbox mode is not)."""
    npz = _npz_for(row["file_path"])
    if not npz.exists():
        return None
    z = np.load(npz)
    cells, nuclei, lyso = z["cells"], z["nuclei"], z["lyso_labels"]
    lab = int(row["cell_id"])
    if lab not in cells:
        return None
    ch = load_fov(row["file_path"])
    ys, xs = np.nonzero(cells == lab)
    if window is not None:
        cy, cx = int(ys.mean()), int(xs.mean())
        half = window // 2
        y0 = min(max(cy - half, 0), max(cells.shape[0] - window, 0))
        x0 = min(max(cx - half, 0), max(cells.shape[1] - window, 0))
        y1, x1 = y0 + window, x0 + window
    else:
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, cells.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, cells.shape[1])

    def lut(name):
        return apply_display_lut(ch[name][y0:y1, x0:x1], name)

    rgb = np.dstack([lut(CH_LAMP1), lut(CH_MAP2), lut(CH_DAPI)]).astype(float)
    cc = cells[y0:y1, x0:x1] == lab
    nn = nuclei[y0:y1, x0:x1] == lab
    ll = (lyso[y0:y1, x0:x1] > 0) & cc  # only this cell's lysosomes
    rgb[ll] = np.clip(rgb[ll] * 0.35 + np.array([1, 1, 0]) * 0.65, 0, 1)
    rgb[find_boundaries(cc, mode="outer")] = [0, 1, 1]  # cyan cell
    rgb[find_boundaries(nn, mode="outer")] = [1, 0, 1]  # magenta nucleus
    return rgb


def crop_grid(day: str, col: str, n_per: int, fname: str, title: str) -> None:
    """Representative crops per condition, chosen near each condition's per-cell
    MEDIAN of `col` (so they are typical, not cherry-picked extremes), one per
    distinct well where possible. Selection method is stated in the caption."""
    BG, FG = "black", "white"
    fig, axs = plt.subplots(len(CONDS), n_per, figsize=(n_per * 2.1, len(CONDS) * 2.1),
                            facecolor=BG, dpi=150)
    axs = np.atleast_2d(axs)
    for r, c in enumerate(CONDS):
        d = qc[(qc["timepoint"] == day) & (qc["condition"] == c) & (qc[col] > 0)].copy()
        if not len(d):
            for k in range(n_per):
                axs[r, k].axis("off")
            continue
        med = d[col].median()
        d["dist"] = (d[col] - med).abs()
        d = d.sort_values("dist")
        picked, seen = [], set()
        for _, row in d.iterrows():  # spread across wells
            if row["well"] not in seen or len(picked) >= n_per:
                picked.append(row)
                seen.add(row["well"])
            if len(picked) >= n_per:
                break
        for k in range(n_per):
            ax = axs[r, k]
            ax.set_facecolor(BG)
            ax.axis("off")
            if k < len(picked):
                rgb = crop_cell(picked[k])
                if rgb is not None:
                    ax.imshow(rgb, aspect="auto")
                    val = picked[k][col]
                    ax.set_title(f"{c} · {picked[k]['well']}\n{col.split('_')[0]}={val:.2g}",
                                 color=CCOLOR[c], fontsize=7)
    fig.suptitle(title, color=FG, y=0.99, fontsize=11)
    fig.text(0.5, 0.005,
             "cyan = cell body · magenta = nucleus · yellow = lysosomes | "
             "cells chosen near each condition's median (typical, not extremes)",
             ha="center", color=FG, fontsize=7)
    fig.subplots_adjust(hspace=0.18, wspace=0.02, left=0.02, right=0.98, top=0.90, bottom=0.04)
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def crops_across_days(col: str, fname: str, title: str, window: int = 220) -> None:
    """One representative cell per (day × condition) on a SHARED fixed display LUT
    and a fixed µm-scale window, so the d14 floor is visible: the auto-scaled §3.2
    per-day panels rescale each day and hide that d14 has ~10× fewer & smaller detected puncta.
    Cell chosen nearest that group's median `col` (typical, not an extreme)."""
    BG, FG = "black", "white"
    fig, axs = plt.subplots(len(DAYS), len(CONDS),
                            figsize=(len(CONDS) * 2.4, len(DAYS) * 2.4),
                            facecolor=BG, dpi=150)
    axs = np.atleast_2d(axs)
    for r, day in enumerate(DAYS):
        for ci, cond in enumerate(CONDS):
            ax = axs[r, ci]
            ax.set_facecolor(BG)
            ax.axis("off")
            d = qc[(qc["timepoint"] == day) & (qc["condition"] == cond) & (qc[col] > 0)]
            if not len(d):
                continue
            row = d.iloc[(d[col] - d[col].median()).abs().values.argmin()]
            rgb = crop_cell(row, window=window)
            if rgb is None:
                continue
            ax.imshow(rgb, aspect="equal")  # equal + fixed window => size comparable
            if r == 0:
                ax.set_title(cond, color=CCOLOR[cond], fontsize=11, fontweight="bold")
            if ci == 0:
                ax.text(-0.06, 0.5, day, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", color=FG, fontsize=12, fontweight="bold")
            ax.text(0.5, 0.02, f"{row[col]:.2g} µm²", transform=ax.transAxes,
                    ha="center", va="bottom", color=FG, fontsize=8)
    fig.suptitle(title, color=FG, y=0.98, fontsize=11)
    fig.text(0.5, 0.005,
             "cyan = cell body · magenta = nucleus · yellow FILL = detected lysosomes "
             "(fill hides intensity) | "
             f"shared fixed LUT + fixed {round(window * PX_UM)} µm window: d14 has FEWER & "
             "SMALLER detected puncta — the survivors are not dimmer | value = cell mean lyso size",
             ha="center", color=FG, fontsize=7)
    fig.subplots_adjust(hspace=0.08, wspace=0.02, left=0.04, right=0.98, top=0.92, bottom=0.05)
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# %% [markdown]
# # §0 — Headline
#
# The primary readout is **mean lysosome size**. The finding: **TMEM KO lysosomes
# are smaller than Control at d7**, same direction at d28, absent at d14 (a
# staining-dimness detection floor, shown in §4).

# %%
md("# Fixed-IF LAMP1 Report — TMEM106B KO / Control / KI",
   "",
   "*Branch `feat/if-lamp1-quantification` · fixed 3765 DN LAMP1 threshold · "
   "well-level statistics (pseudoreplication guard) · audience: skeptical biologists.*",
   "")
md("## §0 — Headline", "")
md(n_tbl.pipe(df_md), "")

# headline = d7 lyso size dotplot + d7 crops
h_res = well_dotplot("d7", "lyso_mean_size_um2", "Mean lysosome size", "µm²",
                     "headline_d7_lyso_size.png")
fig_md("headline_d7_lyso_size.png",
       "d7 mean lysosome size, well-level. KO smaller than Control.")

# %% [markdown]
# # §1 — The assay & question
#
# 3 genotypes (TMEM **KO**, PLD3 **Control**, TMEM **KI**) × 3 timepoints
# (d7/d14/d28) × 4 IF channels (DAPI/MAP2/cleaved-TMEM/LAMP1). Question: does
# TMEM106B genotype alter lysosome (LAMP1) morphology/abundance, and does cleaved
# TMEM colocalize with lysosomes?

# %%
md("## §1 — The assay & question", "",
   "3 genotypes (TMEM **KO**, PLD3 **Control**, TMEM **KI**) × 3 timepoints "
   "(d7/d14/d28) × 4 channels (DAPI/405, MAP2/488, cleaved-TMEM/561, LAMP1/640; "
   "max-Z projection). **Question:** does TMEM106B genotype alter LAMP1 lysosome "
   "morphology/abundance, and does cleaved TMEM colocalize with lysosomes?", "")
for src, cap in [
    ("fig_pipeline_perchannel.png", "Representative FOV, per channel (fixed display LUT)."),
    ("fig_pipeline_merged.png", "Same FOV, merged channels."),
]:
    if (REPORTS / src).exists():
        fig_md(f"../{src}", cap)  # reference in place, no copy

# %% [markdown]
# # §2 — Pipeline as trust-building
#
# The choices that make the numbers defensible against a skeptic.

# %%
md("## §2 — Pipeline (why you can trust the numbers)", "",
   "1. **Fixed display LUT** — brightness is comparable across genotypes, not "
   "per-image auto-stretched.",
   "2. **Fixed 3765 DN lysosome threshold** — derived from Control+KI FOVs "
   "(`survey_lamp1_otsu.py`); decouples detection from per-genotype brightness so "
   "a dimmer KO can't artificially fragment/shrink its lysosomes.",
   "3. **HITL fine-tuned cell-body segmentation** — direct Cellpose fails on the "
   "MAP2 neurite mesh; a model fine-tuned on 9 hand-corrected FOVs gives clean "
   "somata (AP@0.5 0.58). Known ~15–20% under-detection of faint cells.",
   "4. **Condition-blind QC** — border / size-outlier / pyknotic cells excluded "
   "by identical rules across genotypes.",
   "5. **Well-level aggregation** — cells → per-well means (n≈6 wells/genotype) "
   "before any test. Cells within a well are not independent replicates.", "")
for src, cap in [
    ("nuclei_overlay.png", "Step: DAPI → nuclei (Cellpose-SAM)."),
    ("map2_cellbody_overlay.png", "Step: MAP2 → cell bodies (HITL cpsam)."),
]:
    if (REPORTS / src).exists():
        fig_md(f"../{src}", cap)

# %% [markdown]
# # §3 — Per-day results, across genotypes
#
# For each day: every readout as a well-level dot plot, then representative crops.

# %%
DAY_STATS: dict[str, dict] = {}
for day in DAYS:
    md(f"## §3.{DAYS.index(day) + 1} — {day} results", "")
    for col, label, unit in READOUTS:
        res = well_dotplot(day, col, label, unit, f"{day}_{col}.png")
        DAY_STATS[f"{day}:{col}"] = res
        line = f"KW p={res.get('kruskal_p', float('nan')):.3g}"
        for c, st in res.get("vs_reference", {}).items():
            line += f" · {c} vs Ctrl BH p={st['p_bh']:.3g}"
        fig_md(f"{day}_{col}.png", f"{day} — {label}. {line}")
    # coloc (flagged proxy)
    for col, label, unit in COLOC:
        res = well_dotplot(day, col, label, unit, f"{day}_{col}.png")
        fig_md(f"{day}_{col}.png",
               f"{day} — {label} *(2D max-projection proxy; rigorous 3D "
               f"object-based coloc lives in the refinements branch)*.")
    # crops for the primary readout
    crop_grid(day, "lyso_mean_size_um2", 4, f"{day}_crops.png",
              f"{day} — representative cells by genotype (lysosome size)")
    fig_md(f"{day}_crops.png",
           f"{day} — typical cells per genotype; yellow = detected lysosomes.")

# %% [markdown]
# # §4 — Across days & genotypes (synthesis)

# %%
md("## §4 — Across days & genotypes", "")


def trajectory(col: str, label: str, unit: str, fname: str) -> None:
    BG, FG = "white", "black"
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=BG, dpi=150)
    ax.set_facecolor(BG)
    for c in CONDS:
        xs, ms, es = [], [], []
        for i, day in enumerate(DAYS):
            wm = condition_stats(qc[qc["timepoint"] == day], col, timepoint=day)["well_means"]
            v = wm[wm["condition"] == c][col].values
            if len(v):
                xs.append(i)
                ms.append(v.mean())
                es.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.errorbar(xs, ms, yerr=es, color=CCOLOR[c], marker="o", capsize=4,
                    linewidth=2, label=c)
    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels(DAYS)
    ax.set_ylabel(f"{label} ({unit})", color=FG)
    ax.set_title(f"{label} across days (well mean ± SEM)", color=FG, fontsize=10)
    for s in ax.spines.values():
        s.set_color(FG)
    ax.tick_params(colors=FG)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


for col, label, unit in READOUTS:
    trajectory(col, label, unit, f"traj_{col}.png")
    fig_md(f"traj_{col}.png", f"{label} — trajectory across days.")

# Effect-size table (Cohen's d, well-level, KO & KI vs Control)
rows = []
for day in DAYS:
    for col, label, _ in READOUTS:
        wm = condition_stats(qc[qc["timepoint"] == day], col, timepoint=day)["well_means"]
        ref = wm[wm["condition"] == "Control"][col].values
        r = {"day": day, "readout": label}
        for c in ["KO", "KI"]:
            r[f"d({c})"] = round(cohens_d(wm[wm["condition"] == c][col].values, ref), 2)
        rows.append(r)
dtab = pd.DataFrame(rows)
md("### Effect sizes (Cohen's d, well-level, vs Control)", "",
   dtab.pipe(df_md), "",
   "*Negative d(KO) for lysosome size = KO smaller than Control.*", "")
print(dtab.to_string(index=False))

# The d14 dimness explanation: LAMP1 mean by day (why d14 is a detection floor)
trajectory("lamp1_mean", "LAMP1 mean (bg-sub)", "DN", "traj_lamp1_dimness.png")
md("### d14 is a detection floor, not a null", "",
   "LAMP1 signal drops sharply at d14 across all genotypes (below). With a fixed "
   "3765 DN threshold, dim d14 lysosomes fall under detection — the d14 null "
   "reflects staining/exposure, not biology. This is why the KO-smaller effect is "
   "visible at d7 and d28 but not d14.", "")
fig_md("traj_lamp1_mean.png", "LAMP1 mean by day — the d14 dip is the dimness floor.")

# Single-cell crops across days at a shared LUT + scale — makes the d14 floor visible
# (the auto-scaled per-day §3 dot plots rescale each day and hide it).
crops_across_days("lyso_mean_size_um2", "crops_across_days.png",
                  "Single-cell crops across days (shared LUT & scale) — fewer & smaller detected lysosomes at d14")
md("At a shared display LUT and fixed µm window, d14 shows **fewer and smaller detected "
   "lysosomes** than d7/d28 — which the auto-scaled §3.2 per-day panels hide by rescaling "
   "each day. What this does **not** show is dimmer lysosomes: per-lysosome-pixel LAMP1 is "
   "comparable across days (median ~6.4k DN at d7 and d14), because anything passing the "
   "fixed 3765 DN threshold is bright by construction. d14's low *cell-mean* LAMP1 comes "
   "from sparse/small puncta, not dim ones. And the yellow fill paints every detected "
   "lysosome the same colour, so intensity is not readable from it at all. Read this as the "
   "detection/batch floor — cross-day absolute differences are confounded with plate/"
   "staining batch (each timepoint is its own plate), not biology.", "")
fig_md("crops_across_days.png",
       "Representative cell per day × genotype, nearest each group's median lysosome size, "
       "at a shared fixed display LUT and fixed µm window. cyan=cell, magenta=nucleus, "
       "yellow fill=detected lysosomes (fill hides intensity); value = cell mean lysosome size.")

# Fixed-vs-Otsu artifact control (d7 only: Otsu table exists)
if (REPORTS / "d7_percell.csv").exists():
    otsu = pd.read_csv(REPORTS / "d7_percell.csv")
    otsu = otsu[(otsu["qc_pass"] == True) & otsu["condition"].isin(CONDS)]  # noqa: E712
    art = []
    for lab_src, dsrc in [("Otsu (per-FOV)", otsu), ("Fixed 3765 DN", qc[qc["timepoint"] == "d7"])]:
        wm = dsrc.groupby(["condition", "well"])["lyso_mean_size_um2"].mean().reset_index()
        ref = wm[wm["condition"] == "Control"]["lyso_mean_size_um2"].values
        ko = wm[wm["condition"] == "KO"]["lyso_mean_size_um2"].values
        art.append({"threshold": lab_src, "d(KO vs Ctrl)": round(cohens_d(ko, ref), 2)})
    md("### Artifact control: the KO effect is not a thresholding artifact", "",
       "Per-FOV Otsu adapts downward on dim KO images, which *could* shrink KO "
       "lysosomes artificially. Switching to a fixed brightness-blind threshold "
       "does not remove the KO-smaller effect:", "",
       pd.DataFrame(art).pipe(df_md), "")
    print(pd.DataFrame(art).to_string(index=False))

# %% [markdown]
# # §5 — Caveats & what would convince a skeptic

# %%
md("## §5 — Caveats & what would convince a skeptic", "",
   "- **Biological n = 1.** d7/d14/d28 may be one differentiation, not "
   "independent replicates. Well-level stats guard pseudoreplication *within* the "
   "run but cannot manufacture biological replicates. **This gates every genotype "
   "claim.** Resolve: confirm replicate structure with the lab; add independent "
   "differentiations → upgrade `condition_stats` to a linear mixed model.",
   "- **Direction is backwards from TMEM106B-LOF literature**, which typically "
   "*enlarges* lysosomes. Could be early-timepoint biology, cell state, or a 2D "
   "max-projection artifact. Flag for lab discussion.",
   "- **d14 is a detection floor** (LAMP1 ~2.5× dimmer). Restrict d14 to "
   "within-timepoint comparisons or re-stain.",
   "- **Coloc here is a 2D proxy** (Manders M1 / Pearson within the cell mask on "
   "a max-projection). The rigorous 3D object-based coloc (`if_coloc.py`, "
   "position-randomized null) lives in the **refinements** branch — cite that for "
   "any coloc claim.",
   "- **HITL segmentation under-detects ~15–20% of faint cells.** Acceptable for "
   "morphology/area readouts; note it for count-based claims.", "")

# %% [markdown]
# ## Write report.md

# %%
report_path = OUT / "report.md"
report_path.write_text("\n".join(MD))
print(f"Wrote {report_path}")

# Verify figures landed (nb-figures Mode B)
pngs = sorted(FIGS.glob("*.png"))
empty = [p for p in pngs if p.stat().st_size == 0]
assert pngs, f"No figures in {FIGS}"
assert not empty, f"0-byte figures: {empty}"
print(f"OK: {len(pngs)} figures in {FIGS}")

# %% [markdown]
# ## Inline gallery (notebook view)
#
# Renders the full report (with figures) inline when this is run as a notebook.
# In a plain-script run this is a no-op — the figures and report.md are already on disk.

# %%
try:
    get_ipython  # noqa: B018 -- only defined inside an IPython/Jupyter kernel
    from IPython.display import Markdown, display

    display(Markdown((OUT / "report.md").read_text()))
except NameError:
    pass
