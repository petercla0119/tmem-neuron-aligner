"""Tier-1 validation of the KO-smaller-lysosome finding.

Check #1: is the KO lyso-size deficit confounded by KO cells having smaller
          cell-body masks (which would clip large peripheral lysosomes)?
Check #3: does the d28 KO effect survive dropping the QC filter (censoring risk)?

Run from the refinements worktree:
    PYTHONPATH=src python notebooks/tier1_validation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tmem_align.analysis.if_features import condition_stats

REP = Path("reports/if_segmentation_pilot")
VAL = "lyso_mean_size_um2"
PX_UM2 = 0.108**2  # µm² per pixel


def fov_then_well(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Canonical aggregation: cell -> per-FOV median -> one row per well."""
    fov = df.groupby(["condition", "well", "fov"])[col].median().reset_index()
    return fov.groupby(["condition", "well"])[col].mean().reset_index()


def check1_cellarea_confound() -> None:
    print("=" * 70)
    print("CHECK #1 — cell-body area confound (d7, qc-pass)")
    print("=" * 70)
    df = pd.read_csv(REP / "d7_percell_full.csv")
    d = df[df["qc_pass"]].copy()
    d["cell_area_um2"] = d["cell_area_px"] * PX_UM2

    print("\n(a) cell-body size by condition (qc-pass cells)")
    g = d.groupby("condition")["cell_area_um2"].agg(["count", "median", "mean"])
    print(g.round(1))
    # Is KO smaller? Mann-Whitney on cell-level (descriptive) + well-level.
    for c in ["KO", "KI"]:
        a = d[d.condition == c]["cell_area_um2"]
        b = d[d.condition == "Control"]["cell_area_um2"]
        u, p = stats.mannwhitneyu(a, b)
        print(f"    {c} vs Control cell area: median {a.median():.0f} vs "
              f"{b.median():.0f} µm²  (MW p={p:.3g})")

    print("\n(b) does lyso size track cell area? (Spearman)")
    rho, p = stats.spearmanr(d["cell_area_um2"], d[VAL])
    print(f"    pooled: rho={rho:+.3f}, p={p:.3g}")
    for c in ["Control", "KI", "KO"]:
        s = d[d.condition == c]
        rho, p = stats.spearmanr(s["cell_area_um2"], s[VAL])
        print(f"    {c:8s}: rho={rho:+.3f}, p={p:.3g}")

    print("\n(c) well-level (n=18): does KO deficit survive adjusting for cell area?")
    wl = fov_then_well(d, VAL).rename(columns={VAL: "lyso"})
    wa = fov_then_well(d, "cell_area_um2").rename(columns={"cell_area_um2": "area"})
    w = wl.merge(wa, on=["condition", "well"])
    print(w.groupby("condition")[["lyso", "area"]].mean().round(3))

    ko = w[w.condition == "KO"]["lyso"]
    ct = w[w.condition == "Control"]["lyso"]
    raw = ko.mean() - ct.mean()
    print(f"\n    RAW well-level KO-Ctrl lyso diff = {raw:+.3f} µm²")

    # ANCOVA-style: regress lyso on area + KO dummy (KO vs Control wells only).
    sub = w[w.condition.isin(["KO", "Control"])].copy()
    sub["ko"] = (sub.condition == "KO").astype(float)
    X = np.column_stack([np.ones(len(sub)), sub["ko"], sub["area"]])
    y = sub["lyso"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(sub) - X.shape[1]
    mse = (resid @ resid) / dof
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    print(f"    AREA-ADJUSTED KO effect (OLS lyso ~ 1 + KO + area):")
    print(f"        KO coef   = {beta[1]:+.3f} µm²  (SE {se[1]:.3f}, p={pval[1]:.3g})")
    print(f"        area coef = {beta[2]:+.4f} µm²/µm² (SE {se[2]:.4f}, p={pval[2]:.3g})")
    print(f"    -> if KO coef stays ~{raw:+.2f} and significant, the deficit is NOT "
          "a cell-area/clipping artifact.")


def check3_d28_qc_sensitivity() -> None:
    print("\n" + "=" * 70)
    print("CHECK #3 — d28 QC-sensitivity (censoring risk)")
    print("=" * 70)
    df = pd.read_csv(REP / "all_percell_full.csv")
    d28 = df[df.timepoint == "d28"].copy()
    print(f"d28 cells: {len(d28)} total, {int(d28.qc_pass.sum())} pass "
          f"({100*d28.qc_pass.mean():.1f}%)")
    for label, apply_qc in [("WITH QC (default)", True), ("WITHOUT QC", False)]:
        r = condition_stats(d28, VAL, apply_qc=apply_qc)
        ko = r["vs_reference"]["KO"]
        lo, hi = ko["ci95"]
        n = r["n_per_condition"]
        print(f"\n  {label}:  n_wells Ctrl={n.get('Control','?')} KO={n.get('KO','?')}  "
              f"kruskal_p={r['kruskal_p']:.3g}")
        print(f"    KO-Ctrl {VAL}: diff={ko['mean_diff']:+.3f} µm² "
              f"CI[{lo:+.3f},{hi:+.3f}] "
              f"d={ko['cohens_d']:+.2f} p_bh={ko['p_bh']:.3g}")


if __name__ == "__main__":
    check1_cellarea_confound()
    check3_d28_qc_sensitivity()
