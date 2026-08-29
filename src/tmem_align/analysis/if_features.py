"""Per-cell fixed-IF feature extraction + condition/timepoint stats.

Sits on top of if_spatial (load_fov + segment_nuclei + segment_cell_bodies):
one ND2 FOV -> per-cell rows for the LAMP1 organelle readout, QC-filtered, then
aggregated to the tidy table all stats/plots read from.

Experimental design (cleaved-TMEM d7/d14/d28 dataset):
  condition (KO/Control/KI) -> well (6, rows C-H one column) -> FOV (4, F1-F4) -> cells
The experimental UNIT is the well, not the cell. Cells within a well/FOV are
correlated; testing pooled cells as n=cells is pseudoreplication. condition_stats
aggregates to well values first. Also: this is ONE pooled differentiation
(wells = technical reps, d7->d28 = same source aged), so biological n = 1 --
these stats describe THIS culture; a genotype claim needs independent diffs.

Design confounds to state in methods, not fixable in code here:
  - time is confounded with plate (one plate per timepoint) -> needs >=3 runs.
  - condition is confounded with plate COLUMN (each genotype = a fixed column) ->
    column/edge/evaporation effects track genotype; randomize on future plates.

Four readouts (primary = lysosome size & count; rest exploratory, BH-correct):
  1. size & count   n_lysosomes, lyso_area_frac, lyso_mean_size_um2
  2. intensity      lamp1_mean, lamp1_integrated  (bg-subtracted)
  3. spatial        perinuclear_index (mean lyso dist from nucleus / cell radius)
  4. coloc w/ TMEM  tmem_manders_m1, tmem_pearson

MIP caveat: load_fov max-projects Z. That is NOT invariant to slice count -- the
noise floor (max of N samples) creeps up with depth, and z-overlapping lysosomes
collapse. Run depth_sensitivity() on controls before trusting 2D; escalate to 3D
(Cellpose anisotropy=z_step/xy) only if the readout correlates with slice count.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .if_spatial import CH_DAPI, CH_LAMP1, CH_MAP2, CH_TMEM

# Immediate-parent dir name -> condition label. Fallback = raw dir name (stays total).
CONDITION_DIRS = {
    "TMEM_KO": "KO",
    "Z59_PLD_Control": "Control",
    "Z60_PLD_TMEMki": "KI",
}
PX_UM = 0.108  # µm per pixel (matches if_spatial nucleus-diameter note)


def parse_fov_metadata(nd2_path: str | Path) -> dict:
    """(condition, well, fov, timepoint, plate) from path + filename.

    Filename e.g. PLD3Control_Plate1_d7_..._C17_F4.nd2 ; condition from parent dir.
    """
    p = Path(nd2_path)
    name = p.name
    well = re.search(r"_([A-P]\d{1,2})_F\d", name)
    fov = re.search(r"_F(\d+)\.nd2$", name)
    day = re.search(r"_d(\d+)_", name)
    plate = re.search(r"_(Plate\d+)_", name)
    return {
        "condition": CONDITION_DIRS.get(p.parent.name, p.parent.name),
        "well": well.group(1) if well else None,
        "fov": int(fov.group(1)) if fov else None,
        "timepoint": f"d{day.group(1)}" if day else None,
        "plate": plate.group(1) if plate else None,
        "file_path": str(p),
    }


def fov_focus_score(img_yx: np.ndarray) -> float:
    """Variance-of-Laplacian focus metric. Higher = sharper. Drop out-of-focus FOVs."""
    from scipy import ndimage as ndi

    img = np.asarray(img_yx, dtype=np.float32)
    return float(ndi.laplace(img).var())


def fov_z_count(nd2_path: str | Path) -> int:
    """Number of Z slices in an ND2 FOV (1 if not a stack). Feeds depth_sensitivity."""
    import nd2

    with nd2.ND2File(Path(nd2_path)) as f:
        return int(f.sizes.get("Z", 1))


def detect_lysosomes(
    lamp1_yx: np.ndarray,
    bg_percentile: float = 50.0,
    threshold: float | None = None,
    min_size_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Background-subtract LAMP1, threshold, label puncta over the whole FOV.

    Returns (labels, corrected). One global FOV threshold (Otsu by default) so puncta
    are detected the SAME way in every cell -- per-cell Otsu is unstable for cells with
    few lysosomes. Pass a fixed DN threshold for cross-plate comparability if Otsu drifts.
    # ponytail: Otsu default; swap to fixed DN (LUT floor) if the threshold wanders
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    img = np.asarray(lamp1_yx, dtype=np.float32)
    corrected = np.clip(img - np.percentile(img, bg_percentile), 0, None)
    thr = threshold_otsu(corrected) if threshold is None else threshold
    labels = label(corrected > thr).astype(np.int32)
    # drop specks below min_size_px (explicit, avoids the deprecated remove_small_objects arg)
    counts = np.bincount(labels.ravel())
    labels[np.isin(labels, np.nonzero(counts < min_size_px)[0])] = 0
    return labels, corrected


def per_cell_features(
    cell_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    channels: dict[str, np.ndarray],
    lyso_labels: np.ndarray,
    lamp1_corrected: np.ndarray,
    tmem_threshold: float | None = None,
    px_um: float = PX_UM,
) -> pd.DataFrame:
    """One row per cell-body label with the four LAMP1 readout families."""
    from skimage.filters import threshold_otsu

    tmem = np.asarray(channels[CH_TMEM], dtype=np.float32)
    if tmem_threshold is None:
        tmem_threshold = threshold_otsu(tmem)

    rows = []
    labels = np.unique(cell_masks)
    labels = labels[labels != 0]
    for lab in labels:
        cell = cell_masks == lab
        area = int(cell.sum())
        if area == 0:
            continue

        # --- 2. intensity ---
        lamp1_corr_cell = lamp1_corrected[cell]
        lamp1_mean = float(lamp1_corr_cell.mean())
        lamp1_integrated = float(lamp1_corr_cell.sum())

        # --- 1. size & count ---
        lyso_in_cell = lyso_labels[cell]
        lyso_ids = np.unique(lyso_in_cell)
        lyso_ids = lyso_ids[lyso_ids != 0]
        lyso_px = int((lyso_in_cell != 0).sum())
        n_lyso = int(lyso_ids.size)
        lyso_area_frac = lyso_px / area
        lyso_mean_size_um2 = (lyso_px / n_lyso) * px_um**2 if n_lyso else 0.0

        # --- 3. spatial: lysosome distance from this cell's nucleus centroid ---
        nuc = nuclei_masks == lab
        perinuclear_index = np.nan
        if nuc.any() and lyso_px:
            ny, nx = np.argwhere(nuc).mean(axis=0)
            ly, lx = np.nonzero(cell & (lyso_labels != 0))
            dist = np.hypot(ly - ny, lx - nx)
            cell_radius = np.sqrt(area / np.pi)  # equivalent-circle radius
            perinuclear_index = float(dist.mean() / cell_radius)  # <1 = perinuclear

        # --- 4. coloc with TMEM (within cell mask) ---
        tmem_cell = tmem[cell]
        tmem_pos = tmem_cell > tmem_threshold
        denom = lamp1_corr_cell.sum()
        manders_m1 = float(lamp1_corr_cell[tmem_pos].sum() / denom) if denom else 0.0
        pearson = (
            float(np.corrcoef(lamp1_corr_cell, tmem_cell)[0, 1])
            if lamp1_corr_cell.std() and tmem_cell.std()
            else np.nan
        )

        rows.append(
            {
                "cell_id": int(lab),
                "cell_area_px": area,
                "n_lysosomes": n_lyso,
                "lyso_area_frac": lyso_area_frac,
                "lyso_mean_size_um2": lyso_mean_size_um2,
                "lamp1_mean": lamp1_mean,
                "lamp1_integrated": lamp1_integrated,
                "perinuclear_index": perinuclear_index,
                "tmem_manders_m1": manders_m1,
                "tmem_pearson": pearson,
            }
        )
    return pd.DataFrame(rows)


def qc_filter_cells(
    cell_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    dapi_yx: np.ndarray,
    min_cell_area_px: int = 2000,
    max_cell_area_px: int = 80000,
    min_nuc_area_px: int = 3000,
) -> pd.DataFrame:
    """Per-cell-label QC: qc_pass=False for border / abnormal-nucleus / size-outlier cells.

    Uses ONLY nuclear + cytoskeletal geometry -- never the LAMP1 organelle channel,
    which is the outcome being tested. Defaults from ~12 µm soma / ~0.108 µm-px:
    nucleus ~9700 px²; abnormal (small + condensed DAPI) < ~3000 px²; soma 2k-80k px².

    Thresholds are fixed constants (condition-blind by construction). "abnormal_nucleus"
    means morphologically abnormal, NOT confirmed dead -- morphology alone cannot call
    death without a viability marker. Report qc_summary() per condition and run
    condition_stats(apply_qc=False) as a check: if excluding these cells changes the
    result, QC may be censoring the phenotype (e.g. KO genuinely injuring cells).
    Returns cols: cell_id, qc_pass, qc_reason.
    """
    dapi = np.asarray(dapi_yx, dtype=np.float32)
    h, w = cell_masks.shape
    dapi_hi = np.percentile(dapi, 99)  # condensed-chromatin brightness gate

    rows = []
    labels = np.unique(cell_masks)
    labels = labels[labels != 0]
    for lab in labels:
        cell = cell_masks == lab
        ys, xs = np.nonzero(cell)
        area = cell.sum()
        nuc = nuclei_masks == lab
        nuc_area = int(nuc.sum())
        nuc_dapi = float(dapi[nuc].mean()) if nuc_area else 0.0

        reason = ""
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            reason = "border"  # truncated by frame edge
        elif area < min_cell_area_px or area > max_cell_area_px:
            reason = "size_outlier"
        elif nuc_area and nuc_area < min_nuc_area_px and nuc_dapi > dapi_hi:
            reason = "abnormal_nucleus"  # small + condensed; NOT confirmed dead
        rows.append({"cell_id": int(lab), "qc_pass": reason == "", "qc_reason": reason})
    return pd.DataFrame(rows)


def analyze_fov(nd2_path: str | Path, min_focus: float | None = None) -> pd.DataFrame:
    """Full per-FOV pipeline: load -> segment -> QC -> features, tagged with metadata.

    Returns one row per cell (QC cells kept, flagged by qc_pass) with condition/well/
    fov/timepoint/n_z_slices columns. Requires nd2 + cellpose (see if_spatial).
    """
    from .if_spatial import load_fov, segment_cell_bodies, segment_nuclei

    meta = parse_fov_metadata(nd2_path)
    meta["n_z_slices"] = fov_z_count(nd2_path)  # for depth_sensitivity (MIP bias check)
    ch = load_fov(nd2_path)

    if min_focus is not None and fov_focus_score(ch[CH_MAP2]) < min_focus:
        return pd.DataFrame()  # out-of-focus FOV dropped whole

    nuclei = segment_nuclei(ch[CH_DAPI])
    cells = segment_cell_bodies(ch[CH_MAP2])
    lyso_labels, lamp1_corr = detect_lysosomes(ch[CH_LAMP1])

    feats = per_cell_features(cells, nuclei, ch, lyso_labels, lamp1_corr)
    qc = qc_filter_cells(cells, nuclei, ch[CH_DAPI])
    df = feats.merge(qc, on="cell_id", how="left")
    for k, v in meta.items():
        df[k] = v
    return df


def build_table(nd2_paths: list[str | Path], min_focus: float | None = None) -> pd.DataFrame:
    """Concatenate analyze_fov over many FOVs into the one tidy table.

    Glob the paths yourself, e.g.
        paths = sorted(Path(data/'d7').rglob('*.nd2'))
        df = build_table(paths)
    """
    frames = []
    for p in nd2_paths:
        try:
            frames.append(analyze_fov(p, min_focus=min_focus))
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            print(f"  SKIP {Path(p).name}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition QC exclusion rate + reason breakdown. Report this with any result.

    Uneven exclusion across conditions is the red flag -- QC may be censoring biology
    (e.g. KO cells failing morphology QC). Pair with condition_stats(apply_qc=False).
    """
    summary = df.groupby("condition")["qc_pass"].agg(n_cells="size", n_pass="sum")
    summary["pct_excluded"] = 100 * (1 - summary["n_pass"] / summary["n_cells"])
    reasons = (
        df[~df["qc_pass"]].groupby(["condition", "qc_reason"]).size().unstack(fill_value=0)
    )
    return summary.join(reasons).reset_index()


def depth_sensitivity(
    df: pd.DataFrame, value_col: str, condition: str = "Control", unit: str = "well"
) -> dict:
    """Spearman corr of the FOV-level readout vs Z-slice count, within one condition.

    MIP intensity/counts should NOT track stack depth. A non-flat correlation here means
    max-projection is biasing the readout by slice count -> switch that readout to 3D.
    Needs the n_z_slices column (present when built via analyze_fov).
    """
    from scipy import stats

    d = df[df["condition"] == condition]
    if "qc_pass" in d.columns:
        d = d[d["qc_pass"] == True]  # noqa: E712
    fov = (
        d.groupby([unit, "fov"])
        .agg(val=(value_col, "median"), n_z=("n_z_slices", "first"))
        .dropna()
    )
    if fov["n_z"].nunique() < 2 or len(fov) < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(len(fov)),
                "note": "insufficient z-depth variation to test"}
    r = stats.spearmanr(fov["n_z"], fov["val"])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(fov))}


def _welch_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean_diff, ci_lo, ci_hi) for a-b via Welch (unequal-variance) t interval."""
    from scipy import stats

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a.mean() - b.mean()
    if len(a) < 2 or len(b) < 2:  # need >=2 wells/side for a CI
        return float(diff), float("nan"), float("nan")
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float(diff), float(diff), float(diff)
    dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t = stats.t.ppf(1 - alpha / 2, dfree)
    return float(diff), float(diff - t * se), float(diff + t * se)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else float("nan")


def condition_stats(
    df: pd.DataFrame,
    value_col: str,
    reference: str = "Control",
    unit: str = "well",
    timepoint: str | None = None,
    apply_qc: bool = True,
) -> dict:
    """Well-level stats for one readout: aggregate to wells, then compare conditions.

    THIS is the pseudoreplication guard. Aggregation is cell -> per-FOV median ->
    equal-weight mean across FOVs -> one well value, so a dense FOV can't dominate the
    well (falls back to a direct well mean if there is no `fov` column). Reports
    Kruskal-Wallis across conditions plus, per condition vs `reference`: Mann-Whitney p
    (BH-corrected), mean difference with a 95% Welch CI, and Cohen's d -- effect size +
    uncertainty, not a p-value alone (n=6 has low power and no CI).

    apply_qc=False keeps ALL cells: run it both ways -- if excluding QC-failing cells
    changes the answer, QC may be censoring the phenotype (e.g. KO injures cells).
    Do NOT normalize `value_col` to the cytoskeletal channel unless that channel is
    shown unaffected by genotype. Cell size is a sensible adjustment covariate (bigger
    cells hold more total signal) -- regress it out upstream if using total intensity.
    # ponytail: well-means + Welch CI; upgrade to an LMM (statsmodels, cell rows,
    # 1|well/fov) once independent differentiations exist.
    """
    from scipy import stats

    d = df
    if apply_qc and "qc_pass" in df.columns:
        d = d[d["qc_pass"] == True]  # noqa: E712
    if timepoint is not None:
        d = d[d["timepoint"] == timepoint]

    # cell -> per-FOV median -> equal-weight mean across FOVs -> well value
    if "fov" in d.columns:
        fov_val = d.groupby(["condition", unit, "fov"])[value_col].median().reset_index()
        well_means = fov_val.groupby(["condition", unit])[value_col].mean().reset_index()
    else:
        well_means = d.groupby(["condition", unit])[value_col].mean().reset_index()
    groups = {c: g[value_col].values for c, g in well_means.groupby("condition")}

    out = {
        "value_col": value_col,
        "well_means": well_means,
        "n_per_condition": {c: len(v) for c, v in groups.items()},
    }
    if len(groups) >= 2:
        out["kruskal_p"] = float(stats.kruskal(*groups.values()).pvalue)

    ref = groups.get(reference)
    pairwise, pvals = [], []
    for cond, vals in groups.items():
        if cond == reference or ref is None:
            continue
        p = float(stats.mannwhitneyu(vals, ref, alternative="two-sided").pvalue)
        diff, lo, hi = _welch_ci(vals, ref)
        pairwise.append((cond, p, diff, lo, hi, _cohens_d(vals, ref)))
        pvals.append(p)
    if pvals:
        from scipy.stats import false_discovery_control

        adj = false_discovery_control(pvals)
        out["vs_reference"] = {
            cond: {
                "mean_diff": diff,
                "ci95": (lo, hi),
                "cohens_d": d_,
                "p_raw": p,
                "p_bh": float(a),
            }
            for (cond, p, diff, lo, hi, d_), a in zip(pairwise, adj)
        }
    return out


def _demo() -> None:
    """Synthetic self-check: no ND2/cellpose. Exercises detect/features/QC/stats."""
    rng = np.random.default_rng(0)
    H = W = 256
    cells = np.zeros((H, W), np.int32)
    nuclei = np.zeros((H, W), np.int32)
    lamp1 = rng.normal(100, 5, (H, W)).astype(np.float32)
    tmem = rng.normal(100, 5, (H, W)).astype(np.float32)
    dapi = rng.normal(100, 5, (H, W)).astype(np.float32)

    # cell 1: interior, cell 2: touches border (should fail QC)
    cells[40:120, 40:120] = 1
    nuclei[70:100, 70:100] = 1
    dapi[70:100, 70:100] = 300
    cells[150:230, 0:80] = 2  # xs.min()==0 -> border
    nuclei[180:210, 20:50] = 2
    dapi[180:210, 20:50] = 300
    # bright LAMP1 puncta inside each cell
    for (cy, cx) in [(60, 60), (65, 90), (95, 55), (170, 30), (200, 60)]:
        lamp1[cy - 2 : cy + 2, cx - 2 : cx + 2] = 3000
    tmem[63:67, 88:92] = 3000  # one punctum overlaps TMEM

    lyso_labels, lamp1_corr = detect_lysosomes(lamp1)
    assert lyso_labels.max() >= 4, "should detect the bright puncta"

    ch = {CH_LAMP1: lamp1, CH_TMEM: tmem, CH_DAPI: dapi, CH_MAP2: lamp1}
    feats = per_cell_features(cells, nuclei, ch, lyso_labels, lamp1_corr)
    assert set(feats["cell_id"]) == {1, 2}
    assert (feats["n_lysosomes"] > 0).all(), "each cell has puncta"
    assert feats.set_index("cell_id").loc[1, "tmem_manders_m1"] > 0, "coloc detected"

    qc = qc_filter_cells(cells, nuclei, dapi)
    qc_i = qc.set_index("cell_id")
    assert qc_i.loc[2, "qc_reason"] == "border" and not qc_i.loc[2, "qc_pass"]
    assert qc_i.loc[1, "qc_pass"], "interior cell should pass"

    # stats: fake 2 conditions x 6 wells x 4 FOVs, KO shifted up -> should separate
    rows = []
    for cond, mu in [("Control", 10.0), ("KO", 20.0)]:
        for wi in range(6):
            for fv in range(4):
                for _ in range(30):  # 30 cells/FOV -> must NOT inflate n
                    rows.append({"condition": cond, "well": f"{cond}{wi}", "fov": fv,
                                 "n_lysosomes": rng.normal(mu, 3), "qc_pass": True,
                                 "qc_reason": "", "n_z_slices": 20 + fv})
    sdf = pd.DataFrame(rows)
    res = condition_stats(sdf, "n_lysosomes")
    assert res["n_per_condition"] == {"Control": 6, "KO": 6}, "n = wells, not cells"
    ko = res["vs_reference"]["KO"]
    assert ko["p_bh"] < 0.05, "clear shift should be significant"
    assert ko["mean_diff"] > 5 and ko["ci95"][0] > 0, "CI should exclude 0 for a real shift"
    assert abs(ko["cohens_d"]) > 0.8, "large effect expected"

    # aggregation must NOT inflate n even with an uneven, dense FOV
    assert len(condition_stats(sdf, "n_lysosomes")["well_means"]) == 12

    qs = qc_summary(sdf)
    assert (qs["pct_excluded"] == 0).all(), "no exclusions in synthetic set"
    dep = depth_sensitivity(sdf, "n_lysosomes")
    assert "rho" in dep and dep["n"] > 0, "depth check should run"

    print(
        "if_features self-check passed:", res["n_per_condition"], "wells/condition;",
        f"KO diff={ko['mean_diff']:.1f} "
        f"CI[{ko['ci95'][0]:.1f},{ko['ci95'][1]:.1f}] d={ko['cohens_d']:.1f}",
    )


if __name__ == "__main__":
    _demo()
