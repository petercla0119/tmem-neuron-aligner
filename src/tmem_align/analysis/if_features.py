"""Per-cell fixed-IF feature extraction + condition/timepoint stats.

Sits on top of if_spatial (load_fov + segment_nuclei + segment_cell_bodies):
one ND2 FOV -> per-cell rows for the LAMP1 organelle readout, QC-filtered, then
aggregated to the tidy table all stats/plots read from.

Experimental design (cleaved-TMEM d7/d14/d28 dataset):
  condition (KO/Control/KI) -> well (6, rows C-H one column) -> FOV (4, F1-F4) -> cells
The experimental UNIT is the well, not the cell. Cells within a well/FOV are
correlated; testing pooled cells as n=cells is pseudoreplication. condition_stats
aggregates to well means first. Also: this is ONE pooled differentiation
(wells = technical reps, d7->d28 = same source aged), so biological n = 1 --
these stats describe THIS culture; a genotype claim needs independent diffs.

Four readouts (primary = lysosome size & count; rest exploratory, BH-correct):
  1. size & count   n_lysosomes, lyso_area_frac, lyso_mean_size_um2
  2. intensity      lamp1_mean, lamp1_integrated  (bg-subtracted)
  3. spatial        perinuclear_index (mean lyso dist from nucleus / cell radius)
  4. coloc w/ TMEM  tmem_manders_m1, tmem_pearson
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


def detect_lysosomes(
    lamp1_yx: np.ndarray,
    bg_percentile: float = 50.0,
    threshold: float | None = None,
    min_size_px: int = 3,
    bg_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Background-subtract LAMP1, threshold, label puncta over the whole FOV.

    Returns (labels, corrected). One global FOV threshold (Otsu by default) so puncta
    are detected the SAME way in every cell -- per-cell Otsu is unstable for cells with
    few lysosomes. Pass a fixed DN threshold for cross-plate comparability if Otsu drifts.

    bg_floor: if given, subtract this fixed DN instead of the per-FOV bg_percentile.
      Eliminates brightness-adaptivity from background estimation. Use LAMP1 LUT floor
      (120 DN) to fully decouple detection from per-condition intensity differences.
      When combined with a fixed `threshold`, detection is completely condition-blind.
    # ponytail: Otsu default; swap to fixed DN (LUT floor) if the threshold wanders
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    img = np.asarray(lamp1_yx, dtype=np.float32)
    if bg_floor is not None:
        corrected = np.clip(img - bg_floor, 0, None)
    else:
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
    """Per-cell-label QC: qc_pass=False for border / pyknotic / size-outlier cells.

    Defaults from the ~12 µm soma / ~0.108 µm-px geometry: nucleus ~9700 px²,
    pyknotic (condensed apoptotic) < ~3000 px² with bright DAPI; soma body 2k-80k px².
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
            reason = "pyknotic"  # small + condensed = dead/apoptotic
        rows.append({"cell_id": int(lab), "qc_pass": reason == "", "qc_reason": reason})
    return pd.DataFrame(rows)


def analyze_fov(
    nd2_path: str | Path,
    min_focus: float | None = None,
    lyso_threshold: float | None = None,
    bg_floor: float | None = None,
) -> pd.DataFrame:
    """Full per-FOV pipeline: load -> segment -> QC -> features, tagged with metadata.

    Returns one row per cell (QC cells kept, flagged by qc_pass) with condition/well/
    fov/timepoint columns prepended. Requires nd2 + cellpose (see if_spatial).

    lyso_threshold: fixed DN threshold for detect_lysosomes (post-bg-subtraction).
      None = per-FOV Otsu (default, backward compatible). Pass a value derived from
      Control/KI FOVs to decouple detection from per-condition brightness.
    bg_floor: fixed DN to subtract as background instead of per-FOV bg_percentile.
      None = per-FOV p50 (default). Use LAMP1 LUT floor (120) for full decoupling.
    """
    from .if_spatial import load_fov, segment_cell_bodies, segment_nuclei

    meta = parse_fov_metadata(nd2_path)
    ch = load_fov(nd2_path)

    if min_focus is not None and fov_focus_score(ch[CH_MAP2]) < min_focus:
        return pd.DataFrame()  # out-of-focus FOV dropped whole

    nuclei = segment_nuclei(ch[CH_DAPI])
    cells = segment_cell_bodies(ch[CH_MAP2])
    lyso_labels, lamp1_corr = detect_lysosomes(
        ch[CH_LAMP1], threshold=lyso_threshold, bg_floor=bg_floor
    )

    feats = per_cell_features(cells, nuclei, ch, lyso_labels, lamp1_corr)
    qc = qc_filter_cells(cells, nuclei, ch[CH_DAPI])
    df = feats.merge(qc, on="cell_id", how="left")
    for k, v in meta.items():
        df[k] = v
    return df


def build_table(
    nd2_paths: list[str | Path],
    min_focus: float | None = None,
    lyso_threshold: float | None = None,
    bg_floor: float | None = None,
) -> pd.DataFrame:
    """Concatenate analyze_fov over many FOVs into the one tidy table.

    Glob the paths yourself, e.g.
        paths = sorted(Path(data/'d7').rglob('*.nd2'))
        df = build_table(paths)

    lyso_threshold / bg_floor: passed through to analyze_fov -> detect_lysosomes.
      See analyze_fov docstring. Both None = original Otsu behavior (backward compat).
    """
    frames = []
    for p in nd2_paths:
        try:
            frames.append(analyze_fov(p, min_focus=min_focus,
                                       lyso_threshold=lyso_threshold, bg_floor=bg_floor))
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            print(f"  SKIP {Path(p).name}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def condition_stats(
    df: pd.DataFrame,
    value_col: str,
    reference: str = "Control",
    unit: str = "well",
    timepoint: str | None = None,
) -> dict:
    """Well-level stats for one readout: aggregate to well means, then compare conditions.

    THIS is the pseudoreplication guard -- cells -> well means (n=6/condition) before any
    test. Kruskal-Wallis across conditions + Mann-Whitney each condition vs `reference`,
    BH-corrected. Pass timepoint= to restrict to one plate.
    # ponytail: well-means + non-parametric; upgrade to an LMM (statsmodels, cell rows,
    # 1|well/fov random effects) when you add independent differentiations.
    """
    from scipy import stats

    d = df[df.get("qc_pass", True) == True]  # noqa: E712 - only QC-passing cells
    if timepoint is not None:
        d = d[d["timepoint"] == timepoint]

    well_means = d.groupby(["condition", unit])[value_col].mean().reset_index()
    groups = {c: g[value_col].values for c, g in well_means.groupby("condition")}

    out = {"value_col": value_col, "well_means": well_means, "n_per_condition": {c: len(v) for c, v in groups.items()}}
    if len(groups) >= 2:
        out["kruskal_p"] = float(stats.kruskal(*groups.values()).pvalue)

    pairwise, pvals = [], []
    for cond, vals in groups.items():
        if cond == reference or reference not in groups:
            continue
        p = float(stats.mannwhitneyu(vals, groups[reference], alternative="two-sided").pvalue)
        pairwise.append(cond)
        pvals.append(p)
    if pvals:
        from scipy.stats import false_discovery_control

        adj = false_discovery_control(pvals)
        out["vs_reference"] = {
            c: {"p_raw": p, "p_bh": float(a)} for c, p, a in zip(pairwise, pvals, adj)
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

    # stats: fake 2 conditions x 6 wells, KO shifted up -> should separate
    rows = []
    for cond, mu in [("Control", 10.0), ("KO", 20.0)]:
        for wi in range(6):
            for _ in range(30):  # 30 cells/well -> must NOT inflate n
                rows.append({"condition": cond, "well": f"{cond}{wi}",
                             "n_lysosomes": rng.normal(mu, 3), "qc_pass": True})
    res = condition_stats(pd.DataFrame(rows), "n_lysosomes")
    assert res["n_per_condition"] == {"Control": 6, "KO": 6}, "n = wells, not cells"
    assert res["vs_reference"]["KO"]["p_bh"] < 0.05, "clear shift should be significant"
    print("if_features self-check passed:", res["n_per_condition"], "wells/condition")


if __name__ == "__main__":
    _demo()
