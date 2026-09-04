"""Run the LAMP1 pipeline across all timepoints (d7/d14/d28), all conditions.

Builds one per-cell table per timepoint (reuses an existing CSV if present), then
reports QC, condition contrasts, and depth checks PER timepoint PER condition.

  PYTHONPATH=src python run_all.py            # build missing + report
  PYTHONPATH=src python run_all.py rebuild    # force re-segment everything
"""
import sys
from pathlib import Path

import pandas as pd

from tmem_align.analysis.if_features import (
    build_table,
    condition_stats,
    depth_sensitivity,
    qc_summary,
)

DATA = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821")
OUT = Path("reports/if_segmentation_pilot")
MASKS = OUT / "masks"  # napari-ready label TIFFs per FOV (nuclei/cells/lysosomes)
TIMEPOINTS = ["d7", "d14", "d28"]
READOUTS = ["lyso_mean_size_um2", "n_lysosomes", "lyso_area_frac",
            "lamp1_integrated", "perinuclear_index", "tmem_manders_m1"]


def table_for(tp: str, rebuild: bool) -> pd.DataFrame:
    """Per-cell table for one timepoint; reuse CSV unless rebuild."""
    csv = OUT / f"{tp}_percell_full.csv"
    if csv.exists() and not rebuild:
        print(f"[{tp}] reuse {csv}")
        return pd.read_csv(csv)
    paths = sorted((DATA / tp).rglob("*.nd2"))
    print(f"[{tp}] segmenting {len(paths)} FOVs (masks -> {MASKS})...")
    df = build_table(paths, masks_dir=MASKS)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"[{tp}] wrote {csv}  ({len(df)} cells)")
    return df


def report(df: pd.DataFrame) -> None:
    for tp in TIMEPOINTS:
        d = df[df["timepoint"] == tp]
        if d.empty:
            continue
        print(f"\n{'='*70}\nTIMEPOINT {tp}  ({len(d)} cells)\n{'='*70}")
        print("cells/condition:", d.groupby("condition").size().to_dict())

        print("\n-- qc_summary --")
        print(qc_summary(d).to_string(index=False))

        for col in READOUTS:
            r = condition_stats(df, col, timepoint=tp)
            print(f"\n-- {col} --  kruskal p={r.get('kruskal_p'):.3g}"
                  f"  n/cond={r['n_per_condition']}")
            for cond, s in r.get("vs_reference", {}).items():
                ci = s["ci95"]
                print(f"   {cond} vs Control: diff={s['mean_diff']:.3g} "
                      f"CI[{ci[0]:.3g},{ci[1]:.3g}] d={s['cohens_d']:.2f} p_bh={s['p_bh']:.3g}")
            # depth check for EVERY condition at this timepoint
            for cond in sorted(d["condition"].unique()):
                dep = depth_sensitivity(df[df["timepoint"] == tp], col, condition=cond)
                note = f" [{dep['note']}]" if dep.get("note") else ""
                print(f"   depth[{cond}]: rho={dep['rho']:.3g} p={dep['p']:.3g} "
                      f"n={dep['n']}{note}" if dep["rho"] == dep["rho"]
                      else f"   depth[{cond}]: n={dep['n']}{note}")

    # trajectory: well-mean of primary readout, condition x timepoint
    print(f"\n{'='*70}\nTRAJECTORY (well-mean lyso_mean_size_um2, condition x timepoint)\n{'='*70}")
    d = df[df["qc_pass"] == True] if "qc_pass" in df else df  # noqa: E712
    traj = (d.groupby(["timepoint", "condition", "well"])["lyso_mean_size_um2"].mean()
              .groupby(level=["timepoint", "condition"]).mean().unstack("condition"))
    print(traj.reindex(TIMEPOINTS).to_string())


def main() -> None:
    rebuild = len(sys.argv) > 1 and sys.argv[1] == "rebuild"
    frames = [table_for(tp, rebuild) for tp in TIMEPOINTS]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "all_percell_full.csv", index=False)
    print(f"\ncombined -> {OUT/'all_percell_full.csv'}  ({len(df)} cells total)")
    report(df)


if __name__ == "__main__":
    main()
