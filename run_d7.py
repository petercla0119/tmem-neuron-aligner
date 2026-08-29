"""Scratch driver: run the LAMP1 pipeline on the d7 plate. Not committed.

  PYTHONPATH=src python run_d7.py smoke   # one FOV per condition (fast, catches errors)
  PYTHONPATH=src python run_d7.py full    # all d7 FOVs -> CSV + stats
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

D7 = Path("/Users/pmihack/claire/tmem_2026/data/cleaved_tmem_pld3_260821/d7")
COND_DIRS = ["TMEM_KO", "Z59_PLD_Control", "Z60_PLD_TMEMki"]
OUT = Path("reports/if_segmentation_pilot")


def paths(mode: str) -> list[Path]:
    if mode == "smoke":  # one FOV per condition
        return [sorted((D7 / c).glob("*.nd2"))[0] for c in COND_DIRS]
    return sorted(D7.rglob("*.nd2"))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    p = paths(mode)
    print(f"[{mode}] {len(p)} FOV(s)")
    df = build_table(p)
    if df.empty:
        print("EMPTY TABLE - all FOVs failed"); return

    print("\ncolumns:", list(df.columns))
    print("cells total:", len(df), "| per condition:")
    print(df.groupby("condition").size())
    print("\nn_z_slices seen:", sorted(df["n_z_slices"].dropna().unique().tolist()))

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"d7_percell_{mode}.csv"
    df.to_csv(csv, index=False)
    print("wrote", csv)

    print("\n=== qc_summary ===")
    print(qc_summary(df).to_string(index=False))

    for col in ["n_lysosomes", "lyso_mean_size_um2", "lamp1_integrated"]:
        print(f"\n=== condition_stats: {col} ===")
        r = condition_stats(df, col)
        print("n/condition:", r["n_per_condition"], "| kruskal p:", r.get("kruskal_p"))
        for cond, s in r.get("vs_reference", {}).items():
            print(f"  {cond} vs Control: diff={s['mean_diff']:.3g} "
                  f"CI[{s['ci95'][0]:.3g},{s['ci95'][1]:.3g}] d={s['cohens_d']:.2f} "
                  f"p_bh={s['p_bh']:.3g}")
        dep = depth_sensitivity(df, col)
        print(f"  depth check (Control): rho={dep['rho']} p={dep['p']} n={dep['n']}"
              + (f" [{dep.get('note')}]" if dep.get('note') else ""))


if __name__ == "__main__":
    main()
