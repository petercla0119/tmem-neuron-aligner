import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_longitudinal_plate_summary import READOUTS, add_baseline_fold, fold_col  # noqa: E402


def _frame():
    rows = []
    for well in ("C03", "D03"):
        for day in range(5, 18):
            rows.append({"well": well, "day": day, **{k: float(day) for k, _, _ in READOUTS}})
    df = pd.DataFrame(rows)
    df.loc[(df.well == "D03") & (df.day == 6), "nuclei_count"] = 0
    return df


def test_fold_uses_requested_baseline_day():
    df = add_baseline_fold(_frame(), 6)
    c03 = df[df.well == "C03"].set_index("day")
    assert c03.loc[6, fold_col("nuclei_count", 6)] == 1
    assert c03.loc[12, fold_col("mcherry_mean_raw", 6)] == 2
    assert "nuclei_count_day5_fold" not in df


def test_zero_baseline_becomes_nan():
    df = add_baseline_fold(_frame(), 6)
    assert np.isnan(df[df.well == "D03"][fold_col("nuclei_count", 6)]).all()
