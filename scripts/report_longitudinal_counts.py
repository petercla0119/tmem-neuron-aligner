"""Build the analysis report and verify saved frame measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output
    df = pd.read_csv(out / "well_day_metrics.csv")
    inv = pd.read_csv(out / "image_inventory.csv")
    cfg = json.loads((out / "config.json").read_text())
    expected = set(zip(inv.well, inv.day))
    actual = set(zip(df.well, df.day))
    assert not df.duplicated(["well", "day"]).any()
    assert actual <= expected
    checks = []
    for r in df.itertuples():
        key = f"{r.well}_d{r.day}"
        labels = np.load(out / "frames" / f"{key}_labels.npz")["labels"]
        data = np.load(out / "preprocessed" / f"{key}.npz")
        green, red, valid = data["green"], data["red"], data["valid"]
        vals = red[valid].astype(np.float64)
        n = len(np.unique(labels[labels > 0]))
        cells = pd.read_csv(out / "frames" / f"{key}_nuclei.csv")
        assert n == r.nuclei_count == len(cells), (key, "count mismatch")
        assert not labels[~valid].any(), (key, "invalid area contains masks")
        assert np.isclose(vals.mean(), r.mcherry_mean_raw), (key, "red units changed")
        assert np.isclose(vals.sum() * cfg["scale"] ** 2, r.mcherry_integrated_raw), (
            key,
            "integrated signal mismatch",
        )
        assert np.isclose((vals.mean() - r.mcherry_background), r.mcherry_mean_bg_corrected)
        assert np.median(green[valid]) > 50, (key, "green was normalized in place")
        checks.append(
            {
                "well": r.well,
                "day": r.day,
                "mask_count": n,
                "table_count": len(cells),
                "red_arithmetic": "pass",
                "green_units": "pass",
            }
        )
    pd.DataFrame(checks).to_csv(out / "artifact_verification.csv", index=False)
    end = []
    for well, g in df.sort_values("day").groupby("well"):
        first, last = g.iloc[0], g.iloc[-1]
        end.append(
            {
                "Well": well,
                "First day": int(first.day),
                "Baseline day": int(last.baseline_day),
                "Last day": int(last.day),
                "Nuclei first": int(first.nuclei_count),
                "Nuclei last": int(last.nuclei_count),
                "Nuclei change (%)": 100 * last.nuclei_count_fold_baseline - 100,
                "Raw mCherry mean change (%)": 100 * last.mcherry_mean_raw_fold_baseline - 100,
                "Corrected mCherry mean change (%)": 100
                * last.mcherry_mean_bg_corrected_fold_baseline
                - 100,
                "Corrected mCherry per nucleus change (%)": 100
                * last.mcherry_integrated_per_nucleus_fold_baseline
                - 100,
            }
        )
    summary = pd.DataFrame(end)
    summary.to_csv(out / "endpoint_summary.csv", index=False)
    complete = actual == expected
    status = "Complete automated first pass" if complete else "Processing in progress"
    progress_note = (
        "<p>While processing, the linked figures and CSV update after each image. "
        "This report table is a verified snapshot and refreshes when the full run finishes.</p>"
        if not complete
        else ""
    )
    gallery = "".join(
        f'<figure><a href="qc/{r.well}_d{r.day}.png"><img loading="lazy" src="qc/{r.well}_d{r.day}.png"></a><figcaption>{r.well}, day {r.day}</figcaption></figure>'
        for r in df.itertuples()
        if r.day in (5, 11, 17)
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Longitudinal nuclei and mCherry</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#193041;background:#f7f9fb;max-width:1240px;margin:auto;padding:35px}}h1{{font-size:32px;line-height:1.2}}h2{{margin-top:40px}}img{{width:100%;height:auto}}.status{{background:#e5f0f5;border-left:5px solid #176788;padding:16px}}table{{border-collapse:collapse;font-size:13px;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}a{{color:#07668e}}.gallery{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}figure{{margin:0;background:white;padding:10px}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.gallery{{grid-template-columns:1fr}}.scroll{{overflow:auto}}}}</style>
<h1>Nuclei and mCherry across time</h1><p>{" · ".join(cfg["wells"])} | experimental days {min(cfg["days"])}–{max(cfg["days"])}</p>
<p class="status"><strong>{status}: {len(df)}/{len(inv)} images.</strong> Automated mNG-positive nuclear candidates; masks require biological review. These are counts in complete acquired fields, not individual-cell survival.</p>
{progress_note}
<a href="longitudinal_summary.png"><img src="longitudinal_summary.png" alt="Six panels showing nuclei and mCherry time courses"></a>
<p><a href="well_day_metrics.csv">All well/day measurements (CSV)</a> · <a href="endpoint_summary.csv">Endpoint summary (CSV)</a> · <a href="longitudinal_summary.pdf">Summary figure (PDF)</a> · <a href="nuclei_counts.png">Nuclei plot</a> · <a href="mcherry_signal.png">mCherry plot</a></p>
<h2>First-to-last observed changes</h2><div class="scroll">{summary.to_html(index=False, border=0, float_format=lambda x: f"{x:.1f}")}</div>
<h2>How to read the measurements</h2><p>Raw mCherry mean retains camera intensity units (ADU). Background-corrected mean subtracts the 10th percentile of valid, block-averaged red pixels in each image. The raw curve is provided alongside it because diffuse biological signal can affect this background estimate. Integrated corrected signal divided by nuclei count is a population-normalized field measurement; it is not a segmented whole-cell measurement or a tracked-cell trajectory.</p>
<h2>Processing and reproducibility</h2><p>Read-only ND2 input; fresh channel and exposure checks; exact {cfg["scale"]}×{cfg["scale"]} block averages; zero-padded blocks excluded. One bottom row and right column are trimmed to obtain an exactly divisible image. Green is copied before Cellpose percentile normalization, keeping saved green and quantitative red in camera units. Cellpose-SAM {cfg["versions"]["cellpose"]} uses only the mNG/488 nm channel with flow threshold 0.4, cell probability threshold 0 and minimum size 15 model pixels. Masks are saved at {cfg["scale"]}× reduced resolution; cell centroids refer to original image coordinates. Field totals include the original-pixel area factor. No darkfield, flatfield, deconvolution, photobleaching correction, temporal alignment, or individual-cell tracking was applied.</p>
<p>Recorded exposures across {len(inv)} selected acquisitions are 488 nm {", ".join(map(str, sorted(inv.exposure_488_ms.unique())))} ms and 561 nm {", ".join(map(str, sorted(inv.exposure_561_ms.unique())))} ms. Matching exposure does not establish identical excitation power, detector gain, or absence of photobleaching. mNG was treated as the nuclear marker based on the existing experiment workflow and image appearance; no independent nuclear co-stain validation was supplied. No treatment labels or replicate group assumptions are assigned.</p>
<p>Sampled area: {df.area_mm2.min():.3f}–{df.area_mm2.max():.3f} mm². Maximum raw red saturated-pixel fraction: {df.mcherry_saturated_fraction.max():.6g}. Image-level focus and boundary-object diagnostics are in the CSV. PFS status codes are preserved without treating undocumented codes as definitive focus failure.</p>
<h2>Resolution sensitivity</h2><p>Fourfold reduction was rejected after late-day tests lost up to 18% of detections compared with twofold reduction. Twofold reduction versus native resolution counted 285 versus 287 in the initial day-5 E04 test, and 183 versus 193 (E04) and 148 versus 161 (F10) in day-17 tests with approximately matched physical size cutoffs. This is a first-pass speed/accuracy compromise; resolution can exaggerate temporal decline. Native-resolution counts are also model outputs, not ground truth. Small changes within this sensitivity should not be interpreted as established biological effects.</p>
<p><a href="resolution_qc/initial_resolution_comparison.png">Initial comparison</a> · <a href="resolution_qc/physical_size_resolution_comparison.png">Late-day native/2×/4× comparison</a> · <a href="resolution_qc/late_day_resolution_comparison.csv">Five-well late-day comparison</a></p>
<h2>Mask review</h2><p>Each panel shows the complete green field, a central detail with cyan nuclear outlines, and mCherry at a shared 90–600 ADU display range. Full label arrays and per-nucleus measurements are in <code>frames/</code>; quantitative preprocessed arrays are in <code>preprocessed/</code>. Nuclear red intensity columns are compartment-specific and must not be presented as whole-cell mCherry expression.</p><div class="gallery">{gallery}</div>
<h2>Verification and provenance</h2><p>{len(checks)} saved frames passed mask/table count agreement, valid-area checks, fluorescence sum/mean recalculation, and preserved green-unit checks. <a href="artifact_verification.csv">Verification table</a> · <a href="image_inventory.csv">Source inventory</a> · <a href="config.json">Model hash, parameters and versions</a></p></html>"""
    (out / "report.html").write_text(body)
    (out / "completion.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "completed": len(df),
                "expected": len(inv),
                "missing": sorted(expected - actual),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    print(f"Artifacts verified: {len(checks)}; complete: {complete}")


if __name__ == "__main__":
    main()
