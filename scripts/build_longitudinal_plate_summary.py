"""Add plate-wide and per-well three-readout summaries to completed counts output."""

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

READOUTS = [
    ("nuclei_count", "Nuclei count", "Nuclei / acquired field"),
    ("mcherry_mean_raw", "mCherry signal", "Mean valid-field intensity (ADU)"),
    ("mng_mean_per_nucleus_raw", "Nuclear mNG signal", "Mean per-nucleus intensity (ADU)"),
]
COLORS = {"WT61": "#1269a8", "KO61": "#d55e00", "WT62": "#159579", "KO62": "#ae4b9d"}
STYLE = """<style>body{font:16px/1.55 system-ui;color:#193041;background:#f7f9fb;max-width:1240px;margin:auto;padding:30px}h1{line-height:1.2}h2{margin-top:32px}li{margin:6px 0}img{width:100%;height:auto}a{color:#07668e}table{border-collapse:collapse;font-size:13px;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0;padding:10px;background:white}code{overflow-wrap:anywhere}.status{padding:14px;background:#e5f0f5}summary{cursor:pointer;font-weight:600}@media(max-width:700px){.gallery{grid-template-columns:1fr}}</style>"""


def prepare_metrics(base, nuclear, plate_map):
    assert not base.duplicated(["well", "day"]).any()
    expected = {
        (f"{r}{c:02d}", d) for r in "CDEFGHIJKLMN" for c in range(3, 11) for d in range(5, 18)
    }
    assert set(zip(base.well, base.day)) == expected
    df = base.merge(nuclear, on=["well", "day"], validate="one_to_one", suffixes=("", "_mng"))
    assert len(df) == len(base) == 1248
    assert (df.nuclei_count == df.mng_nuclei_count_check).all()
    mapping = {}
    for row in plate_map.itertuples():
        for letter in row.rows:
            well = f"{letter}{int(row.column):02d}"
            assert well not in mapping
            mapping[well] = (f"{row.genotype}{int(row.line)}", row.block)
    assert set(mapping) == set(df.well)
    df["condition"] = df.well.map(lambda w: mapping[w][0])
    df["plate_block"] = df.well.map(lambda w: mapping[w][1])
    assert (df[df.day == 5].groupby("condition").size() == 24).all()
    assert (df[df.day == 5].groupby(["condition", "plate_block"]).size() == 12).all()
    for key, _, _ in READOUTS:
        baseline = df[df.day == 5].set_index("well")[key]
        denominators = df.well.map(baseline).replace(0, np.nan)
        df[key + "_day5_fold"] = df[key] / denominators
    return df.sort_values(["well", "day"])


def summarize(df):
    rows = []
    for (condition, day), g in df.groupby(["condition", "day"]):
        for key, _, _ in READOUTS:
            for mode, col in [("raw", key), ("day5_fold", key + "_day5_fold")]:
                x = g[col].dropna()
                rows.append(
                    {
                        "condition": condition,
                        "day": int(day),
                        "metric": key,
                        "mode": mode,
                        "n_wells": len(x),
                        "median": x.median(),
                        "q25": x.quantile(0.25),
                        "q75": x.quantile(0.75),
                    }
                )
    return pd.DataFrame(rows)


def plot_summary(df, destination, title, individual=False, pdf=True):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7), layout="constrained")
    for j, (key, label, unit) in enumerate(READOUTS):
        for i in range(2):
            ax = axes[i, j]
            col = key if i == 0 else key + "_day5_fold"
            for condition, color in COLORS.items():
                g = df[df.condition == condition]
                if g.empty:
                    continue
                byday = g.groupby("day")[col]
                med = byday.median()
                n = g.well.nunique()
                if individual:
                    for _, well in g.groupby("well"):
                        ax.plot(well.day, well[col], color=color, alpha=0.18, lw=0.65)
                elif n > 1:
                    ax.fill_between(
                        med.index,
                        byday.quantile(0.25),
                        byday.quantile(0.75),
                        color=color,
                        alpha=0.13,
                    )
                ax.plot(
                    med.index,
                    med,
                    color=color,
                    lw=2,
                    marker="o",
                    ms=3,
                    label=f"{condition} (n={n} wells)" if n > 1 else condition,
                )
            ax.set(
                title=label if i == 0 else f"{label}: relative to day 5",
                xlabel="Experimental day",
                ylabel=unit if i == 0 else "Fold of day 5 (within each well)",
                xticks=range(5, 18),
                xlim=(4.7, 17.3),
            )
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.axhline(1, color="#666666", lw=0.7, ls="--")
    axes[0, 0].legend(fontsize=8, frameon=True)
    subtitle = (
        "Thin lines: individual wells; bold: condition median"
        if individual
        else "Lines: condition median; shading: interquartile range across wells"
    )
    if df.well.nunique() == 1:
        subtitle = "Each point is one acquired field; no individual-cell tracking"
    fig.suptitle(title + "\n" + subtitle, fontsize=13)
    fig.savefig(destination.with_suffix(".png"), dpi=160)
    if pdf:
        fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


class Links(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.targets = set()
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        self.targets.update(v for k, v in attrs if k in ("href", "src") and v)


def bullets(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def definitions():
    return bullets(
        [
            "<b>Nuclei:</b> mNG-positive nuclear candidates per acquired field; masks need biological review. Counts do not measure individual-cell survival.",
            "<b>mCherry:</b> mean raw red intensity over valid field pixels, in camera units (ADU).",
            "<b>Nuclear mNG:</b> raw mean intensity within each nuclear mask, averaged with equal weight per nucleus. No background subtraction; mask selection can affect this mean.",
            "<b>Day-5 fold:</b> each well divided by its own day-5 value before condition summaries. Shading shows the middle 50% of wells (IQR), not a confidence interval.",
            "<b>Conditions:</b> WT61, KO61, WT62 and KO62 from the experiment plate map; 24 wells each (12 per C–H/I–N block). KO denotes GRN mutant. Both genotypes carry the red reporter; no reporter-negative control. Summaries are descriptive well-level comparisons.",
        ]
    )


def compact_well(old, df, cfg, well):
    table = re.search(
        r'<h2>First-to-last observed changes</h2><div class="scroll">.*?</table></div>',
        old,
        re.DOTALL,
    ).group()
    gallery = re.search(r'<div class="gallery">.*?</div>', old, re.DOTALL).group()
    condition = df.condition.iloc[0]
    return (
        header(f"{well}: nuclei and fluorescence over time")
        + f"<h1>{well} · {condition}</h1><p class='status'>13/13 images verified · experimental days 5–17 · automated first pass; biological mask review pending.</p>"
        + '<a href="readout_summary.png"><img src="readout_summary.png" alt="Nuclei counts, mCherry and nuclear mNG across days 5–17; raw and day-5 fold"></a><p><a href="readout_summary.pdf">Three-readout summary PDF</a> · <a href="../../longitudinal_summary_metrics.csv">All readouts CSV</a> · <a href="../../report.html">Plate overview</a></p>'
        + '<details><summary>Additional mCherry plots and original downloads</summary><a href="longitudinal_summary.png"><img src="longitudinal_summary.png" alt="Additional mCherry time courses"></a><p><a href="well_day_metrics.csv">Original measurements CSV</a> · <a href="endpoint_summary.csv">Endpoint CSV</a> · <a href="longitudinal_summary.pdf">Original summary PDF</a> · <a href="nuclei_counts.png">Nuclei plot</a> · <a href="mcherry_signal.png">mCherry plot</a></p></details>'
        + table
        + "<h2>Readouts</h2>"
        + definitions()
        + bullets(
            [
                "<b>Corrected mCherry:</b> subtracts each image’s 10th percentile of valid block-averaged red pixels. Diffuse biological signal can shift this estimate; retain the raw curve.",
                "<b>Corrected total mCherry / nuclei:</b> a population-normalized field measure, not segmented whole-cell expression or a tracked-cell trajectory.",
            ]
        )
        + "<h2>Processing</h2>"
        + bullets(
            [
                "Read-only ND2 input; channel/exposure checks; exact 2×2 block means. Trim one bottom row/right column and exclude blocks containing zero padding.",
                "Green is copied before Cellpose percentile normalization; saved green/red retain camera units. No darkfield, flatfield, deconvolution, bleaching correction, temporal alignment or tracking.",
                f"Cellpose-SAM {cfg['versions']['cellpose']}: mNG/488 nm only; flow threshold 0.4, cell probability 0, minimum size 15 model pixels. Masks use 2× reduced coordinates; centroids use original coordinates; field integrals include the 4× pixel-area factor.",
                "All 13 acquisitions: 100 ms at 488 and 561 nm. Equal exposure does not establish equal excitation/gain or absence of bleaching. Nuclear identity follows the experiment workflow and image appearance; no independent nuclear co-stain validation.",
                f"Sampled area: {df.area_mm2.min():.3f}–{df.area_mm2.max():.3f} mm²; maximum raw-red saturated fraction: {df.mcherry_saturated_fraction.max():.6g}. CSV includes focus and boundary-object diagnostics; undocumented PFS codes are not definitive focus-failure calls.",
            ]
        )
        + "<h2>Resolution sensitivity</h2>"
        + bullets(
            [
                "4× reduction was rejected: late images lost up to 18% of detections versus 2×.",
                "2× / native counts: E04 day 5, 285 / 287; E04 day 17, 183 / 193; F10 day 17, 148 / 161, with approximately matched physical size cutoffs.",
                "2× is a first-pass speed/accuracy compromise and can exaggerate temporal decline. Native model counts are not ground truth; small changes within this sensitivity are not established biological effects.",
            ]
        )
        + '<p><a href="resolution_qc/initial_resolution_comparison.png">Initial comparison</a> · <a href="resolution_qc/physical_size_resolution_comparison.png">Late-day native/2×/4× comparison</a> · <a href="resolution_qc/late_day_resolution_comparison.csv">Five-well late-day CSV</a></p><h2>Mask review</h2>'
        + bullets(
            [
                "Panels: complete green field; central detail with cyan nuclear outlines; mCherry at a shared 90–600 ADU display range.",
                "<code>frames/</code>: label arrays and per-nucleus tables. <code>preprocessed/</code>: quantitative arrays. Nuclear red columns measure that compartment, not whole-cell mCherry expression.",
            ]
        )
        + gallery
        + "<h2>Verification</h2>"
        + bullets(
            [
                "13 frames passed mask/table counts, valid-area checks, fluorescence sum/mean recalculation and preserved green-unit checks. New mNG mask counts also match all original counts.",
                '<a href="artifact_verification.csv">Verification CSV</a> · <a href="image_inventory.csv">Source inventory</a> · <a href="config.json">Model hash, parameters and versions</a>',
            ]
        )
        + "</html>"
    )


def header(title):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{STYLE}'


def replace_report(path, body, root, backups):
    old = path.read_text()
    backup = backups / path.relative_to(root)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(old)
    original = backup.read_text()
    assert Links(original).targets <= Links(body).targets, (
        path,
        Links(original).targets - Links(body).targets,
    )
    for link in Links(body).targets:
        assert (path.parent / link).exists(), (path, link)
    assert path.read_text() == old, "Report changed during rendering"
    temp = path.with_suffix(".html.tmp")
    temp.write_text(body)
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plate-map", type=Path, required=True)
    ap.add_argument("--backups", type=Path, required=True)
    args = ap.parse_args()
    root = args.output
    original_path = root / "well_day_metrics.csv"
    source_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    df = prepare_metrics(
        pd.read_csv(original_path),
        pd.read_csv(root / "nuclear_mng_metrics.csv"),
        pd.read_csv(args.plate_map),
    )
    df.to_csv(root / "longitudinal_summary_metrics.csv", index=False)
    summary = summarize(df)
    summary.to_csv(root / "condition_day_summary.csv", index=False)
    (root / "longitudinal_plate_map.csv").write_bytes(args.plate_map.read_bytes())
    plot_summary(
        df, root / "plate_longitudinal_summary", "Full plate · nuclei and fluorescence · days 5–17"
    )
    plot_summary(
        df,
        root / "plate_individual_wells",
        "Full plate · individual well trajectories",
        individual=True,
    )
    for i, (well, g) in enumerate(df.groupby("well")):
        out = root / "wells" / well
        cfg = json.loads((out / "config.json").read_text())
        inv = pd.read_csv(out / "image_inventory.csv")
        assert set(inv.exposure_488_ms) == {100} and set(inv.exposure_561_ms) == {100}
        assert cfg["scale"] == 2
        plot_summary(g, out / "readout_summary", f"{well} · {g.condition.iloc[0]} · days 5–17")
        backup = args.backups / "wells" / well / "report.html"
        old = (backup if backup.exists() else out / "report.html").read_text()
        replace_report(out / "report.html", compact_well(old, g, cfg, well), root, args.backups)
        if (i + 1) % 12 == 0:
            print(f"Reports updated: {i + 1}/96", flush=True)
    original = args.backups / "report.html"
    old = (original if original.exists() else root / "report.html").read_text()
    table = re.search(r"<table>.*?</table>", old, re.DOTALL).group()
    body = (
        header("Full-plate longitudinal summary")
        + '<h1>Nuclei and fluorescence over time</h1><p class="status">96/96 wells · 1,248/1,248 images · experimental days 5–17</p><a href="plate_longitudinal_summary.png"><img src="plate_longitudinal_summary.png" alt="Counts, mCherry and nuclear mNG across experimental days; raw and baseline-normalized"></a><p><a href="plate_longitudinal_summary.pdf">Summary PDF</a> · <a href="longitudinal_summary_metrics.csv">All readouts CSV</a> · <a href="condition_day_summary.csv">Condition/day medians and IQR CSV</a></p><h2>Reading the plots</h2>'
        + definitions()
        + '<details><summary>Individual well trajectories</summary><a href="plate_individual_wells.png"><img src="plate_individual_wells.png" alt="All 96 individual well trajectories"></a><a href="plate_individual_wells.pdf">Individual trajectories PDF</a></details><h2>Method and limits</h2>'
        + bullets(
            [
                "Same first-pass segmentation as the five-well pilot: 2× block averages; Cellpose-SAM on mNG; zero padding excluded. Nuclear mNG was measured from saved quantitative green images and existing masks; no resegmentation.",
                "No retrospective illumination or bleaching correction, temporal alignment, individual-cell tracking or survival inference. Resolution can exaggerate count decline; masks need biological review. Full settings, diagnostics and resolution comparisons are in each well report.",
                '<a href="well_day_metrics.csv">Original counts/mCherry CSV</a> · <a href="endpoint_summary.csv">Original endpoint CSV</a> · <a href="nuclear_mng_metrics.csv">Nuclear mNG CSV</a> · <a href="nuclear_mng/provenance.json">mNG provenance</a> · <a href="longitudinal_plate_map.csv">Plate map CSV</a>',
            ]
        )
        + "<h2>Per-well reports</h2>"
        + table
        + "</html>"
    )
    replace_report(root / "report.html", body, root, args.backups)
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == source_hash
    provenance = {
        "frames": len(df),
        "wells": df.well.nunique(),
        "days": sorted(map(int, df.day.unique())),
        "original_measurements_sha256": source_hash,
        "plate_map_sha256": hashlib.sha256(args.plate_map.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": "Condition median and IQR; within-well day-5 normalization; descriptive wells, not confidence intervals",
        "reports_verified": 97,
        "original_links_preserved": True,
    }
    (root / "longitudinal_summary_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
