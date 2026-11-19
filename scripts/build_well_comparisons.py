"""Build linked descriptive within-well and within-condition comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_longitudinal_plate_summary import (
    COLORS,
    READOUTS,
    Links,
    bullets,
    fold_col,
    header,
    plot_summary,
)
from matplotlib.colors import TwoSlopeNorm

from tmem_align.analysis.spatial_comparisons import relative_cv

SPATIAL = [("nuclei_per_mm2", "Nuclei density", "Candidates / valid mm²"), *READOUTS[1:]]
DAYS = [5, 11, 17]


def select_examples(df, baseline_day):
    result = []
    for condition, g in df.groupby("condition"):
        columns = [fold_col(k, baseline_day) for k, _, _ in READOUTS]
        scores = pd.Series(0.0, index=sorted(g.well.unique()))
        for column in columns:
            matrix = g[g.day > baseline_day].pivot(index="well", columns="day", values=column)
            median = matrix.median()
            spread = (matrix.quantile(0.75) - matrix.quantile(0.25)).clip(lower=0.02)
            scores += ((matrix - median) / spread).abs().mean(axis=1)
        scores /= len(columns)
        endpoint = g[g.day == 17].set_index("well")[fold_col("nuclei_count", baseline_day)]
        typical = scores.sort_values(kind="stable").index[0]
        others = endpoint.drop(typical).sort_values(kind="stable")
        for role, well in [
            ("Typical trajectory", typical),
            ("Lowest count fold", others.index[0]),
            ("Highest count fold", others.index[-1]),
        ]:
            result.append(
                {
                    "condition": condition,
                    "role": role,
                    "well": well,
                    "trajectory_distance": scores[well],
                    "nuclei_day17_fold": endpoint[well],
                    "selection_rule": f"Typical: lowest mean absolute distance from median profile across 3 day{baseline_day}-fold readouts and days {baseline_day + 1}-17; divide by daily IQR floored at 0.02. Contrasts: lowest/highest day17 count fold among remaining wells.",
                }
            )
    return pd.DataFrame(result)


def variability(df, tiles, baseline_day):
    within = []
    between = []
    for (condition, well, day), g in tiles.groupby(["condition", "well", "day"]):
        for key, _, _ in SPATIAL:
            values = g.loc[g.valid_fraction >= 0.5, key]
            within.append(
                {
                    "condition": condition,
                    "well": well,
                    "day": int(day),
                    "metric": key,
                    "n_valid_tiles": int(values.notna().sum()),
                    "spatial_cv": relative_cv(values),
                    "spatial_iqr": values.quantile(0.75) - values.quantile(0.25),
                }
            )
    for (condition, day), g in df.groupby(["condition", "day"]):
        for key, _, _ in READOUTS:
            for mode, col in [
                ("raw", key),
                (f"day{baseline_day}_fold", fold_col(key, baseline_day)),
            ]:
                x = g[col].dropna()
                between.append(
                    {
                        "condition": condition,
                        "day": int(day),
                        "metric": key,
                        "mode": mode,
                        "n_wells": len(x),
                        "mean": x.mean(),
                        "sd": x.std(ddof=1),
                        "cv": relative_cv(x),
                        "median": x.median(),
                        "q25": x.quantile(0.25),
                        "q75": x.quantile(0.75),
                        "min": x.min(),
                        "max": x.max(),
                    }
                )
    return pd.DataFrame(within), pd.DataFrame(between)


def save(fig, path, pdf=True):
    fig.savefig(path.with_suffix(".png"), dpi=150)
    if pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def heatmap(g, out, baseline_day):
    wells = sorted(g.well.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 10.5), layout="constrained")
    for ax, (key, label, _) in zip(axes, READOUTS, strict=True):
        m = g.pivot(index="well", columns="day", values=fold_col(key, baseline_day)).loc[wells]
        # Common ratio bounds across all three panels, with every value preserved in CSV.
        im = ax.imshow(
            m,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2),
            interpolation="nearest",
        )
        ax.set(
            xticks=range(13),
            xticklabels=range(5, 18),
            yticks=range(24),
            yticklabels=wells,
            title=label,
            xlabel="Experimental day",
        )
        fig.colorbar(im, ax=ax, label="Fold of own day 5", extend="max", shrink=0.8)
    fig.suptitle(
        f"{g.condition.iloc[0]} · every well across all days\nDay-5 fold; fixed 0–2 display range (values above 2 saturate)",
        fontsize=14,
    )
    save(fig, out / "change_heatmap")


def paired(g, out, baseline_day):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, (key, label, unit) in enumerate(READOUTS):
        for i in range(2):
            col = key if i == 0 else fold_col(key, baseline_day)
            ax = axes[i, j]
            for well, w in g[g.day.isin(DAYS)].groupby("well"):
                ax.plot(
                    w.day,
                    w[col],
                    color=COLORS[g.condition.iloc[0]],
                    alpha=0.3,
                    lw=0.8,
                    marker="o",
                    ms=3,
                )
            med = g[g.day.isin(DAYS)].groupby("day")[col].median()
            ax.plot(med.index, med, color="black", marker="D", lw=2, label="Median of 24 wells")
            ax.set(
                title=label + (" · raw" if i == 0 else " · within-well change"),
                xticks=DAYS,
                xlabel="Experimental day",
                ylabel=unit if i == 0 else "Fold of day 5",
            )
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{g.condition.iloc[0]} · paired well measurements\nEach thin line links the same well; days 5, 11 and 17",
        fontsize=14,
    )
    save(fig, out / "paired_days")


def variability_plot(condition, within, between, out, baseline_day):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, ((key, label, _), (skey, _, _)) in enumerate(zip(READOUTS, SPATIAL, strict=True)):
        b = between[(between.condition == condition) & (between.metric == key)]
        for mode, style in [("raw", "-"), (f"day{baseline_day}_fold", "--")]:
            x = b[b["mode"] == mode]
            axes[0, j].plot(
                x.day, 100 * x.cv, style, marker="o", ms=3, label=mode.replace("_", " ")
            )
        axes[0, j].set(title=f"Between wells · {label}", ylabel="CV across 24 wells (%)")
        w = within[(within.condition == condition) & (within.metric == skey)]
        by = w.groupby("day").spatial_cv
        med = 100 * by.median()
        axes[1, j].plot(med.index, med, color=COLORS[condition], marker="o", ms=3)
        axes[1, j].fill_between(
            med.index,
            100 * by.quantile(0.25),
            100 * by.quantile(0.75),
            color=COLORS[condition],
            alpha=0.2,
        )
        axes[1, j].set(
            title=f"Within wells · {'nuclear density' if j == 0 else label}",
            ylabel="Spatial CV across nine tiles (%)",
        )
    for ax in axes.ravel():
        ax.set(xticks=range(5, 18), xlabel="Experimental day")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{condition} · variation between wells and across each field\nBottom: median and IQR of 24 within-well spatial CVs; descriptive only",
        fontsize=13,
    )
    save(fig, out / "variability")


def spatial_plot(well, g, tiles, limits, out):
    t = tiles[tiles.well == well]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for j, (key, label, unit) in enumerate(SPATIAL):
        matrix = t.pivot(index="tile", columns="day", values=key).sort_index()
        baseline = g.set_index("day")[key]
        for i in range(2):
            ax = axes[i, j]
            values = matrix if i == 0 else matrix / baseline.replace(0, np.nan)
            if i == 0:
                im = ax.imshow(
                    values, aspect="auto", cmap="viridis", vmin=limits[key][0], vmax=limits[key][1]
                )
            else:
                im = ax.imshow(
                    values,
                    aspect="auto",
                    cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=2),
                )
            ax.set(
                title=label + (" · raw" if i == 0 else " · relative to whole field"),
                xticks=range(13),
                xticklabels=range(5, 18),
                yticks=range(9),
                yticklabels=[f"R{k // 3 + 1}C{k % 3 + 1}" for k in range(9)],
                xlabel="Experimental day",
            )
            fig.colorbar(
                im,
                ax=ax,
                label=unit if i == 0 else "Tile / whole field on same day",
                shrink=0.8,
                extend="neither" if i == 0 else "max",
            )
    fig.suptitle(
        f"{well} · spatial patterns across time\nTop: shared plate-wide color scales; bottom: spatial differences after dividing by that day’s whole-field value",
        fontsize=12,
    )
    save(fig, out / "spatial_patterns", pdf=False)


def figure(name, title):
    return f'<h2>{title}</h2><a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{title}"></a><p><a href="{name}.pdf">Download PDF</a></p>'


def write_html(path, body):
    for target in Links(body).targets:
        if not target.startswith("#") and "://" not in target:
            assert (path.parent / target).exists(), (path, target)
    path.write_text(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = a.output
    out = root / "comparisons"
    df = pd.read_csv(root / "longitudinal_summary_metrics.csv")
    # The summary stage owns the baseline day; read it back so the two reports cannot disagree.
    b = json.loads((root / "longitudinal_summary_provenance.json").read_text())["baseline_day"]
    assert fold_col("nuclei_count", b) in df, f"metrics CSV lacks day-{b} fold columns"
    tiles = pd.read_csv(out / "spatial_tile_metrics.csv")
    expected = {(w, d, t) for w in df.well.unique() for d in range(5, 18) for t in range(9)}
    assert set(zip(tiles.well, tiles.day, tiles.tile)) == expected and len(tiles) == 11232
    selected = select_examples(df, b)
    selected.to_csv(out / "selected_image_examples.csv", index=False)
    within, between = variability(df, tiles, b)
    within.to_csv(out / "within_well_spatial_variability.csv", index=False)
    between.to_csv(out / "between_well_variability.csv", index=False)
    endpoints = df[df.day == 17][
        ["condition", "well", "plate_block"] + [fold_col(k, b) for k, _, _ in READOUTS]
    ].copy()
    for k, _, _ in READOUTS:
        endpoints[k + "_change_percent"] = 100 * (endpoints[fold_col(k, b)] - 1)
    endpoints.to_csv(out / "well_endpoint_changes.csv", index=False)
    limits = {key: (float(tiles[key].min()), float(tiles[key].max())) for key, _, _ in SPATIAL}
    conditions = []
    for condition, g in df.groupby("condition", sort=False):
        dest = out / "conditions" / condition
        dest.mkdir(parents=True, exist_ok=True)
        plot_summary(
            g, dest / "trajectories", f"{condition} · all 24 well trajectories", b, individual=True
        )
        heatmap(g, dest, b)
        paired(g, dest, b)
        variability_plot(condition, within, between, dest, b)
        condition_body = (
            header(f"{condition}: within-condition comparisons")
            + f'<h1>{condition} · comparisons among 24 wells</h1><p><a href="../../report.html">Comparison index</a> · <a href="../../../report.html">Plate summary</a></p>'
        )
        condition_body += bullets(
            [
                "Thin lines connect repeated measurements of the same well; black lines in paired plots show the condition median. No hypothesis tests or confidence intervals.",
                "CV = sample SD / mean. Between-well CV compares 24 wells; spatial CV summarizes nine coordinate tiles within each well. Tiles and time points are not independent biological replicates.",
                "Raw fluorescence includes background, so its CV depends on the camera offset and is not directly comparable across channels. Focus, illumination, expression and segmentation can all affect these patterns.",
            ]
        )
        condition_body += (
            figure("trajectories", "Individual well trajectories")
            + figure("change_heatmap", "Changes within every well")
            + figure("paired_days", "Paired early, middle and late measurements")
            + figure("variability", "Between-well and spatial variability")
        )
        condition_body += "<h2>Typical and contrasting image examples</h2><p>Typical = closest to the median three-readout time profile. Contrasts = lowest/highest day-17 count fold among other wells; these are not fluorescence extremes. Selection is descriptive, without excluding outliers.</p>"
        for row in selected[selected.condition == condition].itertuples():
            condition_body += f'<h3>{row.role}: <a href="../../wells/{row.well}/report.html">{row.well}</a> · count fold {row.nuclei_day17_fold:.2f}</h3>'
            for d in (5, 17):
                condition_body += f'<img loading="lazy" src="../../images/{row.well}_d{d}_overview.png" alt="{row.well} day {d} full fields and fixed center crops">'
            condition_body += f'<p><a href="../../wells/{row.well}/report.html">All three days and nine spatial crops</a></p>'
        condition_body += "<h2>All wells</h2><table><tr><th>Well</th><th>Plate block</th><th>Count change</th><th>mCherry change</th><th>Nuclear mNG change</th></tr>"
        for r in endpoints[endpoints.condition == condition].itertuples():
            condition_body += f'<tr><td><a href="../../wells/{r.well}/report.html">{r.well}</a></td><td>{r.plate_block}</td><td>{r.nuclei_count_change_percent:+.1f}%</td><td>{r.mcherry_mean_raw_change_percent:+.1f}%</td><td>{r.mng_mean_per_nucleus_raw_change_percent:+.1f}%</td></tr>'
        condition_body += "</table></html>"
        conditions.append((dest / "report.html", condition_body))
        print("Condition plots:", condition, flush=True)
    for i, (well, g) in enumerate(df.groupby("well"), 1):
        dest = out / "wells" / well
        dest.mkdir(parents=True, exist_ok=True)
        spatial_plot(well, g, tiles, limits, dest)
        condition = g.condition.iloc[0]
        body = (
            header(f"{well}: temporal and spatial comparisons")
            + f'<h1>{well} · {condition}</h1><p><a href="../../conditions/{condition}/report.html">Other wells in {condition}</a> · <a href="../../../wells/{well}/report.html">Original measurements and mask QC</a> · <a href="../../report.html">Comparison index</a></p>'
        )
        body += (
            "<h2>Change within this well</h2>"
            + f'<img src="../../../wells/{well}/readout_summary.png" alt="Three-readout time courses">'
        )
        body += '<h2>Spatial variation across time</h2><img src="spatial_patterns.png" alt="Nine-tile spatial patterns across all13 days">'
        body += bullets(
            [
                "Nine equal coordinate tiles cover the acquired field. Nuclear density uses valid area; red means use valid pixels; nuclear green is the equal-nucleus mean of whole-mask intensities, assigned by centroid.",
                "Bottom panels divide each tile by the whole-field value on the same day. Ratios above 2 saturate the display; the CSV retains all values. Shared raw color limits span all tiles on the plate.",
                "Coordinates are fixed, but fields are not registered and cells are not tracked. Corresponding crops do not establish that the same cells are present.",
            ]
        )
        body += "<h2>Matched image examples</h2><p>Grayscale channel views: green 80–2000 ADU; red 90–600 ADU. Cyan = nuclear outlines; yellow = center crop; full fields use an 8-pixel display stride. Display clipping is not detector saturation.</p>"
        for d in DAYS:
            body += f'<img loading="lazy" src="../../images/{well}_d{d}_overview.png" alt="{well} day {d} fixed-scale views">'
            body += f'<details><summary>Day {d}: nine spatial crops</summary><img loading="lazy" src="../../images/{well}_d{d}_regions.png" alt="Nine fixed-position crops"></details>'
        body += '<p><a href="../../image_display_clipping.csv">Display-clipping fractions</a> · <a href="../../spatial_tile_metrics.csv">All spatial measurements</a> · <a href="../../within_well_spatial_variability.csv">Spatial CV table</a></p></html>'
        # Condition pages are written after all well pages; check their future targets afterwards.
        (dest / "report.html").write_text(body)
        if i % 24 == 0:
            print("Well comparison pages:", i, "/96", flush=True)
    index = (
        header("Within- and between-well comparisons")
        + '<h1>Within-well and within-condition comparisons</h1><p>96 wells · 13 days · 11,232 spatial tiles · image examples for every well on days 5, 11 and 17.</p><p><a href="../report.html">Back to plate summary</a></p><h2>Choose a condition</h2><ul>'
    )
    for condition in COLORS:
        index += f'<li><a href="conditions/{condition}/report.html"><strong>{condition}</strong> — 24 wells, comparisons and image examples</a></li>'
    index += "</ul>" + bullets(
        [
            "Within wells: day-5-normalized trajectories, paired days, spatial maps and nine fixed-location image crops.",
            "Between wells within a condition: all individual curves, 24-well heatmaps, paired measurements and descriptive variability. Every well is retained.",
            "Images share fixed display ranges, scale bars and existing nuclear outlines. The saved quantitative channels and original masks are unchanged.",
            "Automated candidates require biological review. No tracking, alignment, illumination or bleaching correction; count differences are not individual-cell survival. C03 warrants priority review because green intensity falls sharply while later candidate counts recover.",
        ]
    )
    index += "<h2>Downloads and methods</h2><ul>"
    for file, label in [
        ("spatial_tile_metrics.csv", "All nine-tile measurements"),
        ("within_well_spatial_variability.csv", "Within-well spatial CV"),
        ("between_well_variability.csv", "Between-well variability"),
        ("well_endpoint_changes.csv", "Within-well day17 changes"),
        ("selected_image_examples.csv", "Image-example selection rules"),
        ("image_display_clipping.csv", "Image display-clipping fractions"),
        ("measurement_provenance.json", "Measurement provenance"),
    ]:
        index += f'<li><a href="{file}">{label}</a></li>'
    index += "</ul><p>Spatial CV uses tiles with at least 50% valid area; zero-nucleus tiles have missing nuclear intensity. Valid tile counts are exported. Spatial tiles are nested within wells and are not biological replicates.</p></html>"
    (out / "report.html").write_text(index)
    for path, body in conditions:
        (path).write_text(body)
    for path in out.rglob("report.html"):
        write_html(path, path.read_text())
    root_report = root / "report.html"
    old = root_report.read_text()
    marker = "<!-- well-comparisons -->"
    if marker not in old:
        new = old.replace(
            "<h2>Reading the plots</h2>",
            marker
            + '<h2>Within- and between-well comparisons</h2><p><a href="comparisons/report.html"><strong>Open condition comparisons, spatial maps and image examples →</strong></a></p><h2>Reading the plots</h2>',
            1,
        )
        assert new != old and Links(old).targets <= Links(new).targets
        backup = Path(__file__).parents[1] / "plate_report_before_comparisons.html"
        if not backup.exists():
            backup.write_text(old)
        assert root_report.read_text() == old
        write_html(root_report, new)
    provenance = {
        "baseline_day": b,
        "well_pages": 96,
        "condition_pages": 4,
        "spatial_rows": len(tiles),
        "spatial_cv_rows": len(within),
        "between_well_rows": len(between),
        "selected_example_wells": selected.well.tolist(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(
            (root / "longitudinal_summary_metrics.csv").read_bytes()
        ).hexdigest(),
        "raw_spatial_color_limits": limits,
        "relative_spatial_color_limits": [0, 2],
        "all_report_links_exist": True,
        "statistics": "descriptive; sample SD / positive mean; no hypothesis tests",
    }
    (out / "report_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
