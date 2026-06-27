#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile as tif


DEFAULT_ABC_ROOT = Path("~/Documents/TMEM106B_processed/full_mcherry_valid_queue_abc")
DEFAULT_DEFG_ROOT = Path("~/Documents/TMEM106B_processed/full_mcherry_valid_defg_pass5")
DEFAULT_DASHBOARD_ROOT = Path("~/Documents/TMEM106B_processed/dashboard")
MCHERRY_VALID_ROWS = ("E", "F", "I", "J", "M", "N")
MCHERRY_VALID_COLUMNS = range(5, 21)
LOW_RETAINED_FRACTION = 0.50
VERY_LOW_RETAINED_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and document overlap-only dashboard behavior.")
    parser.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    parser.add_argument("--defg-root", type=Path, default=DEFAULT_DEFG_ROOT)
    parser.add_argument("--dashboard-root", type=Path, default=DEFAULT_DASHBOARD_ROOT)
    parser.add_argument("--wells", default="", help="Comma-separated wells. Empty means all 96 mCherry-valid wells.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    abc_root = args.abc_root.expanduser().resolve()
    defg_root = args.defg_root.expanduser().resolve()
    dashboard_root = args.dashboard_root.expanduser().resolve()
    wells = parse_wells(args.wells) if args.wells else all_mcherry_valid_wells()

    rows = []
    for well in wells:
        rows.append(build_well_record(well, abc_root=abc_root, defg_root=defg_root, dashboard_root=dashboard_root))
    summary = pd.DataFrame(rows).sort_values(["retained_fraction", "well"]).reset_index(drop=True)

    audit_root = abc_root / "overlap_only_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    csv_path = audit_root / "overlap_only_summary.csv"
    json_path = audit_root / "overlap_only_summary.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    for record in rows:
        write_well_metadata(record, abc_root=abc_root)

    dashboard_root.mkdir(parents=True, exist_ok=True)
    summary_page = dashboard_root / "overlap_only_qc_summary.html"
    summary_page.write_text(render_overlap_summary_page(summary, csv_path=csv_path, json_path=json_path), encoding="utf-8")
    pi_page = dashboard_root / "overlap_only_pi_summary.html"
    pi_page.write_text(render_pi_summary_page(summary), encoding="utf-8")
    inject_dashboard_index_link(dashboard_root)
    for record in rows:
        inject_well_overlap_section(dashboard_root, record)
    inject_roi_overlap_sections(dashboard_root, rows)

    print(f"Wrote overlap-only summary CSV: {csv_path}")
    print(f"Wrote overlap-only summary JSON: {json_path}")
    print(f"Wrote dashboard overlap summary: {summary_page}")
    print(f"Wrote PI-facing overlap summary: {pi_page}")
    print(f"Updated well pages under: {dashboard_root / 'wells'}")
    print(f"Updated ROI pages under: {dashboard_root / 'rois'}")


def parse_wells(value: str) -> list[str]:
    return [well.strip().upper() for well in value.split(",") if well.strip()]


def all_mcherry_valid_wells() -> list[str]:
    return [f"{row}{column:02d}" for row in MCHERRY_VALID_ROWS for column in MCHERRY_VALID_COLUMNS]


def build_well_record(well: str, *, abc_root: Path, defg_root: Path, dashboard_root: Path) -> dict[str, Any]:
    full = abc_root / "wells" / well / "registered_full" / f"{well}_registered_full_tcyx.ome.tif"
    common = abc_root / "wells" / well / "registered_common_overlap" / f"{well}_registered_common_overlap_tcyx.ome.tif"
    qc_csv = abc_root / "wells" / well / "qc" / f"{well}_registration_qc.csv"
    qc_montage = abc_root / "wells" / well / "qc" / f"{well}_registration_qc_montage.png"
    roi_csv = defg_root / "wells" / well / f"{well}_roi_candidates.csv"
    full_shape = tiff_shape(full)
    common_shape = tiff_shape(common)
    qc = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame()
    condition = str(qc["condition"].iloc[0]) if not qc.empty and "condition" in qc else condition_for_well(well)
    crop = overlap_crop_from_qc(full_shape[-2:], qc) if full_shape and not qc.empty else {}
    retained_fraction = retained_area_fraction(full_shape, common_shape)
    large_shift_days = int(qc["large_shift"].fillna(False).astype(bool).sum()) if "large_shift" in qc else 0
    review_or_exclude_days = review_or_exclude_from_qc(qc)
    roi_source = ""
    roi_count = 0
    if roi_csv.exists():
        rois = pd.read_csv(roi_csv)
        roi_count = int(len(rois))
        roi_source = str(rois["source_stack"].iloc[0]) if not rois.empty and "source_stack" in rois else ""
    overlap_warning = overlap_warning_for_fraction(retained_fraction)
    return {
        "well": well,
        "condition": condition,
        "days_included": "|".join(map(str, qc["day"].astype(int).tolist())) if "day" in qc else "",
        "review_or_exclude_days": "|".join(map(str, review_or_exclude_days)),
        "registered_full_stack": str(full),
        "registered_common_overlap_stack": str(common),
        "registration_qc_csv": str(qc_csv),
        "registration_qc_montage": str(qc_montage),
        "full_shape_tcyx": "x".join(map(str, full_shape)) if full_shape else "",
        "common_overlap_shape_tcyx": "x".join(map(str, common_shape)) if common_shape else "",
        "full_height": int(full_shape[-2]) if full_shape else 0,
        "full_width": int(full_shape[-1]) if full_shape else 0,
        "overlap_height": int(common_shape[-2]) if common_shape else 0,
        "overlap_width": int(common_shape[-1]) if common_shape else 0,
        "full_area_pixels": int(full_shape[-2] * full_shape[-1]) if full_shape else 0,
        "overlap_area_pixels": int(common_shape[-2] * common_shape[-1]) if common_shape else 0,
        "retained_fraction": retained_fraction,
        "retained_percent": retained_fraction * 100.0,
        "overlap_crop_y_start_registered": crop.get("y_start", ""),
        "overlap_crop_y_stop_registered": crop.get("y_stop", ""),
        "overlap_crop_x_start_registered": crop.get("x_start", ""),
        "overlap_crop_x_stop_registered": crop.get("x_stop", ""),
        "original_pixel_traceability": (
            "Overlap crop is stored in registered-frame coordinates. Per-day original-space mapping requires applying "
            "the inverse registration shift for each day."
        ),
        "registration_shift_summary": shift_summary(qc),
        "large_shift_days": large_shift_days,
        "min_day_overlap_fraction": float(qc["overlap_fraction"].min()) if "overlap_fraction" in qc and not qc.empty else 0.0,
        "qc_status": "review_low_overlap" if overlap_warning else "overlap_available",
        "overlap_warning": overlap_warning,
        "black_border_edge_regions_removed": bool(common.exists()),
        "output_level": "overlap_only_analysis",
        "recommended_use": "analysis_preferred" if not overlap_warning else "review_only",
        "roi_metrics_source": "registered_common_overlap" if "registered_common_overlap" in roi_source else "unknown_or_not_generated",
        "roi_metrics_overlap_only": bool("registered_common_overlap" in roi_source),
        "roi_count": roi_count,
        "dashboard_overlap_previews_dir": str(dashboard_root / "previews" / well),
        "dashboard_roi_previews_dir": str(dashboard_root / "roi_previews" / well),
    }


def tiff_shape(path: Path) -> tuple[int, ...]:
    if not path.exists():
        return ()
    with tif.TiffFile(path) as image:
        return tuple(int(value) for value in image.series[0].shape)


def overlap_crop_from_qc(shape_yx: tuple[int, int], qc: pd.DataFrame) -> dict[str, int]:
    if "estimated_y_shift" not in qc or "estimated_x_shift" not in qc:
        return {}
    shifts = [(float(row["estimated_y_shift"]), float(row["estimated_x_shift"])) for _, row in qc.iterrows()]
    height, width = shape_yx
    top = max(int(np.ceil(max(dy, 0))) for dy, _ in shifts)
    bottom = min(height + int(np.floor(min(dy, 0))) for dy, _ in shifts)
    left = max(int(np.ceil(max(dx, 0))) for _, dx in shifts)
    right = min(width + int(np.floor(min(dx, 0))) for _, dx in shifts)
    return {"y_start": top, "y_stop": bottom, "x_start": left, "x_stop": right}


def retained_area_fraction(full_shape: tuple[int, ...], common_shape: tuple[int, ...]) -> float:
    if not full_shape or not common_shape:
        return 0.0
    full_area = full_shape[-2] * full_shape[-1]
    common_area = common_shape[-2] * common_shape[-1]
    return float(common_area / full_area) if full_area else 0.0


def review_or_exclude_from_qc(qc: pd.DataFrame) -> list[int]:
    if qc.empty:
        return []
    mask = qc.get("large_shift", pd.Series(False, index=qc.index)).fillna(False).astype(bool)
    if "qc_pass" in qc:
        mask = mask | ~qc["qc_pass"].fillna(False).astype(bool)
    return [int(day) for day in qc.loc[mask, "day"].tolist()]


def overlap_warning_for_fraction(fraction: float) -> str:
    if fraction < VERY_LOW_RETAINED_FRACTION:
        return "very_low_overlap_retained"
    if fraction < LOW_RETAINED_FRACTION:
        return "low_overlap_retained"
    return ""


def shift_summary(qc: pd.DataFrame) -> str:
    if qc.empty or "estimated_y_shift" not in qc or "estimated_x_shift" not in qc:
        return ""
    rows = []
    for _, row in qc.iterrows():
        rows.append(
            f"day{int(row['day'])}:dy={float(row['estimated_y_shift']):.1f},dx={float(row['estimated_x_shift']):.1f}"
        )
    return "|".join(rows)


def condition_for_well(well: str) -> str:
    if well.startswith(("E", "I", "M")):
        return "PLD3_mCherry_reporter_control"
    if well.startswith(("F", "J", "N")):
        return "PLD3_TMEM106B_mCherry_primary"
    return "not_applicable_no_mCherry"


def write_well_metadata(record: dict[str, Any], *, abc_root: Path) -> None:
    out = abc_root / "wells" / record["well"] / "overlap_only_metadata" / f"{record['well']}_overlap_only_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tmem_overlap_only_metadata_v1",
        **record,
        "dashboard_note": (
            "Overlap-only views remove non-shared edge regions across days to reduce cells popping in/out due "
            "to field-of-view drift. Full registered views are retained for context but should be interpreted cautiously."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_overlap_summary_page(summary: pd.DataFrame, *, csv_path: Path, json_path: Path) -> str:
    worst = summary.sort_values("retained_fraction").head(20)
    best = summary.sort_values("retained_fraction", ascending=False).head(20)
    body = f"""
    <header>
      <h1>Overlap-Only QC Summary</h1>
      <p>Overlap-only views remove non-shared edge regions across days to reduce cells popping in/out due to field-of-view drift. Full registered views are retained for context but should be interpreted cautiously.</p>
      <nav><a href="index.html">Plate overview</a><a href="roi_review_queue.html">ROI review queue</a></nav>
    </header>
    <main>
      <section class="note">
        <h2>Audit Result</h2>
        <p>The existing <code>registered_common_overlap</code> stacks are the current overlap-only analysis branch. Dashboard day previews and pass5 ROI/metrics were generated from these stacks.</p>
        <p>Summary CSV: <code>{html.escape(str(csv_path))}</code></p>
        <p>Summary JSON: <code>{html.escape(str(json_path))}</code></p>
      </section>
      <section>
        <h2>Lowest Retained Overlap</h2>
        {render_table(worst)}
      </section>
      <section>
        <h2>Highest Retained Overlap</h2>
        {render_table(best)}
      </section>
    </main>
    """
    return render_page("Overlap-Only QC Summary", body)


def render_pi_summary_page(summary: pd.DataFrame) -> str:
    roi_source_count = int((summary["roi_metrics_source"] == "registered_common_overlap").sum())
    analysis_preferred = int((summary["recommended_use"] == "analysis_preferred").sum())
    review_only = int((summary["recommended_use"] == "review_only").sum())
    best = ", ".join(summary.sort_values("retained_fraction", ascending=False)["well"].head(9).tolist())
    worst = ", ".join(summary.sort_values("retained_fraction")["well"].head(10).tolist())
    body = f"""
    <header>
      <h1>PI Summary: Overlap-Only Response</h1>
      <p>How the TMEM106B time-series workflow addresses cells appearing or disappearing at field edges after alignment.</p>
      <nav><a href="index.html">Plate overview</a><a href="overlap_only_qc_summary.html">Overlap-only QC summary</a><a href="roi_review_queue.html">ROI review queue</a></nav>
    </header>
    <main>
      <section class="note">
        <h2>Short Answer</h2>
        <p>The pipeline already creates <code>registered_common_overlap</code>, the shared intersection region across all registered days for each well. Current dashboard day previews and pass5 ROI/metrics use this common-overlap region, not the full registered frame.</p>
        <p>This removes non-shared edge regions before ROI generation and quantification, which should reduce cells popping in or out because of day-to-day field-of-view drift. Full registered frames are retained for context only.</p>
      </section>
      <section>
        <h2>Why This Matters</h2>
        <p>The observed popping artifact is most consistent with edge-of-field effects caused by day-to-day plate/FOV shifts, focus differences, registration limits, or distortions. If the full registered frame is used, cells near the border can enter or leave the visible field even when the central overlap is stable.</p>
      </section>
      <section>
        <h2>Pipeline Response</h2>
        <div class="schematic">
          <div>Full registered field across days<br /><small>context only; may include non-shared edges</small></div>
          <div class="arrow">-&gt;</div>
          <div>Intersection/common-overlap crop<br /><small>shared region across all registered days</small></div>
          <div class="arrow">-&gt;</div>
          <div>Artifact-reduced previews, ROIs, and metrics<br /><small>preferred for analysis</small></div>
        </div>
      </section>
      <section>
        <h2>Current Audit Numbers</h2>
        <ul>
          <li>{roi_source_count}/96 wells have ROI/metrics source = <code>registered_common_overlap</code>.</li>
          <li>{analysis_preferred} wells retain at least 50% overlap and are labeled <code>analysis_preferred</code>.</li>
          <li>{review_only} wells are labeled <code>review_only</code> under the strict retained-area threshold. This does not mean they are unusable; it means they need visual review or stricter interpretation before biological use.</li>
          <li>Best retained-overlap examples: {html.escape(best)}.</li>
          <li>Worst retained-overlap examples: {html.escape(worst)}.</li>
        </ul>
      </section>
      <section>
        <h2>Interpretation Boundary</h2>
        <p>Overlap-only cropping reduces edge pop-in/pop-out risk, but it does not solve poor registration, focus problems, or manual same-neuron identity uncertainty. Biological claims remain preliminary until ROI identity review is complete.</p>
      </section>
    </main>
    """
    return render_page("PI Summary: Overlap-Only Response", body)


def inject_dashboard_index_link(dashboard_root: Path) -> None:
    page = dashboard_root / "index.html"
    if not page.exists():
        return
    section = """
<!-- OVERLAP_ONLY_SECTION_START -->
<section>
  <h2>Overlap-Only Analysis Views</h2>
  <p>Overlap-only views remove non-shared edge regions across days to reduce cells popping in/out due to field-of-view drift. Full registered views are retained for context but should be interpreted cautiously.</p>
  <p><a href="overlap_only_pi_summary.html">Open PI-facing overlap-only summary</a></p>
  <p><a href="overlap_only_qc_summary.html">Open overlap-only QC summary</a></p>
</section>
<!-- OVERLAP_ONLY_SECTION_END -->
"""
    replace_marked_section(page, section, "OVERLAP_ONLY_SECTION_START", "OVERLAP_ONLY_SECTION_END")


def inject_well_overlap_section(dashboard_root: Path, record: dict[str, Any]) -> None:
    well = record["well"]
    page = dashboard_root / "wells" / f"{well}.html"
    if not page.exists():
        return
    warning = record["overlap_warning"] or "none"
    section = f"""
<!-- OVERLAP_ONLY_WELL_SECTION_START -->
<section>
  <h2>Full Registered vs Overlap-Only</h2>
  <p>Overlap-only views remove non-shared edge regions across days to reduce cells popping in/out due to field-of-view drift. Full registered views are retained for context but should be interpreted cautiously.</p>
  <dl>
    <dt>Full registered context</dt><dd>{html.escape(record['registered_full_stack'])}</dd>
    <dt>Overlap-only analysis stack</dt><dd>{html.escape(record['registered_common_overlap_stack'])}</dd>
    <dt>Overlap retained</dt><dd>{record['retained_percent']:.1f}% ({record['overlap_width']} x {record['overlap_height']} px from {record['full_width']} x {record['full_height']} px)</dd>
    <dt>Overlap warning</dt><dd>{html.escape(warning)}</dd>
    <dt>Large-shift days</dt><dd>{record['large_shift_days']}</dd>
    <dt>ROI/metrics source</dt><dd>{html.escape(record['roi_metrics_source'])}</dd>
    <dt>Recommended use</dt><dd>{html.escape(record['recommended_use'])}</dd>
  </dl>
</section>
<!-- OVERLAP_ONLY_WELL_SECTION_END -->
"""
    replace_marked_section(page, section, "OVERLAP_ONLY_WELL_SECTION_START", "OVERLAP_ONLY_WELL_SECTION_END")


def inject_roi_overlap_sections(dashboard_root: Path, records: list[dict[str, Any]]) -> None:
    by_well = {record["well"]: record for record in records}
    for page in (dashboard_root / "rois").glob("*/*.html"):
        match = re.match(r"([A-Z]\d{2})_ROI\d+\.html$", page.name)
        if not match:
            continue
        record = by_well.get(match.group(1))
        if not record:
            continue
        section = f"""
<!-- OVERLAP_ONLY_ROI_SECTION_START -->
<section>
  <h2>Overlap-Only ROI Source</h2>
  <p>This ROI candidate was generated from the overlap-only common-overlap stack when pass5 source metadata points to <code>registered_common_overlap</code>.</p>
  <dl>
    <dt>Output level</dt><dd>overlap_only_analysis</dd>
    <dt>Well retained overlap</dt><dd>{record['retained_percent']:.1f}%</dd>
    <dt>Full-registered crop box</dt><dd>x={record['overlap_crop_x_start_registered']}:{record['overlap_crop_x_stop_registered']}, y={record['overlap_crop_y_start_registered']}:{record['overlap_crop_y_stop_registered']}</dd>
    <dt>Traceability</dt><dd>{html.escape(record['original_pixel_traceability'])}</dd>
  </dl>
</section>
<!-- OVERLAP_ONLY_ROI_SECTION_END -->
"""
        replace_marked_section(page, section, "OVERLAP_ONLY_ROI_SECTION_START", "OVERLAP_ONLY_ROI_SECTION_END")


def replace_marked_section(path: Path, section: str, start_name: str, end_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    start_marker = f"<!-- {start_name} -->"
    end_marker = f"<!-- {end_name} -->"
    if start_marker in text and end_marker in text:
        before = text.split(start_marker, 1)[0]
        after = text.split(end_marker, 1)[1]
        text = before + section.strip() + after
    elif "</main>" in text:
        text = text.replace("</main>", section + "\n</main>", 1)
    elif "</body>" in text:
        text = text.replace("</body>", section + "\n</body>", 1)
    else:
        text += "\n" + section
    path.write_text(text, encoding="utf-8")


def render_table(table: pd.DataFrame) -> str:
    display = table[
        [
            "well",
            "condition",
            "retained_percent",
            "overlap_warning",
            "large_shift_days",
            "min_day_overlap_fraction",
            "roi_metrics_source",
            "recommended_use",
        ]
    ].copy()
    rows = []
    for _, row in display.iterrows():
        rows.append(
            "<tr>"
            f"<td><a href=\"wells/{html.escape(str(row['well']))}.html\">{html.escape(str(row['well']))}</a></td>"
            f"<td>{html.escape(str(row['condition']))}</td>"
            f"<td>{float(row['retained_percent']):.1f}%</td>"
            f"<td>{html.escape(str(row['overlap_warning']) or 'none')}</td>"
            f"<td>{int(row['large_shift_days'])}</td>"
            f"<td>{float(row['min_day_overlap_fraction']):.3f}</td>"
            f"<td>{html.escape(str(row['roi_metrics_source']))}</td>"
            f"<td>{html.escape(str(row['recommended_use']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Well</th><th>Condition</th><th>Retained</th><th>Warning</th>"
        "<th>Large-shift days</th><th>Min day overlap</th><th>ROI source</th><th>Use</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f8fa; color: #20242a; }}
    header, main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    nav {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dde6; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #edf0f4; vertical-align: top; }}
    th {{ background: #f1f4f8; }}
    dl {{ display: grid; grid-template-columns: 220px 1fr; gap: 8px 12px; background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 16px; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .note {{ background: #fff8e6; border: 1px solid #ead59b; border-radius: 8px; padding: 14px 16px; }}
    .schematic {{ display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 12px; align-items: stretch; }}
    .schematic div {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 14px; }}
    .schematic .arrow {{ display: grid; align-items: center; border: 0; background: transparent; font-weight: 700; }}
    small {{ color: #5b6470; }}
  </style>
</head>
<body>{body}</body>
</html>
"""


if __name__ == "__main__":
    main()
