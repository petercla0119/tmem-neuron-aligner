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
SITE_CSS = r"""
:root {
  color-scheme: light;
  --ink: #1f2933;
  --muted: #59636f;
  --line: #d8dde6;
  --panel: #ffffff;
  --bg: #f6f8fb;
  --control: #dff3ec;
  --tmem: #e8edf8;
  --preferred: #dff3ec;
  --review: #fff3cf;
  --manual: #ffe3d3;
  --pseudo: #ffdede;
}
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header, main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  margin: 0 0 8px;
  font-size: 32px;
  letter-spacing: 0;
}
h2 {
  margin-top: 30px;
}
a {
  color: #22577a;
}
nav {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 12px;
}
nav a, .button-link {
  display: inline-block;
  padding: 7px 10px;
  border: 1px solid #22577a;
  border-radius: 6px;
  background: #fff;
  text-decoration: none;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border: 1px solid var(--line);
  font-size: 13px;
}
th, td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid #edf0f4;
  vertical-align: top;
}
th {
  background: #f1f4f8;
}
dl {
  display: grid;
  grid-template-columns: minmax(150px, 220px) 1fr;
  gap: 8px 12px;
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
dd {
  margin: 0;
  overflow-wrap: anywhere;
}
img {
  max-width: 100%;
}
.site-header p, .mini-note, small {
  color: var(--muted);
}
.viewer-panel, .well-viewer section {
  background: transparent;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: end;
  margin: 10px 0 16px;
}
.toolbar input, .toolbar select {
  display: block;
  min-width: 160px;
  margin-top: 4px;
  padding: 7px 8px;
  border: 1px solid #bcc6d4;
  border-radius: 6px;
}
.plate-grid {
  display: grid;
  grid-template-columns: 34px repeat(var(--plate-columns), minmax(38px, 1fr));
  gap: 5px;
  align-items: stretch;
}
.plate-corner, .plate-col-label, .plate-row-label {
  display: grid;
  place-items: center;
  min-height: 28px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.plate-well {
  min-height: 58px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 5px;
  text-decoration: none;
  color: inherit;
  display: grid;
  gap: 2px;
  align-content: start;
  font-size: 11px;
  overflow: hidden;
}
.plate-well strong {
  font-size: 13px;
}
.plate-well em {
  font-size: 10px;
  color: var(--muted);
  font-style: normal;
}
.plate-well.inactive {
  background: #e5e7eb;
  color: #8a929d;
  opacity: 0.75;
}
.plate-well.reporter-control {
  background: var(--control);
}
.plate-well.tmem106b {
  background: var(--tmem);
}
.plate-well.analysis-preferred {
  box-shadow: inset 0 -5px 0 #2f7d32;
}
.plate-well.review-only {
  box-shadow: inset 0 -5px 0 #b7791f;
}
.plate-well.manual-alignment-needed {
  box-shadow: inset 0 -5px 0 #d96c06;
}
.plate-well.pseudo-fov-alignment-needed {
  box-shadow: inset 0 -5px 0 #b42318;
}
.legend-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
  margin-top: 16px;
}
.legend-chip {
  display: inline-block;
  margin: 4px 6px 4px 0;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  font-size: 12px;
}
.legend-chip.reporter-control { background: var(--control); }
.legend-chip.tmem106b { background: var(--tmem); }
.legend-chip.inactive { background: #e5e7eb; color: #68717d; }
.legend-chip.analysis-preferred { border-bottom: 5px solid #2f7d32; }
.legend-chip.review-only { border-bottom: 5px solid #b7791f; }
.legend-chip.manual-alignment-needed { border-bottom: 5px solid #d96c06; }
.legend-chip.pseudo-fov-alignment-needed { border-bottom: 5px solid #b42318; }
.caution, .warning-banner, .note, .callout {
  background: #fff8e6;
  border: 1px solid #ead59b;
  border-radius: 8px;
  padding: 12px 14px;
}
.warning-banner {
  background: #fff1eb;
  border-color: #f0b28d;
}
.time-series-viewer {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.viewer-toolbar, .day-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;
}
.viewer-toolbar button, .day-controls button, .viewer-toolbar select {
  padding: 7px 10px;
  border: 1px solid #bcc6d4;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
}
.viewer-toolbar label {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 13px;
}
.day-controls button.active {
  background: #22577a;
  color: white;
  border-color: #22577a;
}
.time-series-frame {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 340px;
  background: #111;
  border-radius: 8px;
  overflow: hidden;
}
.time-series-frame img {
  display: none;
  width: 100%;
  max-height: 720px;
  object-fit: contain;
  margin: 0;
  border: 0;
  background: #111;
}
.time-series-frame img.active {
  display: block;
}
.time-series-viewer.onion .time-series-frame img.previous {
  display: block;
  position: absolute;
  opacity: 0.35;
  filter: grayscale(1);
}
.time-series-viewer.onion .time-series-frame img.active {
  position: relative;
  z-index: 2;
}
.time-series-montage {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
figure {
  margin: 0;
  padding: 10px;
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
}
figcaption {
  margin-top: 8px;
  font-size: 13px;
  color: var(--muted);
}
.roi-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 14px;
}
.roi-viewer-card {
  background: white;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.roi-viewer-card.has-warning {
  border-left: 6px solid #b7791f;
}
.roi-card-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}
.roi-card-header h3 {
  margin: 0;
  font-size: 22px;
}
.roi-viewer-card img {
  width: 100%;
  height: auto;
  border: 1px solid #edf0f4;
  background: #111;
}
.roi-viewer-card dl {
  grid-template-columns: 125px 1fr;
  font-size: 13px;
  padding: 12px;
}
.roi-detail-preview img {
  display: block;
  width: 100%;
  background: #111;
}
tr.warning-large-shift, tr.review {
  background: #fff8e6;
}
tr.warning-extreme-shift {
  background: #fff0ef;
}
.hidden-by-filter {
  opacity: 0.14;
}
@media (max-width: 760px) {
  header, main {
    padding: 16px;
  }
  .plate-grid {
    grid-template-columns: 28px repeat(var(--plate-columns), minmax(26px, 1fr));
    gap: 3px;
  }
  .plate-well {
    min-height: 40px;
    padding: 3px;
  }
  .plate-well span, .plate-well small, .plate-well em {
    display: none;
  }
  dl {
    grid-template-columns: 1fr;
  }
}
"""

VIEWER_JS = r"""
(function () {
  function initViewer(viewer) {
    const frames = Array.from(viewer.querySelectorAll(".viewer-frame"));
    if (!frames.length) return;
    const slider = viewer.querySelector("[data-viewer-slider]");
    const label = viewer.querySelector("[data-viewer-label]");
    const playButton = viewer.querySelector('[data-viewer-action="play"]');
    const blinkButton = viewer.querySelector('[data-viewer-action="blink"]');
    const onionButton = viewer.querySelector('[data-viewer-action="onion"]');
    const speed = viewer.querySelector("[data-viewer-speed]");
    const opacity = viewer.querySelector("[data-viewer-opacity]");
    const dayButtons = Array.from(viewer.querySelectorAll("[data-day-index]"));
    let index = 0;
    let timer = null;
    let blink = false;
    let blinkToggle = false;
    function show(nextIndex) {
      index = (nextIndex + frames.length) % frames.length;
      frames.forEach((frame, frameIndex) => {
        frame.classList.toggle("active", frameIndex === index);
        frame.classList.toggle("previous", frameIndex === ((index - 1 + frames.length) % frames.length));
        frame.style.opacity = frameIndex === index ? (opacity ? opacity.value : "1") : "";
      });
      dayButtons.forEach((button, buttonIndex) => button.classList.toggle("active", buttonIndex === index));
      if (slider) slider.value = String(index);
      if (label) label.textContent = "Day " + (frames[index].dataset.day || String(index + 1));
    }
    function step() {
      if (blink && frames.length > 1) {
        blinkToggle = !blinkToggle;
        show(blinkToggle ? index : index + 1);
      } else {
        show(index + 1);
      }
    }
    function stop() {
      if (timer) window.clearInterval(timer);
      timer = null;
      if (playButton) playButton.textContent = "Play";
    }
    function play() {
      stop();
      timer = window.setInterval(step, Number(speed ? speed.value : 700));
      if (playButton) playButton.textContent = "Pause";
    }
    viewer.addEventListener("click", function (event) {
      const target = event.target.closest("button");
      if (!target) return;
      if (target.dataset.dayIndex) {
        stop();
        show(Number(target.dataset.dayIndex));
      }
      const action = target.dataset.viewerAction;
      if (action === "prev") {
        stop();
        show(index - 1);
      }
      if (action === "next") {
        stop();
        show(index + 1);
      }
      if (action === "play") {
        timer ? stop() : play();
      }
      if (action === "blink") {
        blink = !blink;
        target.textContent = blink ? "Blink on" : "Blink off";
      }
      if (action === "onion") {
        viewer.classList.toggle("onion");
        target.textContent = viewer.classList.contains("onion") ? "Onion skin on" : "Onion skin off";
      }
    });
    if (slider) {
      slider.addEventListener("input", function () {
        stop();
        show(Number(slider.value));
      });
    }
    if (opacity) {
      opacity.addEventListener("input", function () {
        frames[index].style.opacity = opacity.value;
      });
    }
    if (speed) {
      speed.addEventListener("change", function () {
        if (timer) play();
      });
    }
    show(0);
  }
  function initPlateFilters() {
    const search = document.querySelector("[data-plate-search]");
    const condition = document.querySelector('[data-plate-filter="condition"]');
    const qc = document.querySelector('[data-plate-filter="qc"]');
    const wells = Array.from(document.querySelectorAll(".plate-well.active"));
    function applyFilters() {
      const query = search ? search.value.trim().toUpperCase() : "";
      const conditionValue = condition ? condition.value : "";
      const qcValue = qc ? qc.value : "";
      wells.forEach((well) => {
        const wellId = well.dataset.well || "";
        const cond = well.dataset.condition || "";
        const qcLabel = well.dataset.qc || "";
        const matchesSearch = !query || wellId.includes(query);
        const matchesCondition = !conditionValue ||
          (conditionValue === "reporter_control" && cond.includes("reporter_control")) ||
          (conditionValue === "tmem106b" && cond.includes("TMEM106B"));
        const matchesQc = !qcValue || qcLabel === qcValue;
        well.classList.toggle("hidden-by-filter", !(matchesSearch && matchesCondition && matchesQc));
      });
    }
    [search, condition, qc].forEach((control) => {
      if (control) control.addEventListener("input", applyFilters);
      if (control) control.addEventListener("change", applyFilters);
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-tmem-viewer]").forEach(initViewer);
    initPlateFilters();
  });
})();
"""


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
    roi_dashboard = load_roi_dashboard_data(defg_root=defg_root, dashboard_root=dashboard_root)

    audit_root = abc_root / "overlap_only_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    csv_path = audit_root / "overlap_only_summary.csv"
    json_path = audit_root / "overlap_only_summary.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    for record in rows:
        write_well_metadata(record, abc_root=abc_root)

    dashboard_root.mkdir(parents=True, exist_ok=True)
    write_static_assets(dashboard_root)
    summary_page = dashboard_root / "overlap_only_qc_summary.html"
    summary_page.write_text(render_overlap_summary_page(summary, csv_path=csv_path, json_path=json_path), encoding="utf-8")
    pi_page = dashboard_root / "overlap_only_pi_summary.html"
    pi_page.write_text(render_pi_summary_page(summary), encoding="utf-8")
    alignment_page = dashboard_root / "alignment_qc_review.html"
    alignment_page.write_text(render_alignment_qc_review_page(summary), encoding="utf-8")
    inject_dashboard_index_link(dashboard_root, summary, roi_dashboard)
    for record in rows:
        inject_well_viewer_section(dashboard_root, record, roi_dashboard.get(record["well"], []))
    inject_roi_overlap_sections(dashboard_root, rows, roi_dashboard)

    print(f"Wrote overlap-only summary CSV: {csv_path}")
    print(f"Wrote overlap-only summary JSON: {json_path}")
    print(f"Wrote dashboard overlap summary: {summary_page}")
    print(f"Wrote PI-facing overlap summary: {pi_page}")
    print(f"Wrote alignment QC review page: {alignment_page}")
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
    max_shift_magnitude = max_shift_from_qc(qc)
    review_or_exclude_days = review_or_exclude_from_qc(qc)
    roi_source = ""
    roi_count = 0
    if roi_csv.exists():
        rois = pd.read_csv(roi_csv)
        roi_count = int(len(rois))
        roi_source = str(rois["source_stack"].iloc[0]) if not rois.empty and "source_stack" in rois else ""
    overlap_warning = overlap_warning_for_fraction(retained_fraction)
    alignment_category = alignment_category_for_well(retained_fraction, large_shift_days, max_shift_magnitude)
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
        "registration_qc_table": qc_table_records(qc),
        "large_shift_days": large_shift_days,
        "max_shift_magnitude": max_shift_magnitude,
        "min_day_overlap_fraction": float(qc["overlap_fraction"].min()) if "overlap_fraction" in qc and not qc.empty else 0.0,
        "qc_status": "review_low_overlap" if overlap_warning else "overlap_available",
        "overlap_warning": overlap_warning,
        "black_border_edge_regions_removed": bool(common.exists()),
        "output_level": "overlap_only_analysis",
        "recommended_use": recommended_use_for_alignment(alignment_category, retained_fraction),
        "alignment_qc_category": alignment_category,
        "manual_alignment_review": alignment_category in {"manual_alignment_needed", "pseudo_fov_alignment_needed"},
        "pseudo_fov_alignment_review": alignment_category == "pseudo_fov_alignment_needed",
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


def max_shift_from_qc(qc: pd.DataFrame) -> float:
    if qc.empty or "estimated_y_shift" not in qc or "estimated_x_shift" not in qc:
        return 0.0
    magnitudes = np.sqrt(qc["estimated_y_shift"].astype(float) ** 2 + qc["estimated_x_shift"].astype(float) ** 2)
    return float(magnitudes.max()) if len(magnitudes) else 0.0


def qc_table_records(qc: pd.DataFrame) -> list[dict[str, Any]]:
    if qc.empty:
        return []
    rows = []
    for _, row in qc.iterrows():
        dy = float(row.get("estimated_y_shift", 0.0))
        dx = float(row.get("estimated_x_shift", 0.0))
        rows.append(
            {
                "day": int(row["day"]),
                "dy": dy,
                "dx": dx,
                "shift_magnitude": float(np.sqrt(dy**2 + dx**2)),
                "overlap_fraction": float(row.get("overlap_fraction", 0.0)),
                "large_shift": bool(row.get("large_shift", False)),
                "max_shift_exceeded": bool(row.get("max_shift_exceeded", False)),
                "qc_pass": bool(row.get("qc_pass", False)),
                "registration_channel": str(row.get("registration_channel", "")),
                "qc_note": str(row.get("qc_note", "")),
            }
        )
    return rows


def alignment_category_for_well(retained_fraction: float, large_shift_days: int, max_shift_magnitude: float) -> str:
    if retained_fraction < VERY_LOW_RETAINED_FRACTION:
        return "pseudo_fov_alignment_needed"
    if large_shift_days >= 5 or max_shift_magnitude >= 700:
        return "manual_alignment_needed"
    if retained_fraction < LOW_RETAINED_FRACTION or large_shift_days:
        return "review_only"
    return "analysis_preferred"


def recommended_use_for_alignment(alignment_category: str, retained_fraction: float) -> str:
    if alignment_category == "analysis_preferred":
        return "analysis_preferred"
    if alignment_category == "pseudo_fov_alignment_needed":
        return "pseudo_fov_alignment_needed"
    if alignment_category == "manual_alignment_needed":
        return "manual_alignment_needed"
    if retained_fraction <= 0:
        return "context_only"
    return "review_only"


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


def load_roi_dashboard_data(*, defg_root: Path, dashboard_root: Path) -> dict[str, list[dict[str, Any]]]:
    label = "full_mcherry_valid_pass5"
    roi_csv = defg_root / f"{label}_roi_candidates.csv"
    metrics_csv = defg_root / f"{label}_mcherry_metrics.csv"
    coloc_csv = defg_root / f"{label}_colocalization_metrics.csv"
    review_csv = defg_root / "review_summaries" / f"{label}_roi_identity_review_working.csv"
    if not roi_csv.exists():
        return {}
    rois = pd.read_csv(roi_csv)
    metrics = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()
    coloc = pd.read_csv(coloc_csv) if coloc_csv.exists() else pd.DataFrame()
    review = pd.read_csv(review_csv) if review_csv.exists() else pd.DataFrame()
    metric_summary = summarize_mcherry_by_roi(metrics)
    coloc_summary = summarize_colocalization_by_roi(coloc)
    review_summary = review[["roi_id", "manual_identity_status", "reviewer_notes"]] if "roi_id" in review else pd.DataFrame()
    merged = rois.merge(metric_summary, on="roi_id", how="left")
    merged = merged.merge(coloc_summary, on="roi_id", how="left")
    if not review_summary.empty:
        merged = merged.drop(columns=[column for column in ["manual_identity_status", "reviewer_notes"] if column in merged], errors="ignore")
        merged = merged.merge(review_summary, on="roi_id", how="left")
    records: dict[str, list[dict[str, Any]]] = {}
    for row in merged.to_dict("records"):
        well = str(row["well"])
        roi_id = str(row["roi_id"])
        row["dashboard_preview_rel_from_well"] = f"../roi_previews/{well}/{roi_id}.png"
        row["dashboard_preview_path"] = str(dashboard_root / "roi_previews" / well / f"{roi_id}.png")
        row["roi_detail_rel_from_well"] = f"../rois/{well}/{roi_id}.html"
        row["overlap_only_source_label"] = (
            "overlap-only/common-overlap source" if "registered_common_overlap" in str(row.get("source_stack", "")) else "source review needed"
        )
        records.setdefault(well, []).append(row)
    for well in records:
        records[well] = sorted(records[well], key=lambda row: str(row["roi_id"]))
    return records


def summarize_mcherry_by_roi(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "roi_id" not in metrics:
        return pd.DataFrame(columns=["roi_id"])
    summary = metrics.groupby("roi_id", as_index=False).agg(
        mean_puncta_count=("puncta_count", "mean"),
        mean_punctate_fraction=("punctate_fraction", "mean"),
        mean_diffuse_to_punctate_ratio=("diffuse_to_punctate_ratio", "mean"),
        mcherry_metric_rows=("day", "count"),
    )
    return summary


def summarize_colocalization_by_roi(coloc: pd.DataFrame) -> pd.DataFrame:
    if coloc.empty or "roi_id" not in coloc:
        return pd.DataFrame(columns=["roi_id"])
    summary = coloc.groupby("roi_id", as_index=False).agg(
        mean_pearson_correlation=("pearson_correlation", "mean"),
        mean_manders_a_in_b=("manders_a_in_b", "mean"),
        mean_manders_b_in_a=("manders_b_in_a", "mean"),
        colocalization_rows=("day", "count"),
        saturation_warning_rows=("warnings", lambda values: int(values.fillna("").str.contains("saturation|clipping", case=False, regex=True).sum())),
    )
    warnings = coloc.groupby("roi_id")["warnings"].apply(join_warnings).rename("colocalization_warnings")
    return summary.merge(warnings, on="roi_id", how="left")


def join_warnings(values: pd.Series) -> str:
    warnings = sorted({str(value).strip() for value in values.dropna() if str(value).strip() and str(value).lower() != "nan"})
    return "; ".join(warnings)


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


def inject_dashboard_index_link(dashboard_root: Path, summary: pd.DataFrame, roi_dashboard: dict[str, list[dict[str, Any]]]) -> None:
    page = dashboard_root / "index.html"
    page.write_text(render_plate_index_page(summary, roi_dashboard), encoding="utf-8")


def render_plate_index_page(summary: pd.DataFrame, roi_dashboard: dict[str, list[dict[str, Any]]]) -> str:
    records = {str(row["well"]): row for _, row in summary.iterrows()}
    active_wells = sorted(records)
    rows, columns = infer_plate_layout(active_wells)
    plate_cells = []
    for row_label in rows:
        plate_cells.append(f'<div class="plate-row-label">{row_label}</div>')
        for column in columns:
            well = f"{row_label}{column:02d}"
            record = records.get(well)
            if record is None:
                plate_cells.append(f'<div class="plate-well inactive" data-well="{well}" aria-label="{well} inactive">{well}</div>')
                continue
            roi_records = roi_dashboard.get(well, [])
            statuses = [str(row.get("manual_identity_status", "uncertain_identity") or "uncertain_identity") for row in roi_records]
            confirmed = statuses.count("confirmed_same_neuron")
            uncertain = statuses.count("uncertain_identity")
            excluded = sum(1 for status in statuses if status in {"exclude", "poor_registration", "lost_or_dead"})
            condition_class = condition_class_for_record(str(record["condition"]))
            qc_class = css_token(str(record.get("alignment_qc_category", record.get("recommended_use", "review_only"))))
            manual_note = "manual/pseudo-FOV review" if bool(record.get("manual_alignment_review")) else "standard visual review"
            plate_cells.append(
                f'<a class="plate-well active {condition_class} {qc_class}" '
                f'href="wells/{html.escape(well)}.html" data-well="{html.escape(well)}" '
                f'data-condition="{html.escape(str(record["condition"]))}" data-qc="{html.escape(str(record.get("alignment_qc_category", "")))}">'
                f'<strong>{html.escape(well)}</strong>'
                f'<span>{len(roi_records)} ROI</span>'
                f'<small>{float(record["retained_percent"]):.0f}% overlap</small>'
                f'<small>{confirmed}/{uncertain}/{excluded}</small>'
                f'<em>{html.escape(manual_note)}</em>'
                "</a>"
            )
    column_labels = '<div class="plate-corner"></div>' + "".join(f'<div class="plate-col-label">{column:02d}</div>' for column in columns)
    table_rows = []
    for _, record in summary.sort_values(["alignment_qc_category", "retained_fraction", "well"], ascending=[True, False, True]).iterrows():
        well = str(record["well"])
        roi_records = roi_dashboard.get(well, [])
        statuses = [str(row.get("manual_identity_status", "uncertain_identity") or "uncertain_identity") for row in roi_records]
        confirmed = statuses.count("confirmed_same_neuron")
        uncertain = statuses.count("uncertain_identity")
        excluded = sum(1 for status in statuses if status in {"exclude", "poor_registration", "lost_or_dead"})
        table_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(well)}</strong></td>"
            f"<td>{html.escape(str(record['condition']))}</td>"
            f"<td>{html.escape(str(record.get('alignment_qc_category', 'review_only')))}</td>"
            f"<td>{html.escape(str(record.get('recommended_use', 'review_only')))}<br><small>{float(record['retained_percent']):.1f}% retained</small></td>"
            f"<td>{int(record.get('large_shift_days', 0))}<br><small>max {float(record.get('max_shift_magnitude', 0.0)):.1f} px</small></td>"
            f"<td>{len(roi_records)}</td>"
            f"<td>{confirmed}</td>"
            f"<td>{uncertain}</td>"
            f"<td>{excluded}</td>"
            f'<td><a class="button-link" href="wells/{html.escape(well)}.html">Open Well Viewer</a></td>'
            "</tr>"
        )
    body = f"""
    <header class="site-header">
      <h1>TMEM106B Plate Viewer</h1>
      <p>384-well-style layout with the 96 mCherry-valid wells active in their detected row/column positions.</p>
      <nav><a href="alignment_qc_review.html">Alignment QC review</a><a href="roi_review_queue.html">ROI review queue</a><a href="overlap_only_pi_summary.html">PI overlap summary</a></nav>
    </header>
    <main>
      <section class="viewer-panel">
        <h2>Plate-Style Home Screen</h2>
        <div class="toolbar">
          <label>Search well <input id="well-search" type="search" placeholder="J19" data-plate-search /></label>
          <label>Condition <select data-plate-filter="condition"><option value="">All</option><option value="reporter_control">Reporter controls</option><option value="tmem106b">TMEM106B + mCherry</option></select></label>
          <label>QC <select data-plate-filter="qc"><option value="">All</option><option value="analysis_preferred">Analysis preferred</option><option value="review_only">Review only</option><option value="manual_alignment_needed">Manual alignment needed</option><option value="pseudo_fov_alignment_needed">Pseudo-FOV needed</option></select></label>
        </div>
        <div class="plate-grid" style="--plate-columns: {len(columns)}">
          {column_labels}
          {''.join(plate_cells)}
        </div>
        <div class="legend-grid">
          <div><h3>Condition</h3><span class="legend-chip reporter-control">Reporter controls</span><span class="legend-chip tmem106b">TMEM106B + mCherry</span><span class="legend-chip inactive">Inactive / not public</span></div>
          <div><h3>QC / alignment</h3><span class="legend-chip analysis-preferred">analysis_preferred</span><span class="legend-chip review-only">review_only</span><span class="legend-chip manual-alignment-needed">manual_alignment_needed</span><span class="legend-chip pseudo-fov-alignment-needed">pseudo_fov_alignment_needed</span></div>
        </div>
      </section>
      <section class="viewer-panel">
        <h2>Well Table</h2>
        <p>Open a well to review the same-frame overlap-only time course and its separated neuron/ROI candidates. Metrics remain preliminary until ROI identity is manually reviewed.</p>
        <table>
          <thead>
            <tr><th>Well</th><th>Condition</th><th>Alignment QC</th><th>Recommendation</th><th>Shift warnings</th><th>ROIs</th><th>Confirmed</th><th>Uncertain</th><th>Excluded/flagged</th><th>Viewer</th></tr>
          </thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </section>
    </main>
    <script src="assets/viewer.js"></script>
    """
    return render_page("TMEM106B Plate Viewer", body, asset_prefix="")


def infer_plate_layout(wells: list[str]) -> tuple[list[str], list[int]]:
    parsed = [(well[0], int(well[1:])) for well in wells if re.match(r"^[A-Z]\d{2}$", well)]
    max_row = max((ord(row) - ord("A") + 1 for row, _ in parsed), default=8)
    max_col = max((column for _, column in parsed), default=12)
    if max_row > 8 or max_col > 12:
        return [chr(ord("A") + index) for index in range(16)], list(range(1, 25))
    return [chr(ord("A") + index) for index in range(8)], list(range(1, 13))


def condition_class_for_record(condition: str) -> str:
    if "TMEM106B" in condition:
        return "tmem106b"
    if "reporter_control" in condition:
        return "reporter-control"
    return "not-applicable"


def css_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def inject_well_viewer_section(dashboard_root: Path, record: dict[str, Any], roi_records: list[dict[str, Any]]) -> None:
    well = record["well"]
    page = dashboard_root / "wells" / f"{well}.html"
    if not page.exists():
        return
    warning = record["overlap_warning"] or "none"
    pass_days, review_days, excluded_days = day_groups_for_well(roi_records, record)
    warning_banner = ""
    if str(record.get("alignment_qc_category", "")) != "analysis_preferred":
        warning_banner = (
            '<div class="warning-banner"><strong>Alignment uncertain:</strong> review before interpretation. '
            "Manual alignment or pseudo-FOV alignment may be needed.</div>"
        )
    section = f"""
<!-- WELL_VIEWER_SECTION_START -->
<section class="well-viewer">
  <h2>Well Summary</h2>
  <p class="caution">This is a first-pass technical review viewer. Alignment and ROI identity must be reviewed before biological interpretation.</p>
  <dl>
    <dt>Well</dt><dd>{html.escape(well)}</dd>
    <dt>Condition</dt><dd>{html.escape(str(record['condition']))}</dd>
    <dt>Days included</dt><dd>{html.escape(str(record['days_included']))}</dd>
    <dt>Included pass days</dt><dd>{html.escape(pass_days or 'none listed')}</dd>
    <dt>Review-only days</dt><dd>{html.escape(review_days or 'none')}</dd>
    <dt>Excluded days</dt><dd>{html.escape(excluded_days or 'none')}</dd>
    <dt>Overlap retained</dt><dd>{record['retained_percent']:.1f}% ({record['overlap_width']} x {record['overlap_height']} px from {record['full_width']} x {record['full_height']} px)</dd>
    <dt>QC status</dt><dd>{html.escape(str(record['qc_status']))}</dd>
    <dt>Alignment QC category</dt><dd>{html.escape(str(record.get('alignment_qc_category', 'review_only')))}</dd>
    <dt>Warning labels</dt><dd>{html.escape(warning)}</dd>
    <dt>Recommended use</dt><dd>{html.escape(str(record['recommended_use']))}</dd>
    <dt>Manual offset review</dt><dd>{'yes' if record.get('manual_alignment_review') else 'not flagged'}</dd>
    <dt>Pseudo-FOV review</dt><dd>{'yes' if record.get('pseudo_fov_alignment_review') else 'not flagged'}</dd>
  </dl>
  <h2>Well-Level Overlay / Time-Lapse Viewer</h2>
  {warning_banner}
  <p><strong>Overlap-only/common-overlap time-series preview - preferred for analysis.</strong> The controls switch days inside the same aligned frame, like a lightweight movie viewer. Full registered context is retained separately and may include edge artifacts.</p>
  {render_time_series_viewer(well, dashboard_root)}
  <h2>Alignment QC</h2>
  <p>iNeurons should not move much over 1-2 day intervals. Poor overlays suggest imaging-session field shifts, weak reference signal, focus differences, or failed registration. DAPI/reference signal can be low SNR; a stable nuclear/morphology/reference channel is preferred, and changing phenotype channels should not be the main registration anchor. Bad alignment is shown here so it can be reviewed rather than hidden.</p>
  {render_alignment_qc_table(record)}
  <h2>Neuron / ROI Candidate Viewers</h2>
  <p>Each card is a separate neuron/ROI candidate. Manual identity status remains preliminary until reviewed.</p>
  <div class="roi-card-grid">{''.join(render_well_roi_card(row) for row in roi_records)}</div>
  <h2>Full Registered vs Overlap-Only</h2>
  <p>Overlap-only views remove non-shared edge regions across days to reduce cells popping in/out due to field-of-view drift. Full registered views are retained for context but should be interpreted cautiously.</p>
  <dl>
    <dt>Full registered context</dt><dd>{html.escape(record['registered_full_stack'])}</dd>
    <dt>Overlap-only analysis stack</dt><dd>{html.escape(record['registered_common_overlap_stack'])}</dd>
    <dt>Large-shift days</dt><dd>{record['large_shift_days']}</dd>
    <dt>ROI/metrics source</dt><dd>{html.escape(record['roi_metrics_source'])}</dd>
  </dl>
</section>
<script src="../assets/viewer.js"></script>
<!-- WELL_VIEWER_SECTION_END -->
"""
    replace_marked_section_top(page, section, "WELL_VIEWER_SECTION_START", "WELL_VIEWER_SECTION_END")


def day_groups_for_well(roi_records: list[dict[str, Any]], record: dict[str, Any]) -> tuple[str, str, str]:
    if roi_records:
        first = roi_records[0]
        return (
            clean_pipe_days(first.get("pass_days", "")),
            clean_pipe_days(first.get("review_days", "")),
            clean_pipe_days(first.get("excluded_days", "")),
        )
    return clean_pipe_days(record.get("days_included", "")), clean_pipe_days(record.get("review_or_exclude_days", "")), ""


def clean_pipe_days(value: Any) -> str:
    text = str(value) if value is not None and not pd.isna(value) else ""
    if not text:
        return ""
    return ", ".join(part for part in text.split("|") if part)


def render_time_series_viewer(well: str, dashboard_root: Path) -> str:
    preview_paths = sorted(
        (dashboard_root / "previews" / well).glob(f"{well}_day*_preview.png"),
        key=lambda path: day_from_preview(path.name),
    )
    if not preview_paths:
        return "<p>No dashboard preview PNGs found for this well.</p>"
    viewer_id = f"viewer-{well}"
    frames = []
    montage = []
    day_buttons = []
    for index, path in enumerate(preview_paths):
        day = day_from_preview(path.name)
        rel = f"../previews/{html.escape(well)}/{html.escape(path.name)}"
        active = " active" if index == 0 else ""
        frames.append(
            f'<img class="viewer-frame{active}" data-day="{day}" src="{rel}" '
            f'alt="{html.escape(well)} day {day} overlap-only preview" />'
        )
        day_buttons.append(f'<button type="button" data-day-index="{index}">Day {day}</button>')
        montage.append(f'<figure><img src="{rel}" alt="{html.escape(well)} day {day}" /><figcaption>Day {day}</figcaption></figure>')
    return f"""
    <div class="time-series-viewer overlay-viewer" id="{viewer_id}" data-tmem-viewer>
      <div class="viewer-toolbar">
        <button type="button" data-viewer-action="prev">Previous</button>
        <button type="button" data-viewer-action="play">Play</button>
        <button type="button" data-viewer-action="next">Next</button>
        <label>Day <input type="range" min="0" max="{len(preview_paths) - 1}" value="0" step="1" data-viewer-slider /></label>
        <label>Speed <select data-viewer-speed><option value="1200">slow</option><option value="700" selected>medium</option><option value="350">fast</option></select></label>
        <label>Opacity <input type="range" min="0.15" max="1" step="0.05" value="1" data-viewer-opacity /></label>
        <button type="button" data-viewer-action="blink">Blink off</button>
        <button type="button" data-viewer-action="onion">Onion skin off</button>
        <strong data-viewer-label>Day {day_from_preview(preview_paths[0].name)}</strong>
      </div>
      <div class="day-controls">{''.join(day_buttons)}</div>
      <div class="time-series-frame">{''.join(frames)}</div>
      <details>
        <summary>Show all days as montage</summary>
        <div class="time-series-montage">{''.join(montage)}</div>
      </details>
    </div>
    """


def render_alignment_qc_table(record: dict[str, Any]) -> str:
    rows = []
    for row in record.get("registration_qc_table", []):
        classes = []
        if row.get("large_shift"):
            classes.append("warning-large-shift")
        if row.get("max_shift_exceeded"):
            classes.append("warning-extreme-shift")
        if not row.get("qc_pass"):
            classes.append("review")
        rows.append(
            f'<tr class="{" ".join(classes)}">'
            f"<td>{int(row['day'])}</td>"
            f"<td>{float(row['dy']):.1f}</td>"
            f"<td>{float(row['dx']):.1f}</td>"
            f"<td>{float(row['shift_magnitude']):.1f}</td>"
            f"<td>{float(row['overlap_fraction']):.3f}</td>"
            f"<td>{'yes' if row.get('large_shift') else 'no'}</td>"
            f"<td>{'yes' if row.get('max_shift_exceeded') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('registration_channel', '')))}</td>"
            f"<td>{html.escape(str(row.get('qc_note', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Day</th><th>dy</th><th>dx</th><th>|shift| px</th><th>Overlap fraction</th>"
        "<th>Large shift</th><th>Old threshold exceeded</th><th>Registration channel</th><th>QC note</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def day_from_preview(name: str) -> int:
    match = re.search(r"_day(\d+)_", name)
    return int(match.group(1)) if match else 0


def render_well_roi_card(row: dict[str, Any]) -> str:
    roi_id = str(row.get("roi_id", ""))
    status = str(row.get("manual_identity_status", "uncertain_identity") or "uncertain_identity")
    warnings = str(row.get("colocalization_warnings", "") or "")
    if warnings.lower() == "nan" or not warnings:
        warnings = "None recorded"
    crop = (
        f"x={row.get('crop_common_overlap_x0', '')}:{row.get('crop_common_overlap_x1', '')}, "
        f"y={row.get('crop_common_overlap_y0', '')}:{row.get('crop_common_overlap_y1', '')}"
    )
    title = roi_id.split("_")[-1] if "_" in roi_id else roi_id
    warning_class = " has-warning" if warnings != "None recorded" else ""
    inherited = "Inherited from parent well: check the well-level alignment warning before interpreting this ROI."
    return f"""
    <article class="roi-viewer-card{warning_class}">
      <div class="roi-card-header"><h3>{html.escape(title)}</h3><strong>{html.escape(roi_id)}</strong></div>
      <p>{html.escape(str(row.get('condition', '')))}</p>
      <p class="mini-note">{html.escape(inherited)}</p>
      <img src="{html.escape(str(row.get('dashboard_preview_rel_from_well', '')))}" alt="{html.escape(roi_id)} preview montage" />
      <p class="mini-note">ROI montage fallback: per-day ROI frame PNGs were not found in the dashboard-safe assets, so this card does not fake a video.</p>
      <dl>
        <dt>Manual identity</dt><dd>{html.escape(status)}</dd>
        <dt>Pass days</dt><dd>{html.escape(clean_pipe_days(row.get('pass_days', '')))}</dd>
        <dt>Review days</dt><dd>{html.escape(clean_pipe_days(row.get('review_days', '')) or 'none')}</dd>
        <dt>Excluded days</dt><dd>{html.escape(clean_pipe_days(row.get('excluded_days', '')) or 'none')}</dd>
        <dt>mCherry</dt><dd>puncta {format_number(row.get('mean_puncta_count'))}; punctate fraction {format_number(row.get('mean_punctate_fraction'))}; diffuse/punctate {format_number(row.get('mean_diffuse_to_punctate_ratio'))}</dd>
        <dt>Colocalization</dt><dd>Pearson {format_number(row.get('mean_pearson_correlation'))}; Manders A-in-B {format_number(row.get('mean_manders_a_in_b'))}; B-in-A {format_number(row.get('mean_manders_b_in_a'))}</dd>
        <dt>Warnings</dt><dd>{html.escape(warnings)}</dd>
        <dt>Source</dt><dd>{html.escape(str(row.get('overlap_only_source_label', '')))}</dd>
        <dt>Crop</dt><dd>{html.escape(crop)}</dd>
      </dl>
      <p><a href="{html.escape(str(row.get('roi_detail_rel_from_well', '')))}">Open ROI detail page</a></p>
    </article>
    """


def render_alignment_qc_review_page(summary: pd.DataFrame) -> str:
    best = summary.sort_values(["retained_fraction", "large_shift_days"], ascending=[False, True]).head(20)
    worst = summary.sort_values(["retained_fraction", "large_shift_days"], ascending=[True, False]).head(20)
    manual = summary[summary["manual_alignment_review"].astype(bool)].sort_values(["retained_fraction", "max_shift_magnitude"], ascending=[True, False])
    pseudo = summary[summary["pseudo_fov_alignment_review"].astype(bool)].sort_values("retained_fraction")
    low_overlap = summary[summary["retained_fraction"] < LOW_RETAINED_FRACTION].sort_values("retained_fraction")
    body = f"""
    <header class="site-header">
      <h1>Alignment QC Review</h1>
      <p>Technical review page for deciding which wells are trustworthy, which need manual offset review, and which may need pseudo-FOV alignment.</p>
      <nav><a href="index.html">Plate viewer</a><a href="overlap_only_qc_summary.html">Overlap-only QC</a><a href="roi_review_queue.html">ROI review queue</a></nav>
    </header>
    <main>
      <section class="viewer-panel warning-banner">
        <h2>How To Read This</h2>
        <p>iNeurons should overlay closely across 1-2 day intervals if the imaging-session alignment is good. Poor overlays can reflect plate/FOV shifts, failed automatic registration, weak DAPI/reference signal, focus differences, or insufficient single-FOV context.</p>
        <p>SBS mean or an alternate stable signal may sometimes outperform DAPI, but changing phenotype channels should not be the main registration anchor. These flags are technical QC labels, not biological findings.</p>
      </section>
      <section class="viewer-panel"><h2>Safest Wells To Review First</h2>{render_table(best)}</section>
      <section class="viewer-panel"><h2>Worst Alignment / Highest Pop-In Risk</h2>{render_table(worst)}</section>
      <section class="viewer-panel"><h2>Manual Alignment Review Needed</h2>{render_table(manual.head(40))}</section>
      <section class="viewer-panel"><h2>Pseudo-FOV Alignment Likely Needed</h2>{render_table(pseudo.head(40))}</section>
      <section class="viewer-panel"><h2>Low Retained Overlap</h2>{render_table(low_overlap.head(60))}</section>
    </main>
    """
    return render_page("Alignment QC Review", body, asset_prefix="")


def write_static_assets(dashboard_root: Path) -> None:
    assets = dashboard_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "site.css").write_text(SITE_CSS, encoding="utf-8")
    (assets / "viewer.js").write_text(VIEWER_JS, encoding="utf-8")


def format_number(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return "n/a"


def inject_roi_overlap_sections(
    dashboard_root: Path,
    records: list[dict[str, Any]],
    roi_dashboard: dict[str, list[dict[str, Any]]],
) -> None:
    by_well = {record["well"]: record for record in records}
    by_roi = {str(row["roi_id"]): row for roi_rows in roi_dashboard.values() for row in roi_rows}
    for page in (dashboard_root / "rois").glob("*/*.html"):
        match = re.match(r"([A-Z]\d{2})_ROI\d+\.html$", page.name)
        if not match:
            continue
        well = match.group(1)
        roi_id = page.stem
        record = by_well.get(well)
        if not record:
            continue
        roi = by_roi.get(roi_id, {})
        section = f"""
<!-- OVERLAP_ONLY_ROI_SECTION_START -->
<section>
  <p><a href="../../wells/{html.escape(well)}.html">Back to well viewer</a></p>
  <h2>ROI Candidate Review Viewer</h2>
  <p class="caution">This ROI is a candidate neuron track, not a confirmed same-neuron identity until manually reviewed.</p>
  <p>This ROI candidate was generated from the overlap-only common-overlap stack when pass5 source metadata points to <code>registered_common_overlap</code>.</p>
  <figure class="roi-detail-preview">
    <img src="../../roi_previews/{html.escape(well)}/{html.escape(roi_id)}.png" alt="{html.escape(roi_id)} preview montage" />
    <figcaption>Montage fallback. Per-day ROI frame PNGs were not found in the dashboard-safe assets, so no ROI video is shown.</figcaption>
  </figure>
  <dl>
    <dt>Parent well</dt><dd>{html.escape(well)}</dd>
    <dt>ROI ID</dt><dd>{html.escape(roi_id)}</dd>
    <dt>Condition</dt><dd>{html.escape(str(roi.get('condition', record.get('condition', ''))))}</dd>
    <dt>Manual identity</dt><dd>{html.escape(str(roi.get('manual_identity_status', 'uncertain_identity') or 'uncertain_identity'))}</dd>
    <dt>Review status options</dt><dd>confirmed_same_neuron, uncertain_identity, lost_or_dead, poor_registration, exclude</dd>
    <dt>Output level</dt><dd>overlap_only_analysis</dd>
    <dt>Overlap source label</dt><dd>{html.escape(str(roi.get('overlap_only_source_label', 'overlap-only/common-overlap source')))}</dd>
    <dt>Well retained overlap</dt><dd>{record['retained_percent']:.1f}%</dd>
    <dt>Inherited alignment QC</dt><dd>{html.escape(str(record.get('alignment_qc_category', 'review_only')))}</dd>
    <dt>Full-registered crop box</dt><dd>x={record['overlap_crop_x_start_registered']}:{record['overlap_crop_x_stop_registered']}, y={record['overlap_crop_y_start_registered']}:{record['overlap_crop_y_stop_registered']}</dd>
    <dt>Traceability</dt><dd>{html.escape(record['original_pixel_traceability'])}</dd>
    <dt>mCherry summary</dt><dd>puncta {format_number(roi.get('mean_puncta_count'))}; punctate fraction {format_number(roi.get('mean_punctate_fraction'))}; diffuse/punctate {format_number(roi.get('mean_diffuse_to_punctate_ratio'))}</dd>
    <dt>Colocalization summary</dt><dd>Pearson {format_number(roi.get('mean_pearson_correlation'))}; Manders A-in-B {format_number(roi.get('mean_manders_a_in_b'))}; B-in-A {format_number(roi.get('mean_manders_b_in_a'))}</dd>
    <dt>Warnings</dt><dd>{html.escape(str(roi.get('colocalization_warnings', '') or 'None recorded'))}</dd>
  </dl>
  <p><strong>Review instruction:</strong> confirm same-neuron identity before biological interpretation; leave uncertain candidates as <code>uncertain_identity</code> or mark poor-registration/exclude in the working review CSV.</p>
  <p><a href="../../wells/{html.escape(well)}.html">Back to well viewer</a></p>
</section>
<script src="../../assets/viewer.js"></script>
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


def replace_marked_section_top(path: Path, section: str, start_name: str, end_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    start_marker = f"<!-- {start_name} -->"
    end_marker = f"<!-- {end_name} -->"
    if start_marker in text and end_marker in text:
        before = text.split(start_marker, 1)[0]
        after = text.split(end_marker, 1)[1]
        text = before + section.strip() + after
    elif "<main>" in text:
        text = text.replace("<main>", "<main>\n" + section, 1)
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


def render_page(title: str, body: str, asset_prefix: str = "") -> str:
    css_href = f'{asset_prefix}assets/site.css'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{html.escape(css_href)}" />
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
