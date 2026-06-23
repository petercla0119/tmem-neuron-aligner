#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile as tif


SUPPORTED_FILE_SUFFIXES = (".nd2", ".tif", ".tiff")
SUPPORTED_DIR_SUFFIXES = (".zarr",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory a 260213 Feb recopy imaging folder.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_inventory(args.data_root, inspect_metadata=args.inspect_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote dataset inventory: {args.output}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df["extension"].value_counts(dropna=False).to_string())
        warnings = df[df["parse_confidence"] != "high"]
        if not warnings.empty:
            print(f"Warning rows needing review: {len(warnings)}")


def build_inventory(data_root: Path, *, inspect_metadata: bool = False) -> pd.DataFrame:
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    rows: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and _is_supported_file(path):
            rows.append(inventory_path(path, data_root, inspect_metadata=inspect_metadata))
        elif path.is_dir() and path.suffix.lower() in SUPPORTED_DIR_SUFFIXES:
            rows.append(inventory_path(path, data_root, inspect_metadata=False))
    return pd.DataFrame(rows)


def inventory_path(path: Path, data_root: Path, *, inspect_metadata: bool) -> dict[str, Any]:
    parsed = parse_image_filename(path)
    stat = path.stat()
    row: dict[str, Any] = {
        "path": str(path),
        "relative_path": str(path.relative_to(data_root)),
        "filename": path.name,
        "extension": _extension(path),
        "file_size_bytes": stat.st_size if path.is_file() else "",
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "inferred_well": parsed.get("well", ""),
        "inferred_day_timepoint": parsed.get("day", ""),
        "inferred_channel": parsed.get("channel", ""),
        "inferred_site_fov_tile": parsed.get("site", ""),
        "acquisition_folder": path.parent.name,
        "parse_confidence": parsed["parse_confidence"],
        "parse_warnings": "; ".join(parsed["warnings"]),
        "metadata_status": "not_requested",
        "axis_order": "",
        "sizes_json": "",
        "channel_names": "",
    }
    if inspect_metadata:
        row.update(inspect_lazy_metadata(path))
    return row


def parse_image_filename(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    warnings: list[str] = []

    well_match = re.search(r"Well([A-P]\d{2})", name, re.IGNORECASE)
    day_match = re.search(r"day\s*[_-]?(\d+)", name, re.IGNORECASE)
    channel_match = re.search(r"Channel(.+?)(?:_Seq|_s\d+|\.nd2|\.ome\.tiff?|\.tiff?$)", name, re.IGNORECASE)
    seq_match = re.search(r"Seq(\d+)", name, re.IGNORECASE)
    site_match = re.search(r"(?:site|fov|tile|position|pos)[_-]?(\d+)", name, re.IGNORECASE)

    if not well_match:
        warnings.append("well_not_parsed")
    if not day_match:
        warnings.append("day_not_parsed")
    if not channel_match:
        warnings.append("channel_not_parsed")
    site = site_match.group(1) if site_match else (f"Seq{seq_match.group(1)}" if seq_match else "")
    if not site:
        warnings.append("site_or_sequence_not_parsed")

    return {
        "well": well_match.group(1).upper() if well_match else "",
        "day": int(day_match.group(1)) if day_match else "",
        "channel": channel_match.group(1).replace(",", "|").strip() if channel_match else "",
        "site": site,
        "parse_confidence": "high" if not warnings else ("medium" if well_match and day_match else "low"),
        "warnings": warnings,
    }


def inspect_lazy_metadata(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".nd2":
            import nd2  # type: ignore

            with nd2.ND2File(path) as image:
                sizes = {str(k): int(v) for k, v in image.sizes.items()}
                channels = []
                try:
                    for index, channel in enumerate(image.metadata.channels):
                        name = getattr(getattr(channel, "channel", channel), "name", None)
                        channels.append(str(name) if name else f"Channel{index}")
                except Exception:
                    channels = [f"Channel{i}" for i in range(sizes.get("C", 1))]
                return {
                    "metadata_status": "ok",
                    "axis_order": "".join(sizes.keys()),
                    "sizes_json": json.dumps(sizes),
                    "channel_names": "|".join(channels),
                }
        if _extension(path) in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
            with tif.TiffFile(path) as image:
                series = image.series[0]
                axes = getattr(series, "axes", "")
                shape = getattr(series, "shape", "")
                return {
                    "metadata_status": "ok",
                    "axis_order": axes,
                    "sizes_json": json.dumps({"shape": shape}),
                    "channel_names": "",
                }
    except Exception as exc:
        return {"metadata_status": f"metadata_error: {exc}"}
    return {"metadata_status": "unsupported_metadata"}


def _is_supported_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(SUPPORTED_FILE_SUFFIXES) or lower.endswith((".ome.tif", ".ome.tiff"))


def _extension(path: Path) -> str:
    lower = path.name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".nd2", ".zarr"):
        if lower.endswith(suffix):
            return suffix
    return path.suffix.lower()


if __name__ == "__main__":
    main()
