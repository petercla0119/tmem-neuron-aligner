#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tmem_align.export_zarr import export_ome_zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export registered report OME-TIFF stacks to chunked OME-Zarr."
    )
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*registered*_tcyx.ome.tif")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--axes", default="tcyx")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 1, 256, 256])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    output_dir = args.output_dir or (report_root / "ome_zarr")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tif_path in sorted(report_root.glob(args.glob)):
        if not tif_path.is_file():
            continue
        zarr_name = tif_path.name.replace(".ome.tif", ".ome.zarr").replace(".tif", ".ome.zarr")
        zarr_path = output_dir / zarr_name
        export_ome_zarr(tif_path, zarr_path, axes=args.axes, chunks=tuple(args.chunks))
        rows.append(
            {
                "source_ome_tiff": str(tif_path),
                "output_ome_zarr": str(zarr_path),
                "axes": args.axes,
                "chunks": "x".join(map(str, args.chunks)),
                "size_bytes": directory_size(zarr_path),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "ome_zarr_export_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote OME-Zarr export manifest: {manifest_path}")
    print(manifest.to_string(index=False))


def directory_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


if __name__ == "__main__":
    main()

