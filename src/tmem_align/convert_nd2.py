from __future__ import annotations

from pathlib import Path

from .nd2_tools import extract_nd2_selection


def convert_nd2_to_ometiff(nd2_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Convert one ND2 file to OME-TIFF.

    This compatibility wrapper retains all axes and may still create a very large file. For a
    99 GB experiment, prefer `extract_nd2_selection` or the `extract-nd2` CLI command to test one
    position/well, channel, and timepoint at a time.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{nd2_path.stem}.ome.tif"
    return [extract_nd2_selection(nd2_path, out)]
