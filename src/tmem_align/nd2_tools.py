from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io import write_ome_tiff


def _require_nd2():
    try:
        import nd2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ND2 support is not installed. Run `pip install -e \".[nd2]\"` or create the "
            "provided conda environment."
        ) from exc
    return nd2


def inspect_nd2(path: str | Path) -> dict[str, Any]:
    """Read metadata without loading the full ND2 pixel array into memory."""
    nd2 = _require_nd2()
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with nd2.ND2File(path) as image:
        sizes = {str(k): int(v) for k, v in image.sizes.items()}
        axis_order = "".join(sizes.keys())
        try:
            voxel = image.voxel_size()
            voxel_size = {"x_um": float(voxel.x), "y_um": float(voxel.y), "z_um": float(voxel.z)}
        except Exception:
            voxel_size = None

        channel_names: list[str] = []
        try:
            for index, channel in enumerate(image.metadata.channels):
                name = getattr(getattr(channel, "channel", channel), "name", None)
                channel_names.append(str(name) if name else f"Channel{index}")
        except Exception:
            channel_names = [f"Channel{i}" for i in range(sizes.get("C", 1))]

        return {
            "file_path": str(path),
            "file_name": path.name,
            "size_gb": round(path.stat().st_size / (1024**3), 4),
            "axis_order": axis_order,
            "sizes": sizes,
            "shape": tuple(int(v) for v in image.shape),
            "dtype": str(image.dtype),
            "is_rgb": bool(image.is_rgb),
            "channel_names": channel_names,
            "voxel_size": voxel_size,
            "position_count": int(sizes.get("P", 1)),
            "time_count": int(sizes.get("T", 1)),
            "z_count": int(sizes.get("Z", 1)),
        }


def print_nd2_report(path: str | Path) -> dict[str, Any]:
    report = inspect_nd2(path)
    print(json.dumps(report, indent=2))
    return report


def build_manifest(
    root: str | Path,
    output_csv: str | Path | None = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """Create a one-row-per-ND2 inventory without reading all pixels."""
    root = Path(root).expanduser().resolve()
    paths: Iterable[Path] = root.rglob("*.nd2") if recursive else root.glob("*.nd2")
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            info = inspect_nd2(path)
            rows.append(
                {
                    "file_path": info["file_path"],
                    "file_name": info["file_name"],
                    "size_gb": info["size_gb"],
                    "axis_order": info["axis_order"],
                    "sizes_json": json.dumps(info["sizes"]),
                    "channel_names": "|".join(info["channel_names"]),
                    "position_count": info["position_count"],
                    "time_count": info["time_count"],
                    "z_count": info["z_count"],
                    "plate": "",
                    "day": "",
                    "well": "",
                    "well_group": "",
                    "condition": "",
                    "status": "needs_review",
                }
            )
        except Exception as exc:
            rows.append({"file_path": str(path), "file_name": path.name, "status": f"ERROR: {exc}"})

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def read_fov_positions(path: str | Path) -> list[dict[str, Any]]:
    """Read per-FOV stage coordinates from a multi-position ND2 file."""
    from .stage_qc import stage_coordinates_from_frame_metadata

    nd2 = _require_nd2()
    path = Path(path).expanduser().resolve()
    with nd2.ND2File(path) as image:
        sizes = {str(k): int(v) for k, v in image.sizes.items()}
        n_positions = sizes.get("P", 1)
        frames_per_position = 1
        for axis in ("T", "C", "Z"):
            frames_per_position *= sizes.get(axis, 1)

        positions: list[dict[str, Any]] = []
        for p in range(n_positions):
            frame_idx = p * frames_per_position
            try:
                meta = image.frame_metadata(frame_idx)
                coords = stage_coordinates_from_frame_metadata(meta)
                positions.append({
                    "position_index": p,
                    "stage_x_um": coords["stage_x_um"],
                    "stage_y_um": coords["stage_y_um"],
                })
            except Exception:
                positions.append({
                    "position_index": p,
                    "stage_x_um": None,
                    "stage_y_um": None,
                })
    return positions


def read_fov_tile(
    path: str | Path,
    position: int,
    channel: int = 0,
    z_project: str = "max",
    z_index: int | None = None,
) -> np.ndarray:
    """Extract a single FOV as a 2D array, handling Z-stacks."""
    nd2 = _require_nd2()
    path = Path(path).expanduser().resolve()
    with nd2.ND2File(path) as image:
        sizes = {str(k): int(v) for k, v in image.sizes.items()}
        axis_order = list(sizes.keys())
        selection: list[Any] = []
        remaining: list[str] = []
        requested = {"P": position, "C": channel}
        if z_index is not None:
            requested["Z"] = z_index
        for axis in axis_order:
            idx = requested.get(axis)
            if idx is not None:
                selection.append(idx)
            else:
                selection.append(slice(None))
                remaining.append(axis)
        data = image.to_dask()
        arr = np.asarray(data[tuple(selection)].compute())

    if z_index is None and "Z" in remaining:
        z_ax = remaining.index("Z")
        if z_project == "max":
            arr = arr.max(axis=z_ax)
        elif z_project == "mean":
            arr = arr.mean(axis=z_ax).astype(arr.dtype)
        else:
            raise ValueError(f"Unknown z_project: {z_project}")
    return np.squeeze(arr)


def extract_nd2_selection(
    nd2_path: str | Path,
    output_path: str | Path,
    *,
    position: int | None = None,
    time: int | None = None,
    channel: int | None = None,
    z: int | None = None,
    max_project_z: bool = False,
    max_read_bytes: int = 2 * 1024**3,
    max_output_bytes: int = 5 * 1024**3,
) -> Path:
    """Lazily select a small ND2 subset and write it as OME-TIFF.

    The function uses the ND2 file's reported axis order. Unspecified axes are retained, but the
    estimated selected array size is checked before compute. It is intended for pilot extraction,
    not for blindly converting an entire 99 GB collection.
    """
    nd2 = _require_nd2()
    nd2_path = Path(nd2_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    requested = {"P": position, "T": time, "C": channel, "Z": z}
    with nd2.ND2File(nd2_path) as image:
        axis_order = list(image.sizes.keys())
        data = image.to_dask()
        selection: list[Any] = []
        remaining_axes: list[str] = []
        for axis in axis_order:
            index = requested.get(axis)
            if index is None:
                selection.append(slice(None))
                remaining_axes.append(axis)
            else:
                if index < 0 or index >= int(image.sizes[axis]):
                    raise IndexError(f"{axis} index {index} outside 0..{int(image.sizes[axis]) - 1}")
                selection.append(index)

        selected = data[tuple(selection)]
        estimated_read_bytes = int(np.prod(selected.shape, dtype=np.int64)) * np.dtype(selected.dtype).itemsize
        if estimated_read_bytes > max_read_bytes:
            raise ValueError(
                f"Requested ND2 subset is estimated at {estimated_read_bytes:,} bytes, "
                f"which exceeds max_read_bytes={max_read_bytes:,}. Select fewer indices."
            )
        arr = np.asarray(selected.compute())

    if max_project_z and "Z" in remaining_axes:
        z_axis = remaining_axes.index("Z")
        arr = arr.max(axis=z_axis)
        remaining_axes.pop(z_axis)

    if arr.nbytes > max_output_bytes:
        raise ValueError(
            f"Output array is {arr.nbytes:,} bytes before TIFF overhead, "
            f"which exceeds max_output_bytes={max_output_bytes:,}. Select fewer indices."
        )

    axes = "".join(remaining_axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ome_tiff(output_path, arr, axes=axes)
    return output_path
