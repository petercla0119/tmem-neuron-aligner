from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile as tif


def find_images(folder: str | Path, suffixes: Iterable[str] = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".nd2")) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in suffixes or any(str(p).lower().endswith(s) for s in suffixes))


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".nd2":
        import nd2
        return np.asarray(nd2.imread(str(path)))
    arr = tif.imread(path)
    return np.asarray(arr)


def write_ome_tiff(path: str | Path, image: np.ndarray, axes: str = "CYX", pixel_size_um: float | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"axes": axes}
    if pixel_size_um:
        metadata["PhysicalSizeX"] = pixel_size_um
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeY"] = pixel_size_um
        metadata["PhysicalSizeYUnit"] = "µm"
    tif.imwrite(path, image, photometric="minisblack", metadata=metadata, ome=True)


def save_mask_for_napari(
    out_dir: str | Path,
    name: str,
    mask: np.ndarray,
    image: np.ndarray | None = None,
    axes: str | None = None,
    voxel_size_um: tuple[float, ...] | None = None,
) -> list[Path]:
    """Write a mask (+ optional paired image) as napari-openable TIFF(s).

    Reusable across any mask-generating script (coloc, surveys, puncta detection…):
    writes ``<name>_mask.tif`` (labels/boolean → uint8 if <256 labels, else uint16)
    and, when ``image`` is given, ``<name>_img.tif``. Both carry the same ``axes``
    (inferred "ZYX"/"YX" from ndim when None) so napari overlays them 1:1. Open with
    ``napari --layer-type image <name>_img.tif --layer-type labels <name>_mask.tif``.
    ``voxel_size_um`` = (z,y,x) or (y,x) sets the physical scale in the TIFF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask)
    if axes is None:
        axes = {2: "YX", 3: "ZYX"}.get(mask.ndim)
        if axes is None:
            raise ValueError(f"pass axes explicitly for {mask.ndim}D mask")

    def _meta() -> dict:
        m: dict = {"axes": axes}
        if voxel_size_um and len(voxel_size_um) == len(axes):
            unit = "µm"
            for ax, sz in zip(axes, voxel_size_um):
                m[f"PhysicalSize{ax}"] = float(sz)
                m[f"PhysicalSize{ax}Unit"] = unit
        return m

    written: list[Path] = []
    mdtype = np.uint8 if int(mask.max(initial=0)) < 256 else np.uint16
    mpath = out_dir / f"{name}_mask.tif"
    tif.imwrite(mpath, mask.astype(mdtype), metadata=_meta())
    written.append(mpath)
    if image is not None:
        image = np.asarray(image)
        idtype = image.dtype if np.issubdtype(image.dtype, np.integer) else np.uint16
        ipath = out_dir / f"{name}_img.tif"
        tif.imwrite(ipath, image.astype(idtype), metadata=_meta())
        written.append(ipath)
    return written


def normalize_to_2d(image: np.ndarray) -> np.ndarray:
    """Collapse simple singleton axes to a 2D image for registration/stitching."""
    arr = np.squeeze(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Conservative fallback: maximum projection over first axis.
        return arr.max(axis=0)
    raise ValueError(f"Cannot normalize image with shape {image.shape} to 2D")
