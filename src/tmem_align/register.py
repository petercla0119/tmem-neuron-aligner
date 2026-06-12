from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from .io import normalize_to_2d, read_image, write_ome_tiff


def register_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 10,
    max_shift_pixels: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    ref2d = normalize_to_2d(reference)
    mov2d = normalize_to_2d(moving)
    shift, _, _ = phase_cross_correlation(ref2d, mov2d, upsample_factor=upsample_factor)
    dy, dx = float(shift[0]), float(shift[1])
    if max_shift_pixels is not None and (abs(dy) > max_shift_pixels or abs(dx) > max_shift_pixels):
        raise ValueError(f"Estimated shift {(dy, dx)} exceeds max_shift_pixels={max_shift_pixels}")
    registered = apply_shift(moving, dy, dx)
    return registered, (dy, dx)


def apply_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        shift_vec = (dy, dx)
    else:
        shift_vec = (0,) * (arr.ndim - 2) + (dy, dx)
    return ndi_shift(arr, shift=shift_vec, order=1, mode="constant", cval=0).astype(arr.dtype)


def register_file_to_reference(
    reference_path: str | Path,
    moving_path: str | Path,
    output_path: str | Path,
    upsample_factor: int = 10,
    max_shift_pixels: float | None = None,
) -> tuple[Path, tuple[float, float]]:
    reference = read_image(reference_path)
    moving = read_image(moving_path)
    registered, shift = register_translation(reference, moving, upsample_factor, max_shift_pixels)
    write_ome_tiff(output_path, registered, axes=_guess_axes(registered.ndim))
    return Path(output_path), shift


def _guess_axes(ndim: int) -> str:
    return {2: "YX", 3: "CYX", 4: "TCYX", 5: "TCZYX"}.get(ndim, "YX")
