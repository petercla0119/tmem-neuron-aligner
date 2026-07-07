"""Illumination correction and flatfield preprocessing.

Adapted from BrieFlow's CellProfiler-based IC approach (Singh et al. 2014).
Runs BEFORE registration to normalize illumination artifacts across tiles/wells.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
from skimage import morphology
from skimage.filters import median as skimage_median
from skimage.restoration import rolling_ball, ball_kernel
from skimage.transform import rescale, resize

from tmem_align.io import find_images, read_image, normalize_to_2d


# ---------------------------------------------------------------------------
# IC field calculation (per-well or per-plate)
# ---------------------------------------------------------------------------


def calculate_ic_field(
    images: list[np.ndarray] | list[str | Path],
    smooth: int | None = None,
    rescale_field: bool = True,
    sample_fraction: float = 1.0,
    channel: int | None = None,
) -> np.ndarray:
    """Calculate an illumination correction field from a collection of images.

    Computes the pixelwise average then applies median-filter smoothing, then
    optionally rescales so the minimum correction factor is 1.

    For multi-channel images (CYX), either specify a channel index or the
    function computes a per-channel IC field and returns shape (C, Y, X).

    Args:
        images: List of 2D/3D image arrays or file paths.
        smooth: Disk radius for median smoothing. Defaults to sqrt(area / 20π).
        rescale_field: If True, rescale so minimum correction ≥ 1.
        sample_fraction: Fraction of images to use (randomly sampled). Default 1.0.
        channel: If set, extract this channel from multi-channel images before
            computing the IC field (returns 2D). If None and images are
            multi-channel, computes per-channel IC fields (returns CYX).

    Returns:
        IC field array — 2D (YX) or 3D (CYX) depending on input/channel arg.
    """
    if not images:
        raise ValueError("No images provided for IC field calculation")

    if sample_fraction < 1.0:
        k = max(1, int(len(images) * sample_fraction))
        images = random.sample(list(images), k)

    first = _load_image(images[0])

    # Multi-channel case: recurse per channel
    if first.ndim == 3 and channel is None:
        n_channels = first.shape[0]
        return np.stack(
            [calculate_ic_field(images, smooth=smooth, rescale_field=rescale_field,
                                sample_fraction=1.0, channel=c)
             for c in range(n_channels)],
            axis=0,
        )

    # Single channel computation
    first_2d = _extract_channel(first, channel)
    accumulator = first_2d.astype(np.float64) / len(images)
    for img in images[1:]:
        loaded = _load_image(img)
        accumulator += _extract_channel(loaded, channel).astype(np.float64) / len(images)

    avg = accumulator.astype(np.uint16)

    if smooth is None:
        smooth = int(np.sqrt((avg.shape[-1] * avg.shape[-2]) / (np.pi * 20)))

    selem = morphology.disk(smooth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = skimage_median(avg, selem)

    if rescale_field:
        smoothed = _rescale_field(smoothed.astype(np.float64))

    return smoothed


def apply_ic_field(image: np.ndarray, ic_field: np.ndarray) -> np.ndarray:
    """Apply illumination correction by dividing image by the IC field.

    Handles 2D/3D/4D images with 2D or 3D (per-channel) IC fields.
    Broadcasting rules:
      - 2D field + 3D image (CYX): broadcasts field across channels
      - 3D field (CYX) + 3D image (CYX): per-channel division
      - 2D field + 4D image (TCYX): broadcasts across T and C
      - 3D field (CYX) + 4D image (TCYX): broadcasts across T

    Returns uint16 corrected image.
    """
    if ic_field is None:
        return image

    field = ic_field.astype(np.float64)
    field[field == 0] = 1

    img = image.astype(np.float64)

    if img.ndim == field.ndim:
        return (img / field).astype(np.uint16)

    # Reshape field for broadcasting
    if field.ndim == 2:
        while field.ndim < img.ndim:
            field = field[np.newaxis]
    elif field.ndim == 3 and img.ndim == 4:
        field = field[np.newaxis]

    return (img / field).astype(np.uint16)


# ---------------------------------------------------------------------------
# Rolling ball background subtraction
# ---------------------------------------------------------------------------


def subtract_background(
    image: np.ndarray,
    radius: int = 100,
    shrink_factor: int | None = None,
) -> np.ndarray:
    """Rolling ball background subtraction.

    Shrinks the image for speed, computes the rolling ball background, then
    resizes back and subtracts. Works on 2D images; for multi-channel, apply
    per-channel via preprocess_image().

    Returns uint16 background-subtracted image.
    """
    img_2d = normalize_to_2d(image) if image.ndim > 2 else image
    img_f = img_2d.astype(np.float64)

    if shrink_factor is None:
        if radius <= 10:
            shrink_factor = 1
        elif radius <= 30:
            shrink_factor = 2
        elif radius <= 100:
            shrink_factor = 4
        else:
            shrink_factor = 8

    if shrink_factor > 1:
        small = rescale(img_f, 1.0 / shrink_factor, preserve_range=True)
        kernel = ball_kernel(max(1, radius // shrink_factor), ndim=2)
        bg_small = rolling_ball(small, kernel=kernel)
        bg = resize(bg_small, img_f.shape, preserve_range=True)
    else:
        kernel = ball_kernel(radius, ndim=2)
        bg = rolling_ball(img_f, kernel=kernel)

    bg = np.minimum(bg, img_f)
    result = img_f - bg
    return np.clip(result, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# High-level preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(
    image: np.ndarray,
    ic_field: np.ndarray | None = None,
    background_radius: int | None = None,
) -> np.ndarray:
    """Apply illumination correction and optional background subtraction.

    Processes each channel independently for multi-channel images.

    Args:
        image: Input image (2D, 3D CYX, or 4D TCYX).
        ic_field: Illumination correction field (2D). None to skip IC.
        background_radius: Rolling ball radius. None to skip background subtraction.

    Returns:
        Preprocessed uint16 image.
    """
    result = image

    if ic_field is not None:
        result = apply_ic_field(result, ic_field)

    if background_radius is not None:
        if result.ndim == 2:
            result = subtract_background(result, radius=background_radius)
        elif result.ndim == 3:
            result = np.stack(
                [subtract_background(result[c], radius=background_radius) for c in range(result.shape[0])],
                axis=0,
            )
        elif result.ndim == 4:
            corrected_frames = []
            for t in range(result.shape[0]):
                corrected_channels = [
                    subtract_background(result[t, c], radius=background_radius)
                    for c in range(result.shape[1])
                ]
                corrected_frames.append(np.stack(corrected_channels, axis=0))
            result = np.stack(corrected_frames, axis=0)

    return result


def calculate_ic_field_for_well(
    image_folder: str | Path,
    sample_fraction: float = 1.0,
    smooth: int | None = None,
) -> np.ndarray:
    """Calculate IC field from all images in a well folder.

    Convenience wrapper: finds all TIFF/ND2 images in folder, loads them,
    calculates the IC field.
    """
    paths = find_images(image_folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_folder}")
    return calculate_ic_field(paths, smooth=smooth, sample_fraction=sample_fraction)


def calculate_ic_field_for_plate(
    plate_folder: str | Path,
    well_pattern: str = "*",
    sample_fraction: float = 0.25,
    smooth: int | None = None,
) -> np.ndarray:
    """Calculate IC field from images across an entire plate.

    Collects images from all well subfolders matching well_pattern, samples
    a fraction, and computes a single plate-level IC field.
    """
    plate_path = Path(plate_folder)
    all_images = []
    for well_dir in sorted(plate_path.glob(well_pattern)):
        if well_dir.is_dir():
            all_images.extend(find_images(well_dir))

    if not all_images:
        raise FileNotFoundError(f"No images found in {plate_folder}/{well_pattern}")

    # ponytail: default 25% sampling for plate-level (many images)
    return calculate_ic_field(
        all_images, smooth=smooth, sample_fraction=sample_fraction
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_image(img) -> np.ndarray:
    """Load an image, whether it's an array or a path."""
    if isinstance(img, (str, Path)):
        return read_image(str(img))
    return np.asarray(img)


def _extract_channel(img: np.ndarray, channel: int | None) -> np.ndarray:
    """Extract a single channel from an image, or normalize to 2D."""
    if channel is not None and img.ndim >= 3:
        return img[channel]
    return normalize_to_2d(img)


def _rescale_field(field: np.ndarray) -> np.ndarray:
    """Rescale IC field so minimum value is 1 (no darkening)."""
    robust_min = np.quantile(field.ravel(), 0.02)
    if robust_min == 0:
        robust_min = 1
    field = field / robust_min
    field[field < 1] = 1
    return field
