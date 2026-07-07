"""Illumination correction and flatfield preprocessing.

Adapted from BrieFlow's CellProfiler-based IC approach (Singh et al. 2014).
Runs BEFORE registration to normalize illumination artifacts across tiles/wells.
"""

from __future__ import annotations

import random
import warnings
from concurrent.futures import ThreadPoolExecutor
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
    n_workers: int = 4,
) -> np.ndarray:
    """Calculate an illumination correction field from a collection of images.

    Computes the pixelwise average then applies median-filter smoothing, then
    optionally rescales so the minimum correction factor is 1.

    For multi-channel images (CYX), loads each image once and accumulates
    all channels simultaneously (avoids redundant I/O). File-based images
    are loaded in parallel using threads.

    Args:
        images: List of 2D/3D image arrays or file paths.
        smooth: Disk radius for median smoothing. Defaults to sqrt(area / 20π).
        rescale_field: If True, rescale so minimum correction ≥ 1.
        sample_fraction: Fraction of images to use (randomly sampled). Default 1.0.
        channel: If set, extract this channel from multi-channel images before
            computing the IC field (returns 2D). If None and images are
            multi-channel, computes per-channel IC fields (returns CYX).
        n_workers: Thread pool size for parallel file loading. Default 4.

    Returns:
        IC field array — 2D (YX) or 3D (CYX) depending on input/channel arg.
    """
    if not images:
        raise ValueError("No images provided for IC field calculation")

    if sample_fraction < 1.0:
        k = max(1, int(len(images) * sample_fraction))
        images = random.sample(list(images), k)

    n = len(images)
    first = _load_image(images[0])

    # Multi-channel: accumulate all channels in one pass (no re-loading)
    if first.ndim == 3 and channel is None:
        accumulator = first.astype(np.float64) / n
        for loaded in _iter_images(images[1:], n_workers):
            accumulator += loaded.astype(np.float64) / n
        return _smooth_and_rescale_multichannel(
            accumulator.astype(np.uint16), smooth, rescale_field
        )

    # Single channel
    first_2d = _extract_channel(first, channel)
    accumulator = first_2d.astype(np.float64) / n
    for loaded in _iter_images(images[1:], n_workers):
        accumulator += _extract_channel(loaded, channel).astype(np.float64) / n

    return _smooth_and_rescale_2d(accumulator.astype(np.uint16), smooth, rescale_field)


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
# Timepoint-aware IC (one IC field per imaging session)
# ---------------------------------------------------------------------------


def calculate_ic_fields_by_timepoint(
    plate_dir: str | Path,
    sample_fraction: float = 0.25,
    smooth: int | None = None,
) -> dict[str, np.ndarray]:
    """Calculate per-timepoint IC fields for a plate.

    Expects plate_dir to contain subdirectories, one per timepoint/imaging
    session. Each subdir should contain the raw images for that session.
    Returns a dict keyed by timepoint directory name.

    This is the standard approach in bioimage analysis: microscope illumination
    drifts between sessions, so each timepoint needs its own IC field.

    Args:
        plate_dir: Top-level plate directory containing timepoint subdirs.
        sample_fraction: Fraction of images to sample per timepoint.
        smooth: Median filter disk radius. None for auto.

    Returns:
        Dict mapping timepoint dirname → IC field array (2D or CYX).
    """
    plate_path = Path(plate_dir)
    timepoint_dirs = sorted(
        d for d in plate_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not timepoint_dirs:
        raise FileNotFoundError(f"No timepoint subdirectories in {plate_dir}")

    ic_fields = {}
    for tp_dir in timepoint_dirs:
        images = find_images(tp_dir)
        if not images:
            continue
        ic_fields[tp_dir.name] = calculate_ic_field(
            images, smooth=smooth, sample_fraction=sample_fraction
        )

    if not ic_fields:
        raise FileNotFoundError(f"No images found in any subdirectory of {plate_dir}")

    return ic_fields


def preprocess_with_lookup(
    image_path: str | Path,
    ic_fields: dict[str, np.ndarray],
    background_radius: int | None = None,
) -> np.ndarray:
    """Load an image and preprocess with auto-selected IC field.

    Resolves the correct IC field by matching the image's parent directory
    name against the ic_fields dict keys (timepoint directory names).

    Args:
        image_path: Path to the image file.
        ic_fields: Dict from calculate_ic_fields_by_timepoint().
        background_radius: Rolling ball radius, or None to skip.

    Returns:
        Preprocessed uint16 image.

    Raises:
        KeyError: If no IC field found for the image's timepoint.
    """
    image_path = Path(image_path)
    image = read_image(str(image_path))

    # Walk up parents to find a matching timepoint key
    ic_field = _resolve_ic_field(image_path, ic_fields)

    return preprocess_image(image, ic_field=ic_field, background_radius=background_radius)


def _resolve_ic_field(
    image_path: Path, ic_fields: dict[str, np.ndarray]
) -> np.ndarray:
    """Find the IC field matching an image path's timepoint directory."""
    path = Path(image_path)
    for parent in [path.parent] + list(path.parents):
        if parent.name in ic_fields:
            return ic_fields[parent.name]
    raise KeyError(
        f"No IC field for image {image_path}. "
        f"Parent dirs checked against keys: {list(ic_fields.keys())}"
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_image(img) -> np.ndarray:
    """Load an image, whether it's an array or a path."""
    if isinstance(img, (str, Path)):
        return read_image(str(img))
    return np.asarray(img)


def _iter_images(images, n_workers: int):
    """Yield loaded images, using threaded I/O for file paths."""
    is_paths = images and isinstance(images[0], (str, Path))
    if is_paths and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            yield from pool.map(_load_image, images)
    else:
        for img in images:
            yield _load_image(img)


def _extract_channel(img: np.ndarray, channel: int | None) -> np.ndarray:
    """Extract a single channel from an image, or normalize to 2D."""
    if channel is not None and img.ndim >= 3:
        return img[channel]
    return normalize_to_2d(img)


def _smooth_and_rescale_2d(
    avg: np.ndarray, smooth: int | None, rescale_field: bool
) -> np.ndarray:
    """Apply median filter smoothing and optional rescaling to a 2D IC average."""
    if smooth is None:
        smooth = int(np.sqrt((avg.shape[-1] * avg.shape[-2]) / (np.pi * 20)))
    selem = morphology.disk(smooth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = skimage_median(avg, selem)
    if rescale_field:
        smoothed = _rescale_field(smoothed.astype(np.float64))
    return smoothed


def _smooth_and_rescale_multichannel(
    avg: np.ndarray, smooth: int | None, rescale_field: bool
) -> np.ndarray:
    """Apply smoothing/rescaling independently per channel."""
    return np.stack(
        [_smooth_and_rescale_2d(avg[c], smooth, rescale_field) for c in range(avg.shape[0])],
        axis=0,
    )


def _rescale_field(field: np.ndarray) -> np.ndarray:
    """Rescale IC field so minimum value is 1 (no darkening)."""
    robust_min = np.quantile(field.ravel(), 0.02)
    if robust_min == 0:
        robust_min = 1
    field = field / robust_min
    field[field < 1] = 1
    return field
