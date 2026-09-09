from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from .io import normalize_to_2d, read_image, write_ome_tiff
from .registration_qc import (
    classify_registration_qc,
    common_overlap_crop,
    correlation,
    overlap_fraction,
    robust_registration_image,
)


def register_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 10,
    max_shift_pixels: float | None = None,
    robust_preprocess: bool = True,
    mask_percentile: float | None = None,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Estimate the (dy, dx) translation aligning ``moving`` onto ``reference``.

    mask_percentile: if set, run *masked* phase cross-correlation that ignores background
    below that intensity percentile so the sparse fluorescent foreground drives the peak.
    This is the recommended path for sparse-neuron frames; do NOT combine it with
    robust_preprocess — the clip+blur smears the point-like signal and makes the peak lock
    onto image edges/illumination (see docs). Masked correlation is integer-pixel
    (upsample_factor is ignored) and returns error=nan.
    """
    ref2d = normalize_to_2d(reference)
    mov2d = normalize_to_2d(moving)
    if robust_preprocess:
        ref2d = robust_registration_image(ref2d)
        mov2d = robust_registration_image(mov2d)
    if mask_percentile is not None:
        ref2d = np.asarray(ref2d, dtype=np.float32)
        mov2d = np.asarray(mov2d, dtype=np.float32)
        ref_mask = ref2d > np.percentile(ref2d, mask_percentile)
        mov_mask = mov2d > np.percentile(mov2d, mask_percentile)
        result = phase_cross_correlation(
            ref2d, mov2d, reference_mask=ref_mask, moving_mask=mov_mask
        )
        # masked variant returns just the shift (older skimage) or a 3-tuple (newer)
        shift = np.asarray(result[0] if isinstance(result, tuple) else result).ravel()
        error = float("nan")
    else:
        shift, error, _ = phase_cross_correlation(ref2d, mov2d, upsample_factor=upsample_factor)
    dy, dx = float(shift[0]), float(shift[1])
    if max_shift_pixels is not None and (abs(dy) > max_shift_pixels or abs(dx) > max_shift_pixels):
        raise ValueError(f"Estimated shift {(dy, dx)} exceeds max_shift_pixels={max_shift_pixels}")
    registered = apply_shift(moving, dy, dx)
    return registered, (dy, dx), float(error)


def apply_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        shift_vec = (dy, dx)
    else:
        shift_vec = (0,) * (arr.ndim - 2) + (dy, dx)
    return ndi_shift(arr, shift=shift_vec, order=3, mode="constant", cval=0).astype(arr.dtype)


def register_file_to_reference(
    reference_path: str | Path,
    moving_path: str | Path,
    output_path: str | Path,
    upsample_factor: int = 10,
    max_shift_pixels: float | None = None,
    robust_preprocess: bool = True,
) -> tuple[Path, tuple[float, float], float]:
    reference = read_image(reference_path)
    moving = read_image(moving_path)
    registered, shift, error = register_translation(
        reference, moving, upsample_factor, max_shift_pixels, robust_preprocess
    )
    write_ome_tiff(output_path, registered, axes=_guess_axes(registered.ndim))
    return Path(output_path), shift, error


def _guess_axes(ndim: int) -> str:
    return {2: "YX", 3: "CYX", 4: "TCYX", 5: "TCZYX"}.get(ndim, "YX")


def _plate_offset(plate_offsets: dict | None, day: Any) -> tuple[float, float]:
    """Plate-remount prior (dy, dx) for a timepoint, or (0, 0) when absent (default = off)."""
    if not plate_offsets:
        return (0.0, 0.0)
    dy, dx = plate_offsets.get(day, (0.0, 0.0))
    return (float(dy), float(dx))


def _anchored_shifts(
    stable_frames: np.ndarray,
    thresh: float,
    plate_shifts: list[tuple[float, float]] | None = None,
) -> tuple[list[tuple[float, float]], list[float], list[bool]]:
    """Per-timepoint net (dy, dx)-to-t0, post-corr, and reanchored flag. Single source of the
    anchor math; mirrors scripts/plot_day_shift_overlay.register_anchored on the masked engine:
    register to the current anchor; if post-corr < thresh and t>=2 re-anchor to the LAST GOOD
    frame (never the current one) and re-register; compose net = anchor_net + pairwise; only
    frames with post >= thresh become eligible future anchors. No image application.

    ``plate_shifts`` (default None = byte-identical): per-timepoint plate-remount prior (dy, dx).
    When given, each frame is pre-shifted by its prior so registration sees only the residual
    drift, and the prior is added back into the returned net (plate-first, then per-well residual)."""
    if plate_shifts is not None:
        stable_frames = [
            frame if (pdy == 0.0 and pdx == 0.0) else apply_shift(frame, pdy, pdx)
            for frame, (pdy, pdx) in zip(stable_frames, plate_shifts, strict=True)
        ]
    anchor, anchor_net = stable_frames[0], (0.0, 0.0)
    last_good_img, last_good_net = stable_frames[0], (0.0, 0.0)
    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    post: list[float] = [1.0]
    reanchored: list[bool] = [False]
    for time_index in range(1, len(stable_frames)):
        moving = stable_frames[time_index]
        aligned, (pdy, pdx), _ = register_translation(
            anchor, moving, robust_preprocess=False, mask_percentile=20.0
        )
        p = correlation(anchor, aligned)
        did = False
        if p < thresh and time_index >= 2:
            anchor, anchor_net = last_good_img, last_good_net
            aligned, (pdy, pdx), _ = register_translation(
                anchor, moving, robust_preprocess=False, mask_percentile=20.0
            )
            p = correlation(anchor, aligned)
            did = True
        net = (anchor_net[0] + pdy, anchor_net[1] + pdx)
        shifts.append(net)
        post.append(p)
        reanchored.append(did)
        if p >= thresh:  # trustworthy → eligible future anchor
            last_good_img, last_good_net = moving, net
    if plate_shifts is not None:  # net = plate prior + residual (total, for apply_shift + crop)
        shifts = [
            (net[0] + pdy, net[1] + pdx)
            for net, (pdy, pdx) in zip(shifts, plate_shifts, strict=True)
        ]
    return shifts, post, reanchored


def register_stack(
    stack: np.ndarray,
    *,
    well: str,
    rows: list[dict[str, Any]],
    alignment_channel_index: int,
    alignment_channel_label: str,
    robust_crop: bool = True,
    ref_mode: str = "to_first",
    anchor_corr_thresh: float = 0.10,
    min_post_correlation: float = 0.07,
    plate_offsets: dict | None = None,
    condition: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    # ``condition`` fills the QC "condition" field; the caller maps well→condition (kept out of
    # this package). None leaves it blank.
    # Register on the RAW stable channel with masked phase correlation (ignore background
    # below p20). Do NOT clip+blur here: it smears the sparse neuron signal and makes the
    # peak lock onto image edges, producing axis-locked 500-1400 px garbage shifts (see docs).
    reference = stack[0, alignment_channel_index]

    # Plate-remount prior per timepoint (default None → all-zero → byte-identical to no correction).
    # Composed plate-first: the moving frame is pre-shifted by its prior so registration measures
    # only the per-well residual; net = prior + residual feeds both apply_shift and the crop.
    plate_shifts = (
        [_plate_offset(plate_offsets, row["day"]) for row in rows] if plate_offsets else None
    )

    if ref_mode == "anchored":
        return _register_stack_anchored(
            stack,
            reference=reference,
            well=well,
            rows=rows,
            alignment_channel_index=alignment_channel_index,
            alignment_channel_label=alignment_channel_label,
            robust_crop=robust_crop,
            anchor_corr_thresh=anchor_corr_thresh,
            min_post_correlation=min_post_correlation,
            plate_shifts=plate_shifts,
            condition=condition,
        )

    registered = [stack[0]]
    shifts = [(0.0, 0.0)]
    qc_rows = [
        {
            "well": well,
            "condition": condition,
            "timepoint_day": rows[0]["day"],
            "registration_channel": alignment_channel_label,
            "estimated_y_shift": 0.0,
            "estimated_x_shift": 0.0,
            "pre_registration_correlation": 1.0,
            "post_registration_correlation": 1.0,
            "overlap_fraction": 1.0,
            "registration_error": 0.0,
            "qc_pass": True,
            "qc_note": "reference_timepoint",
        }
    ]

    for time_index in range(1, stack.shape[0]):
        moving = stack[time_index, alignment_channel_index]
        pdy, pdx = plate_shifts[time_index] if plate_shifts else (0.0, 0.0)
        reg_moving = moving if (pdy == 0.0 and pdx == 0.0) else apply_shift(moving, pdy, pdx)
        _, (rdy, rdx), error = register_translation(
            reference,
            reg_moving,
            robust_preprocess=False,
            mask_percentile=20.0,
        )
        dy, dx = pdy + rdy, pdx + rdx  # net = plate prior + per-well residual
        shifted_channel = apply_shift(moving, dy, dx)
        registered.append(apply_shift(stack[time_index], dy, dx))
        shifts.append((dy, dx))
        overlap = overlap_fraction(stack.shape[-2:], (dy, dx))
        post_corr = correlation(reference, shifted_channel)
        qc_rows.append(
            {
                "well": well,
                "condition": condition,
                "timepoint_day": rows[time_index]["day"],
                "registration_channel": alignment_channel_label,
                "estimated_y_shift": dy,
                "estimated_x_shift": dx,
                "pre_registration_correlation": correlation(reference, moving),
                "post_registration_correlation": post_corr,
                "overlap_fraction": overlap,
                "registration_error": float(error),
                **classify_registration_qc(
                    overlap,
                    dy,
                    dx,
                    stack.shape[-2],
                    stack.shape[-1],
                    post_correlation=post_corr,
                ),
                "qc_note": "masked_phase_cross_correlation_on_raw_stable_channel",
            }
        )

    registered_stack = np.stack(registered, axis=0)
    return (
        registered_stack,
        qc_rows,
        common_overlap_crop(stack.shape[-2:], shifts, robust=robust_crop),
    )


def _register_stack_anchored(
    stack: np.ndarray,
    *,
    reference: np.ndarray,
    well: str,
    rows: list[dict[str, Any]],
    alignment_channel_index: int,
    alignment_channel_label: str,
    robust_crop: bool,
    anchor_corr_thresh: float,
    min_post_correlation: float,
    plate_shifts: list[tuple[float, float]] | None = None,
    condition: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Anchored/masked temporal registration. Net shifts come from _anchored_shifts (the single
    source of anchor math); we apply each net shift to ALL channels and feed the net shifts to
    the common-overlap crop, exactly like the to_first path."""
    stable_frames = stack[:, alignment_channel_index]
    net_shifts, post_corrs, reanchored_flags = _anchored_shifts(
        stable_frames, anchor_corr_thresh, plate_shifts=plate_shifts
    )

    # Reconstruct which frame served as the anchor per timepoint (for auditable anchor_ref_day):
    # anchor is day0 until a re-anchor promotes the last good frame; last-good tracks post>=thresh.
    anchor_ref_days = [rows[0]["day"]]
    last_good_index = 0
    anchor_index = 0
    for time_index in range(1, stack.shape[0]):
        if reanchored_flags[time_index]:
            anchor_index = last_good_index
        anchor_ref_days.append(rows[anchor_index]["day"])
        if post_corrs[time_index] >= anchor_corr_thresh:
            last_good_index = time_index

    # Per-well churn verdict (§2.6): fail if re-anchoring more than every other frame, or any
    # timepoint still below the QC gate after its retry.
    n_timepoints = stack.shape[0]
    n_reanchors = int(sum(reanchored_flags))
    anchor_churn = n_reanchors / (n_timepoints - 1) if n_timepoints > 1 else 0.0
    any_below_gate = any(p < min_post_correlation for p in post_corrs[1:])
    well_registration_qc_pass = not (anchor_churn > 0.5 or any_below_gate)

    registered = [stack[0]]
    qc_rows = [
        {
            "well": well,
            "condition": condition,
            "timepoint_day": rows[0]["day"],
            "registration_channel": alignment_channel_label,
            "estimated_y_shift": 0.0,
            "estimated_x_shift": 0.0,
            "pre_registration_correlation": 1.0,
            "post_registration_correlation": 1.0,
            "overlap_fraction": 1.0,
            "registration_error": float("nan"),
            "qc_pass": True,
            "large_shift": False,
            "reanchored": False,
            "anchor_ref_day": anchor_ref_days[0],
            "n_reanchors": n_reanchors,
            "anchor_churn": anchor_churn,
            "well_registration_qc_pass": well_registration_qc_pass,
            "qc_note": "anchored_masked_phase_cross_correlation",
        }
    ]

    for time_index in range(1, stack.shape[0]):
        dy, dx = net_shifts[time_index]
        post = post_corrs[time_index]
        moving = stable_frames[time_index]
        registered.append(apply_shift(stack[time_index], dy, dx))
        overlap = overlap_fraction(stack.shape[-2:], (dy, dx))
        qc_rows.append(
            {
                "well": well,
                "condition": condition,
                "timepoint_day": rows[time_index]["day"],
                "registration_channel": alignment_channel_label,
                "estimated_y_shift": dy,
                "estimated_x_shift": dx,
                "pre_registration_correlation": correlation(reference, moving),
                "post_registration_correlation": post,
                "overlap_fraction": overlap,
                "registration_error": float("nan"),
                **classify_registration_qc(
                    overlap,
                    dy,
                    dx,
                    stack.shape[-2],
                    stack.shape[-1],
                    post_correlation=post,
                    min_post_correlation=min_post_correlation,
                ),
                "reanchored": bool(reanchored_flags[time_index]),
                "anchor_ref_day": anchor_ref_days[time_index],
                "n_reanchors": n_reanchors,
                "anchor_churn": anchor_churn,
                "well_registration_qc_pass": well_registration_qc_pass,
                "qc_note": "anchored_masked_phase_cross_correlation",
            }
        )

    registered_stack = np.stack(registered, axis=0)
    return (
        registered_stack,
        qc_rows,
        common_overlap_crop(stack.shape[-2:], net_shifts, robust=robust_crop),
    )
