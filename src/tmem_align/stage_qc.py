from __future__ import annotations

import math
from pathlib import Path
from typing import Any


DEFAULT_STAGE_XY_THRESHOLD_UM = 5.0


def read_nd2_stage_coordinates(path: str | Path) -> dict[str, Any]:
    """Read stage coordinates from ND2 metadata without loading pixels."""
    try:
        import nd2  # type: ignore
    except ImportError as exc:
        raise ImportError("ND2 support is required for stage-coordinate prefiltering.") from exc

    path = Path(path)
    with nd2.ND2File(path) as image:
        try:
            metadata = image.frame_metadata(0)
            coords = stage_coordinates_from_frame_metadata(metadata)
            if coords["stage_x_um"] is not None and coords["stage_y_um"] is not None:
                coords["stage_coordinate_source"] = "frame_metadata"
                return coords
        except Exception:
            pass

        try:
            return stage_coordinates_from_unstructured_metadata(image.unstructured_metadata())
        except Exception:
            return empty_stage_coordinates("unavailable")


def stage_coordinates_from_frame_metadata(metadata: Any) -> dict[str, Any]:
    """Extract XYZ stage coordinates from an nd2 FrameMetadata-like object."""
    for channel in getattr(metadata, "channels", []) or []:
        position = getattr(channel, "position", None)
        stage_position = getattr(position, "stagePositionUm", None)
        if stage_position is None:
            continue
        return {
            "stage_x_um": _as_float(getattr(stage_position, "x", None)),
            "stage_y_um": _as_float(getattr(stage_position, "y", None)),
            "stage_z_um": _as_float(getattr(stage_position, "z", None)),
            "stage_coordinate_source": "frame_metadata",
        }
    return empty_stage_coordinates("frame_metadata_missing")


def stage_coordinates_from_unstructured_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract XYZ stage coordinates from nd2 unstructured metadata when available."""
    picture_metadata = metadata.get("ImageMetadataSeqLV|0", {}).get("SLxPictureMetadata", {})
    return {
        "stage_x_um": _as_float(picture_metadata.get("XPos")),
        "stage_y_um": _as_float(picture_metadata.get("YPos")),
        "stage_z_um": _as_float(picture_metadata.get("ZPos")),
        "stage_coordinate_source": "unstructured_metadata"
        if picture_metadata
        else "unstructured_metadata_missing",
    }


def empty_stage_coordinates(source: str) -> dict[str, Any]:
    return {
        "stage_x_um": None,
        "stage_y_um": None,
        "stage_z_um": None,
        "stage_coordinate_source": source,
    }


def stage_distance_xy_um(
    reference: dict[str, Any],
    observed: dict[str, Any],
) -> float:
    """Compute XY stage distance in microns; return NaN if XY is unavailable."""
    ref_x = _as_float(reference.get("stage_x_um"))
    ref_y = _as_float(reference.get("stage_y_um"))
    obs_x = _as_float(observed.get("stage_x_um"))
    obs_y = _as_float(observed.get("stage_y_um"))
    if None in {ref_x, ref_y, obs_x, obs_y}:
        return math.nan
    return float(math.hypot(obs_x - ref_x, obs_y - ref_y))


def stage_distance_z_um(
    reference: dict[str, Any],
    observed: dict[str, Any],
) -> float:
    """Compute absolute Z stage distance in microns; return NaN if Z is unavailable."""
    ref_z = _as_float(reference.get("stage_z_um"))
    obs_z = _as_float(observed.get("stage_z_um"))
    if ref_z is None or obs_z is None:
        return math.nan
    return float(abs(obs_z - ref_z))


def classify_stage_prefilter(
    distance_xy_um: float,
    *,
    threshold_um: float = DEFAULT_STAGE_XY_THRESHOLD_UM,
) -> dict[str, Any]:
    """Classify whether a well/day passes the XY stage-coordinate prefilter."""
    if math.isnan(distance_xy_um):
        return {
            "stage_prefilter_pass": True,
            "stage_prefilter_available": False,
            "stage_prefilter_reason": "stage_coordinates_unavailable",
        }
    if distance_xy_um <= threshold_um:
        return {
            "stage_prefilter_pass": True,
            "stage_prefilter_available": True,
            "stage_prefilter_reason": "stage_xy_distance_within_threshold",
        }
    return {
        "stage_prefilter_pass": False,
        "stage_prefilter_available": True,
        "stage_prefilter_reason": "stage_xy_distance_exceeds_threshold",
    }


def build_stage_prefilter_rows(
    observations: list[dict[str, Any]],
    *,
    reference_day: int,
    threshold_um: float = DEFAULT_STAGE_XY_THRESHOLD_UM,
) -> list[dict[str, Any]]:
    """Add reference-relative stage distances and prefilter decisions to observations."""
    reference_by_well = {
        row["well"]: row for row in observations if int(row["day"]) == int(reference_day)
    }
    rows: list[dict[str, Any]] = []
    for row in observations:
        out = dict(row)
        reference = reference_by_well.get(row["well"])
        if reference is None:
            distance_xy = math.nan
            distance_z = math.nan
            classification = {
                "stage_prefilter_pass": False,
                "stage_prefilter_available": False,
                "stage_prefilter_reason": "reference_day_missing",
            }
        else:
            distance_xy = stage_distance_xy_um(reference, row)
            distance_z = stage_distance_z_um(reference, row)
            classification = classify_stage_prefilter(distance_xy, threshold_um=threshold_um)
        out["reference_day"] = int(reference_day)
        out["stage_distance_xy_um"] = distance_xy
        out["stage_distance_z_um"] = distance_z
        out["stage_xy_threshold_um"] = float(threshold_um)
        out.update(classification)
        rows.append(out)
    return rows


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)
