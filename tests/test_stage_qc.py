import math

from tmem_align.stage_qc import (
    build_stage_prefilter_rows,
    classify_stage_prefilter,
    stage_distance_xy_um,
)


def test_stage_distance_xy_um():
    reference = {"stage_x_um": 0.0, "stage_y_um": 0.0}
    observed = {"stage_x_um": 3.0, "stage_y_um": 4.0}
    assert stage_distance_xy_um(reference, observed) == 5.0


def test_stage_distance_unavailable_is_nan():
    distance = stage_distance_xy_um({"stage_x_um": None, "stage_y_um": 0.0}, {"stage_x_um": 1.0, "stage_y_um": 1.0})
    assert math.isnan(distance)
    classification = classify_stage_prefilter(distance)
    assert classification["stage_prefilter_pass"] is True
    assert classification["stage_prefilter_available"] is False


def test_build_stage_prefilter_rows_flags_large_xy_shift():
    rows = [
        {"well": "E05", "day": 8, "stage_x_um": 0.0, "stage_y_um": 0.0},
        {"well": "E05", "day": 25, "stage_x_um": 10.0, "stage_y_um": 0.0},
    ]
    result = build_stage_prefilter_rows(rows, reference_day=8, threshold_um=5.0)
    day25 = [row for row in result if row["day"] == 25][0]
    assert day25["stage_prefilter_pass"] is False
    assert day25["stage_prefilter_reason"] == "stage_xy_distance_exceeds_threshold"
