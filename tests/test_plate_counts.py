import runpy
from pathlib import Path

import pytest


@pytest.fixture
def plate():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_plate_counts.py"))


def manifest(plate):
    return [
        {"well": w, "day": d, "relative_path": f"{w}_{d}.nd2", "size_bytes": 1}
        for w in plate["WELLS"]
        for d in plate["DAYS"]
    ]


def test_incomplete_transfer_rejected(tmp_path, plate):
    with pytest.raises(ValueError, match="finish the data copy"):
        plate["check_inputs"](tmp_path, manifest(plate))


def test_partial_manifest_rejected(tmp_path, plate):
    with pytest.raises(ValueError, match="96 wells"):
        plate["check_inputs"](tmp_path, manifest(plate)[:-1])


def test_full_manifest_and_truncated_file(tmp_path, plate):
    rows = manifest(plate)
    for r in rows:
        (tmp_path / r["relative_path"]).write_bytes(b"x")
    assert plate["check_inputs"](tmp_path, rows) == 1248
    (tmp_path / rows[0]["relative_path"]).write_bytes(b"")
    with pytest.raises(ValueError, match="Incomplete or changed size"):
        plate["check_inputs"](tmp_path, rows)


def test_parent_escape_rejected(tmp_path, plate):
    rows = manifest(plate)
    rows[0]["relative_path"] = "../outside.nd2"
    with pytest.raises(ValueError, match="Unsafe"):
        plate["check_inputs"](tmp_path, rows)
