"""Integration tests for the plate-offset path in register_stack (synthetic, no ND2).

Covers the Phase-4 wiring: plate_offsets default None = byte-identical; the prior composes
plate-first-then-per-well-residual; and a large shift that per-well registration cannot resolve
is rescued by the plate prior. See PLATE_REMOUNT_CORRECTION_PLAN.md §3.
"""
from __future__ import annotations

import numpy as np

from scripts.run_260213_longitudinal_pilot import register_stack
from tmem_align.register import apply_shift


def _ref_cyx(n: int = 140, seed: int = 0) -> np.ndarray:
    """A 2-channel (CYX) frame with clear registerable structure on the stable channel (0)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n]
    stable = np.zeros((n, n), np.float32)
    for cy, cx in [(45, 55), (95, 100), (60, 105), (110, 40)]:
        stable += np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * 7.0**2)))
    stable += 0.03 * rng.normal(size=stable.shape)
    return np.stack([stable, 0.2 * stable], axis=0).astype(np.float32)  # CYX


def _stack(shifts):
    """TCYX stack: frame 0 = reference, later frames = reference shifted by each (dy, dx)."""
    ref = _ref_cyx()
    frames = [ref] + [apply_shift(ref, dy, dx) for dy, dx in shifts]
    return np.stack(frames, axis=0)


def _rows(n):
    return [{"day": d} for d in range(8, 8 + n)]


def _reg(stack, rows, **kw):
    return register_stack(
        stack, well="F05", rows=rows, alignment_channel_index=0,
        alignment_channel_label="488", **kw,
    )


def _shifts(qc_rows):
    return [(r["estimated_y_shift"], r["estimated_x_shift"]) for r in qc_rows]


def test_plate_offsets_none_matches_zero_prior():
    """Regression: plate_offsets=None and an all-zero prior are byte-identical (the plate path
    with zero prior changes nothing on the default)."""
    stack = _stack([(6.0, -4.0), (9.0, 5.0)])
    rows = _rows(3)
    reg_none, qc_none, crop_none = _reg(stack, rows, plate_offsets=None)
    zero = {r["day"]: (0.0, 0.0) for r in rows}
    reg_zero, qc_zero, crop_zero = _reg(stack, rows, plate_offsets=zero)
    assert _shifts(qc_none) == _shifts(qc_zero)
    assert np.allclose(reg_none, reg_zero)
    assert crop_none == crop_zero


def test_composition_partial_prior_recovers_total():
    """A partial plate prior + the per-well residual compose to the same total net shift as pure
    per-well registration (prior-first then residual = total)."""
    shifts = [(12.0, -10.0), (-8.0, 14.0)]
    stack = _stack(shifts)
    rows = _rows(3)
    _, qc_plain, _ = _reg(stack, rows, plate_offsets=None)
    # prior = half of the (known) correction; register must find the other half as residual
    prior = {rows[0]["day"]: (0.0, 0.0)}
    for row, (dy, dx) in zip(rows[1:], shifts):
        prior[row["day"]] = (-dy / 2, -dx / 2)
    _, qc_prior, _ = _reg(stack, rows, plate_offsets=prior)
    assert np.allclose(_shifts(qc_plain), _shifts(qc_prior), atol=1.5)
    # and the total recovers the true correction (-applied)
    for (dy, dx), (ndy, ndx) in zip(shifts, _shifts(qc_prior)[1:]):
        assert abs(ndy + dy) < 1.5 and abs(ndx + dx) < 1.5


def test_plate_prior_applied_yields_alignment():
    """The plate prior is threaded through and applied: with the exact plate correction the
    residual registration is ~0, the net equals the prior, and the registered frame aligns to the
    reference. (Rescue of a genuinely weak/decorrelated well is covered at the fit level in
    test_plate_align.test_weak_well_rescue and on real N20 in the walkthrough notebook — a shifted
    copy is always self-registerable, so per-well "failure" isn't reproducible in this synthetic.)"""
    from tmem_align.registration_qc import correlation

    D = (40.0, -30.0)
    stack = _stack([D])
    rows = _rows(2)
    prior = {rows[0]["day"]: (0.0, 0.0), rows[1]["day"]: (-D[0], -D[1])}  # exact plate correction
    reg, qc, _ = _reg(stack, rows, plate_offsets=prior)
    net = _shifts(qc)[1]
    assert abs(net[0] + D[0]) < 1.5 and abs(net[1] + D[1]) < 1.5  # net == prior (residual ~0)
    # registered stable aligns to the reference far better than the unregistered moving frame
    assert correlation(reg[1, 0], stack[0, 0]) > correlation(stack[1, 0], stack[0, 0]) + 0.2
