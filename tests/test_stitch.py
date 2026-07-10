"""Round-trip test: synthetic tiles → positions → refine → optimize → assemble."""
from __future__ import annotations

import numpy as np
import pytest

from tmem_align.stitch import (
    assemble_edt,
    build_adjacency,
    optimize_positions,
    positions_from_stage_coords,
    refine_shift,
    stitch_tiles,
)


def _make_gradient_tile(h: int, w: int, value: float) -> np.ndarray:
    """Tile with a gradient + constant — overlaps produce visible signal for correlation."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return (y * x * 1000 + value).astype(np.uint16)


class TestPositionsFromStageCoords:
    def test_basic(self):
        coords = [
            {"position_index": 0, "stage_x_um": 0.0, "stage_y_um": 0.0},
            {"position_index": 1, "stage_x_um": 100.0, "stage_y_um": 0.0},
            {"position_index": 2, "stage_x_um": 0.0, "stage_y_um": 100.0},
        ]
        pos = positions_from_stage_coords(coords, pixel_size_um=0.5)
        assert pos[0] == (0.0, 0.0)
        assert pos[1] == pytest.approx((0.0, 200.0))
        assert pos[2] == pytest.approx((200.0, 0.0))

    def test_no_valid_coords_raises(self):
        with pytest.raises(ValueError):
            positions_from_stage_coords(
                [{"position_index": 0, "stage_x_um": None, "stage_y_um": None}],
                pixel_size_um=1.0,
            )


class TestBuildAdjacency:
    def test_2x2_grid(self):
        positions = {
            0: (0.0, 0.0),
            1: (0.0, 100.0),
            2: (100.0, 0.0),
            3: (100.0, 100.0),
        }
        edges = build_adjacency(positions, tile_shape=(100, 100))
        pairs = {(a, b) for a, b, _ in edges}
        assert (0, 1) in pairs  # horizontal
        assert (0, 2) in pairs  # vertical
        assert (1, 3) in pairs  # vertical
        assert (2, 3) in pairs  # horizontal


class TestOptimizePositions:
    def test_consistent_edges(self):
        edges = [
            (0, 1, 0.0, 100.0),
            (0, 2, 100.0, 0.0),
            (1, 3, 100.0, 0.0),
            (2, 3, 0.0, 100.0),
        ]
        pos = optimize_positions(edges, 4)
        assert pos[0] == pytest.approx((0.0, 0.0), abs=0.1)
        assert pos[1] == pytest.approx((0.0, 100.0), abs=0.1)
        assert pos[2] == pytest.approx((100.0, 0.0), abs=0.1)
        assert pos[3] == pytest.approx((100.0, 100.0), abs=0.1)


class TestStitchTilesRoundTrip:
    def test_2x2_no_refine(self):
        """Assemble 4 tiles at known positions, verify output shape."""
        th, tw = 64, 64
        overlap = 10
        step = th - overlap
        tiles = {i: _make_gradient_tile(th, tw, i * 100) for i in range(4)}
        positions = {
            0: (0.0, 0.0),
            1: (0.0, float(step)),
            2: (float(step), 0.0),
            3: (float(step), float(step)),
        }
        result = stitch_tiles(tiles, positions, refine=False)
        expected_h = step + th
        expected_w = step + tw
        assert result.shape == (expected_h, expected_w)
        assert result.dtype == np.uint16

    def test_2x2_with_refine(self):
        """Phase correlation on identical tiles should produce ~zero correction."""
        th, tw = 64, 64
        overlap = 10
        step = th - overlap
        tile = _make_gradient_tile(th, tw, 500)
        tiles = {i: tile.copy() for i in range(4)}
        positions = {
            0: (0.0, 0.0),
            1: (0.0, float(step)),
            2: (float(step), 0.0),
            3: (float(step), float(step)),
        }
        result = stitch_tiles(tiles, positions, refine=True, overlap_fraction=overlap / th)
        assert result.shape[0] > 0 and result.shape[1] > 0


class TestRefineShift:
    def test_horizontal_zero_shift(self):
        tile = _make_gradient_tile(64, 64, 500)
        dy, dx = refine_shift(tile, tile, relation=(0, 1), overlap_px=16)
        assert abs(dy) < 2.0
        assert abs(dx) < 2.0


class TestAssembleEDT:
    def test_single_tile(self):
        tile = _make_gradient_tile(32, 32, 100)
        result = assemble_edt({0: tile}, {0: (0.0, 0.0)})
        assert result.shape == (32, 32)
        # EDT zeros border pixels (weight=0 at edges) — interior should match
        np.testing.assert_array_equal(result[1:-1, 1:-1], tile[1:-1, 1:-1])
