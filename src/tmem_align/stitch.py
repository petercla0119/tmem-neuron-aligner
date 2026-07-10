from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from .io import find_images, normalize_to_2d, read_image, write_ome_tiff


# ---------------------------------------------------------------------------
# Legacy grid stitcher (kept for backward compat)
# ---------------------------------------------------------------------------


def stitch_grid(
    tile_paths: list[str | Path],
    grid_rows: int,
    grid_cols: int,
    overlap_fraction: float = 0.10,
    snake_order: bool = False,
) -> np.ndarray:
    """Simple grid stitcher — nominal overlap, no refinement."""
    if len(tile_paths) != grid_rows * grid_cols:
        raise ValueError(f"Expected {grid_rows * grid_cols} tiles, found {len(tile_paths)}")

    tiles = [normalize_to_2d(read_image(p)) for p in tqdm(tile_paths, desc="Reading tiles")]
    th, tw = tiles[0].shape
    step_y = int(round(th * (1 - overlap_fraction)))
    step_x = int(round(tw * (1 - overlap_fraction)))
    canvas = np.zeros(
        (step_y * (grid_rows - 1) + th, step_x * (grid_cols - 1) + tw), dtype=np.float32
    )
    weight = np.zeros_like(canvas, dtype=np.float32)

    for r in range(grid_rows):
        cols = range(grid_cols)
        if snake_order and r % 2 == 1:
            cols = reversed(range(grid_cols))
        for c_display, c in enumerate(cols):
            tile_index = r * grid_cols + c_display if snake_order else r * grid_cols + c
            tile = tiles[tile_index].astype(np.float32)
            y0 = r * step_y
            x0 = c * step_x
            canvas[y0 : y0 + th, x0 : x0 + tw] += tile
            weight[y0 : y0 + th, x0 : x0 + tw] += 1

    weight[weight == 0] = 1
    return (canvas / weight).astype(tiles[0].dtype)


def stitch_folder_to_ometiff(
    tile_folder: str | Path,
    output_path: str | Path,
    grid_rows: int,
    grid_cols: int,
    overlap_fraction: float = 0.10,
    snake_order: bool = False,
) -> Path:
    tile_paths = find_images(tile_folder)
    stitched = stitch_grid(tile_paths, grid_rows, grid_cols, overlap_fraction, snake_order)
    write_ome_tiff(output_path, stitched, axes="YX")
    return Path(output_path)


# ---------------------------------------------------------------------------
# Tile-based stitcher
# ---------------------------------------------------------------------------


def positions_from_stage_coords(
    coords: list[dict],
    pixel_size_um: float,
) -> dict[int, tuple[float, float]]:
    """Convert stage µm coordinates to pixel positions, origin-normalized.

    Returns {position_index: (y_px, x_px)}.
    """
    valid = [(c["position_index"], c["stage_x_um"], c["stage_y_um"]) for c in coords
             if c["stage_x_um"] is not None and c["stage_y_um"] is not None]
    if not valid:
        raise ValueError("No valid stage coordinates found")
    xs = [v[1] for v in valid]
    ys = [v[2] for v in valid]
    min_x, min_y = min(xs), min(ys)
    return {
        idx: ((y - min_y) / pixel_size_um, (x - min_x) / pixel_size_um)
        for idx, x, y in valid
    }


def build_adjacency(
    positions: dict[int, tuple[float, float]],
    tile_shape: tuple[int, int],
) -> list[tuple[int, int, tuple[int, int]]]:
    """Find horizontal/vertical neighbor pairs from tile positions.

    Returns [(idx_a, idx_b, relation), ...] where relation is (dy_sign, dx_sign).
    """
    th, tw = tile_shape
    indices = sorted(positions.keys())
    edges = []
    for i, a in enumerate(indices):
        ya, xa = positions[a]
        for b in indices[i + 1:]:
            yb, xb = positions[b]
            dy, dx = yb - ya, xb - xa
            # ponytail: 0.7 tolerance on expected tile step — catches real grids
            if abs(dy) < th * 0.3 and abs(dx - tw) < tw * 0.3:
                edges.append((a, b, (0, 1)))
            elif abs(dx) < tw * 0.3 and abs(dy - th) < th * 0.3:
                edges.append((a, b, (1, 0)))
    return edges


def refine_shift(
    tile_a: np.ndarray,
    tile_b: np.ndarray,
    relation: tuple[int, int],
    overlap_px: int,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """Phase-correlation refinement on overlap ROIs between adjacent tiles."""
    if relation == (0, 1):
        roi_a = tile_a[:, -overlap_px:]
        roi_b = tile_b[:, :overlap_px]
    elif relation == (1, 0):
        roi_a = tile_a[-overlap_px:, :]
        roi_b = tile_b[:overlap_px, :]
    else:
        raise ValueError(f"Unknown relation: {relation}")

    shift, _, _ = phase_cross_correlation(
        roi_a.astype(np.float32),
        roi_b.astype(np.float32),
        upsample_factor=upsample_factor,
    )
    return float(shift[0]), float(shift[1])


def optimize_positions(
    edges: list[tuple[int, int, float, float]],
    n_tiles: int,
    index_map: dict[int, int] | None = None,
) -> dict[int, tuple[float, float]]:
    """Solve globally consistent positions from pairwise shifts via least-squares.

    edges: [(idx_a, idx_b, measured_dy, measured_dx), ...]
    Returns {original_index: (y, x)}.
    """
    if index_map is None:
        all_indices = sorted({e[0] for e in edges} | {e[1] for e in edges})
        index_map = {idx: i for i, idx in enumerate(all_indices)}
        n_tiles = len(all_indices)

    inv_map = {v: k for k, v in index_map.items()}
    n_edges = len(edges)
    rows_y, cols_y, vals_y, rhs_y = [], [], [], []
    rows_x, cols_x, vals_x, rhs_x = [], [], [], []

    for row_idx, (a, b, dy, dx) in enumerate(edges):
        ia, ib = index_map[a], index_map[b]
        rows_y.extend([row_idx, row_idx])
        cols_y.extend([ia, ib])
        vals_y.extend([-1.0, 1.0])
        rhs_y.append(dy)
        rows_x.extend([row_idx, row_idx])
        cols_x.extend([ia, ib])
        vals_x.extend([-1.0, 1.0])
        rhs_x.append(dx)

    # pin tile 0 at origin
    rows_y.extend([n_edges])
    cols_y.extend([0])
    vals_y.extend([1.0])
    rhs_y.append(0.0)
    rows_x.extend([n_edges])
    cols_x.extend([0])
    vals_x.extend([1.0])
    rhs_x.append(0.0)

    A_y = coo_matrix((vals_y, (rows_y, cols_y)), shape=(n_edges + 1, n_tiles))
    A_x = coo_matrix((vals_x, (rows_x, cols_x)), shape=(n_edges + 1, n_tiles))
    pos_y = lsqr(A_y.tocsr(), np.array(rhs_y))[0]
    pos_x = lsqr(A_x.tocsr(), np.array(rhs_x))[0]

    return {inv_map[i]: (float(pos_y[i]), float(pos_x[i])) for i in range(n_tiles)}


def _edt_weights(shape: tuple[int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    w = distance_transform_edt(mask).astype(np.float32)
    w /= w.max() or 1.0
    return w


def assemble_edt(
    tiles: dict[int, np.ndarray],
    positions: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Assemble tiles with EDT-weighted blending."""
    tile_shape = next(iter(tiles.values())).shape[:2]
    th, tw = tile_shape
    edt_w = _edt_weights(tile_shape)

    all_y = [int(round(p[0])) for p in positions.values()]
    all_x = [int(round(p[1])) for p in positions.values()]
    out_h = max(all_y) + th
    out_w = max(all_x) + tw

    canvas = np.zeros((out_h, out_w), dtype=np.float64)
    weight = np.zeros((out_h, out_w), dtype=np.float64)

    for idx in sorted(tiles.keys()):
        tile = tiles[idx].astype(np.float64)
        if tile.ndim > 2:
            tile = normalize_to_2d(tile.astype(np.float32)).astype(np.float64)
        y0 = int(round(positions[idx][0]))
        x0 = int(round(positions[idx][1]))
        canvas[y0 : y0 + th, x0 : x0 + tw] += tile * edt_w
        weight[y0 : y0 + th, x0 : x0 + tw] += edt_w

    weight[weight == 0] = 1.0
    result = canvas / weight
    sample = next(iter(tiles.values()))
    return result.astype(sample.dtype)


def stitch_tiles(
    tiles: dict[int, np.ndarray],
    positions: dict[int, tuple[float, float]],
    refine: bool = True,
    overlap_fraction: float = 0.10,
    upsample_factor: int = 10,
) -> np.ndarray:
    """Stitch pre-loaded tiles using positions, optional phase-correlation refinement, EDT blend."""
    tile_shape = next(iter(tiles.values())).shape[:2]
    th, tw = tile_shape

    if refine:
        edges_raw = build_adjacency(positions, tile_shape)
        overlap_px = max(int(round(min(th, tw) * overlap_fraction)), 16)
        measured_edges = []
        for a, b, relation in tqdm(edges_raw, desc="Refining shifts"):
            tile_a = normalize_to_2d(tiles[a].astype(np.float32))
            tile_b = normalize_to_2d(tiles[b].astype(np.float32))
            ref_dy = positions[b][0] - positions[a][0]
            ref_dx = positions[b][1] - positions[a][1]
            try:
                local_dy, local_dx = refine_shift(
                    tile_a, tile_b, relation, overlap_px, upsample_factor
                )
            except Exception:
                measured_edges.append((a, b, ref_dy, ref_dx))
                continue
            if relation == (0, 1):
                measured_dy = ref_dy + local_dy
                measured_dx = (tw - overlap_px) + local_dx
            else:
                measured_dy = (th - overlap_px) + local_dy
                measured_dx = ref_dx + local_dx
            measured_edges.append((a, b, measured_dy, measured_dx))

        if measured_edges:
            positions = optimize_positions(measured_edges, len(tiles))

    # origin-normalize
    min_y = min(p[0] for p in positions.values())
    min_x = min(p[1] for p in positions.values())
    positions = {k: (v[0] - min_y, v[1] - min_x) for k, v in positions.items()}

    return assemble_edt(tiles, positions)


def stitch_nd2(
    nd2_path: str | Path,
    output_path: str | Path,
    channel: int = 0,
    refine: bool = True,
    z_project: str = "max",
    z_index: int | None = None,
    pixel_size_um: float | None = None,
    overlap_fraction: float = 0.10,
) -> Path:
    """Stitch all FOVs from a multi-position ND2 file to OME-TIFF."""
    from .nd2_tools import read_fov_positions, read_fov_tile, inspect_nd2

    nd2_path = Path(nd2_path).expanduser().resolve()
    info = inspect_nd2(nd2_path)
    n_pos = info["position_count"]
    if pixel_size_um is None:
        voxel = info.get("voxel_size")
        if voxel and voxel.get("x_um"):
            pixel_size_um = voxel["x_um"]
        else:
            raise ValueError("pixel_size_um required — not found in ND2 metadata")

    coords = read_fov_positions(nd2_path)
    positions = positions_from_stage_coords(coords, pixel_size_um)

    tiles = {}
    for p in tqdm(range(n_pos), desc="Loading FOVs"):
        if p not in positions:
            continue
        tiles[p] = read_fov_tile(nd2_path, p, channel=channel,
                                 z_project=z_project, z_index=z_index)

    stitched = stitch_tiles(tiles, positions, refine=refine,
                            overlap_fraction=overlap_fraction)
    write_ome_tiff(output_path, stitched, axes="YX", pixel_size_um=pixel_size_um)
    return Path(output_path)
