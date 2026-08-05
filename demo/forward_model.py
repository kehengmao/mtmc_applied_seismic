"""Synthetic straight-ray travel-time calculations.

This module uses midpoint quadrature along each source-receiver segment. It is
intentionally small and depends only on the geometry defined in this demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RectangularGrid:
    """A regular two-dimensional grid with row-major cell ordering."""

    width: float
    height: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("grid cell counts must be positive")

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def cell_width(self) -> float:
        return self.width / self.nx

    @property
    def cell_height(self) -> float:
        return self.height / self.ny

    def reshape_cells(self, values: ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=float)
        if array.size != self.n_cells:
            raise ValueError(f"expected {self.n_cells} cell values")
        return array.reshape(self.ny, self.nx)


def _point_array(name: str, values: ArrayLike) -> FloatArray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_rays, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def straight_ray_matrix(
    grid: RectangularGrid,
    sources: ArrayLike,
    receivers: ArrayLike,
    *,
    samples_per_ray: int = 1600,
) -> FloatArray:
    """Approximate cell path lengths for straight source-receiver segments.

    Each ray is divided into equal-length pieces. The midpoint of each piece
    identifies the grid cell that receives that piece's path length.
    """

    source_points = _point_array("sources", sources)
    receiver_points = _point_array("receivers", receivers)
    if source_points.shape != receiver_points.shape:
        raise ValueError("sources and receivers must define the same number of rays")
    if samples_per_ray < 10:
        raise ValueError("samples_per_ray must be at least 10")

    all_points = np.vstack((source_points, receiver_points))
    if (
        np.any(all_points[:, 0] < 0)
        or np.any(all_points[:, 0] > grid.width)
        or np.any(all_points[:, 1] < 0)
        or np.any(all_points[:, 1] > grid.height)
    ):
        raise ValueError("all ray endpoints must lie inside the grid")

    fractions = (np.arange(samples_per_ray, dtype=float) + 0.5) / samples_per_ray
    matrix = np.zeros((source_points.shape[0], grid.n_cells), dtype=float)

    for ray_index, (source, receiver) in enumerate(
        zip(source_points, receiver_points, strict=True)
    ):
        displacement = receiver - source
        ray_length = float(np.linalg.norm(displacement))
        if ray_length == 0:
            raise ValueError("a source and receiver cannot occupy the same point")

        midpoints = source + fractions[:, None] * displacement
        x_cells = np.floor(midpoints[:, 0] / grid.cell_width).astype(int)
        y_cells = np.floor(midpoints[:, 1] / grid.cell_height).astype(int)
        x_cells = np.clip(x_cells, 0, grid.nx - 1)
        y_cells = np.clip(y_cells, 0, grid.ny - 1)
        flat_cells = y_cells * grid.nx + x_cells

        piece_length = ray_length / samples_per_ray
        matrix[ray_index] = np.bincount(
            flat_cells,
            weights=np.full(samples_per_ray, piece_length),
            minlength=grid.n_cells,
        )

    return matrix


def travel_times(ray_matrix: ArrayLike, slowness: ArrayLike) -> FloatArray:
    """Calculate travel times from path lengths and cell slowness values."""

    matrix = np.asarray(ray_matrix, dtype=float)
    model = np.asarray(slowness, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != model.size:
        raise ValueError("ray matrix and slowness dimensions are inconsistent")
    return matrix @ model
