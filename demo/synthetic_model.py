"""Create a self-contained synthetic seismic inverse problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .forward_model import RectangularGrid, straight_ray_matrix, travel_times


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticProblem:
    """All arrays required by the generated low-dimensional demonstration."""

    grid: RectangularGrid
    sources: FloatArray
    receivers: FloatArray
    ray_matrix: FloatArray
    true_velocity: FloatArray
    true_slowness: FloatArray
    observed_times: FloatArray
    noise_std: float
    prior_mean: FloatArray
    prior_std: float
    posterior_mean: FloatArray
    posterior_covariance: FloatArray


class GaussianLogPosterior:
    """Unnormalised Gaussian log posterior with an evaluation counter."""

    def __init__(
        self,
        ray_matrix: ArrayLike,
        observed_times: ArrayLike,
        noise_std: float,
        prior_mean: ArrayLike,
        prior_std: float,
    ) -> None:
        self.ray_matrix = np.asarray(ray_matrix, dtype=float)
        self.observed_times = np.asarray(observed_times, dtype=float).reshape(-1)
        self.prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
        self.noise_std = float(noise_std)
        self.prior_std = float(prior_std)
        self.evaluations = 0

        if self.ray_matrix.ndim != 2:
            raise ValueError("ray_matrix must be two-dimensional")
        if self.ray_matrix.shape[0] != self.observed_times.size:
            raise ValueError("one observation is required for each ray")
        if self.ray_matrix.shape[1] != self.prior_mean.size:
            raise ValueError("prior and model dimensions are inconsistent")
        if self.noise_std <= 0 or self.prior_std <= 0:
            raise ValueError("standard deviations must be positive")

    def __call__(self, slowness: ArrayLike) -> float:
        model = np.asarray(slowness, dtype=float).reshape(-1)
        if model.shape != self.prior_mean.shape:
            raise ValueError("unexpected model dimension")
        self.evaluations += 1
        data_residual = (self.ray_matrix @ model - self.observed_times) / self.noise_std
        prior_residual = (model - self.prior_mean) / self.prior_std
        return -0.5 * float(data_residual @ data_residual + prior_residual @ prior_residual)


def analytical_gaussian_posterior(
    ray_matrix: ArrayLike,
    observed_times: ArrayLike,
    noise_std: float,
    prior_mean: ArrayLike,
    prior_std: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the exact posterior mean and covariance for the linear model."""

    matrix = np.asarray(ray_matrix, dtype=float)
    data = np.asarray(observed_times, dtype=float).reshape(-1)
    mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    if matrix.shape != (data.size, mean.size):
        raise ValueError("inconsistent linear-Gaussian dimensions")
    if noise_std <= 0 or prior_std <= 0:
        raise ValueError("standard deviations must be positive")

    precision = matrix.T @ matrix / noise_std**2
    precision += np.eye(mean.size) / prior_std**2
    right_hand_side = matrix.T @ data / noise_std**2
    right_hand_side += mean / prior_std**2

    posterior_mean = np.linalg.solve(precision, right_hand_side)
    posterior_covariance = np.linalg.solve(precision, np.eye(mean.size))
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    return posterior_mean, posterior_covariance


def _cross_boundary_acquisition(
    grid: RectangularGrid,
) -> tuple[FloatArray, FloatArray]:
    left_y = np.linspace(0.08 * grid.height, 0.92 * grid.height, 6)
    right_y = np.linspace(0.05 * grid.height, 0.95 * grid.height, 8)
    left_sites = np.column_stack((np.zeros(left_y.size), left_y))
    right_sites = np.column_stack(
        (np.full(right_y.size, grid.width), right_y)
    )
    horizontal_sources = np.repeat(left_sites, right_y.size, axis=0)
    horizontal_receivers = np.tile(right_sites, (left_y.size, 1))

    top_x = np.linspace(0.10 * grid.width, 0.90 * grid.width, 4)
    bottom_x = np.linspace(0.06 * grid.width, 0.94 * grid.width, 5)
    top_sites = np.column_stack((top_x, np.full(top_x.size, grid.height)))
    bottom_sites = np.column_stack((bottom_x, np.zeros(bottom_x.size)))
    vertical_sources = np.repeat(top_sites, bottom_x.size, axis=0)
    vertical_receivers = np.tile(bottom_sites, (top_x.size, 1))

    sources = np.vstack((horizontal_sources, vertical_sources))
    receivers = np.vstack((horizontal_receivers, vertical_receivers))
    return sources, receivers


def build_synthetic_problem(
    *,
    seed: int = 2026,
    samples_per_ray: int = 1600,
) -> SyntheticProblem:
    """Generate geometry, observations, prior, and analytical reference result."""

    grid = RectangularGrid(width=1200.0, height=800.0, nx=2, ny=2)
    sources, receivers = _cross_boundary_acquisition(grid)
    ray_matrix = straight_ray_matrix(
        grid,
        sources,
        receivers,
        samples_per_ray=samples_per_ray,
    )

    true_velocity_grid = np.array(
        [
            [2020.0, 1975.0],
            [2050.0, 1950.0],
        ]
    )
    true_velocity = true_velocity_grid.reshape(-1)
    true_slowness = 1.0 / true_velocity

    noise_std = 0.01
    random = np.random.default_rng(seed)
    observed_times = travel_times(ray_matrix, true_slowness)
    observed_times = observed_times + random.normal(
        scale=noise_std,
        size=observed_times.size,
    )

    prior_mean = np.full(grid.n_cells, 1.0 / 2000.0)
    prior_std = 3.0e-5
    posterior_mean, posterior_covariance = analytical_gaussian_posterior(
        ray_matrix,
        observed_times,
        noise_std,
        prior_mean,
        prior_std,
    )

    return SyntheticProblem(
        grid=grid,
        sources=sources,
        receivers=receivers,
        ray_matrix=ray_matrix,
        true_velocity=true_velocity,
        true_slowness=true_slowness,
        observed_times=observed_times,
        noise_std=noise_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
    )
