"""Synthetic demonstration of Moving Target Monte Carlo."""

from .forward_model import RectangularGrid, straight_ray_matrix, travel_times
from .metropolis import SamplingResult, metropolis_hastings, random_walk_metropolis
from .mtmc import nearest_neighbour_mtmc
from .proposals import GaussianIndependenceProposal, GaussianRandomWalkProposal
from .synthetic_model import (
    GaussianLogPosterior,
    SyntheticProblem,
    analytical_gaussian_posterior,
    build_synthetic_problem,
)

__all__ = [
    "GaussianLogPosterior",
    "GaussianIndependenceProposal",
    "GaussianRandomWalkProposal",
    "RectangularGrid",
    "SamplingResult",
    "SyntheticProblem",
    "analytical_gaussian_posterior",
    "build_synthetic_problem",
    "metropolis_hastings",
    "nearest_neighbour_mtmc",
    "random_walk_metropolis",
    "straight_ray_matrix",
    "travel_times",
]
