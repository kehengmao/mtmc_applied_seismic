"""Proposal kernels shared by the two demonstration samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class Proposal(Protocol):
    """Interface for a proposal draw and its Hastings correction."""

    def draw(
        self,
        current: FloatArray,
        random: np.random.Generator,
    ) -> tuple[FloatArray, float]:
        """Return a candidate and log(q(current|candidate)/q(candidate|current))."""


def _scale_array(values: ArrayLike, dimension: int) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(dimension, float(array))
    else:
        array = np.broadcast_to(array.reshape(-1), (dimension,)).copy()
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("proposal scales must be finite and positive")
    return array


@dataclass(frozen=True, slots=True)
class GaussianIndependenceProposal:
    """An independent Gaussian proposal with an exact Hastings correction."""

    mean: FloatArray
    scale: FloatArray

    @classmethod
    def create(
        cls,
        mean: ArrayLike,
        scale: ArrayLike,
    ) -> "GaussianIndependenceProposal":
        mean_array = np.asarray(mean, dtype=float).reshape(-1).copy()
        if not np.all(np.isfinite(mean_array)):
            raise ValueError("proposal mean must be finite")
        return cls(mean_array, _scale_array(scale, mean_array.size))

    def _log_kernel(self, state: FloatArray) -> float:
        standardized = (state - self.mean) / self.scale
        return -0.5 * float(standardized @ standardized)

    def draw(
        self,
        current: FloatArray,
        random: np.random.Generator,
    ) -> tuple[FloatArray, float]:
        candidate = random.normal(loc=self.mean, scale=self.scale)
        log_reverse_over_forward = self._log_kernel(current) - self._log_kernel(
            candidate
        )
        return candidate, log_reverse_over_forward


@dataclass(frozen=True, slots=True)
class GaussianRandomWalkProposal:
    """A symmetric Gaussian random walk with a zero Hastings correction."""

    scale: FloatArray

    @classmethod
    def create(
        cls,
        scale: ArrayLike,
        dimension: int,
    ) -> "GaussianRandomWalkProposal":
        return cls(_scale_array(scale, dimension))

    def draw(
        self,
        current: FloatArray,
        random: np.random.Generator,
    ) -> tuple[FloatArray, float]:
        return current + random.normal(scale=self.scale), 0.0
