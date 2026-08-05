"""A conventional random-walk Metropolis baseline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .proposals import GaussianRandomWalkProposal, Proposal


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
LogTarget = Callable[[FloatArray], float]


@dataclass(slots=True)
class SamplingResult:
    """Samples and diagnostics produced by either demonstration sampler."""

    name: str
    samples: FloatArray
    log_targets: FloatArray
    accepted: BoolArray
    target_evaluations: int
    elapsed_seconds: float
    evaluated_points: FloatArray | None = None
    evaluated_log_targets: FloatArray | None = None

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted)) if self.accepted.size else 0.0


def _prepare_state(
    initial: ArrayLike,
    n_steps: int,
) -> FloatArray:
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    state = np.asarray(initial, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(state)):
        raise ValueError("initial state must be finite")
    return state


def _accept(log_ratio: float, random: np.random.Generator) -> bool:
    if np.isnan(log_ratio):
        raise FloatingPointError("acceptance log ratio is NaN")
    uniform = max(float(random.random()), np.finfo(float).tiny)
    return np.log(uniform) < min(0.0, log_ratio)


def metropolis_hastings(
    log_target: LogTarget,
    initial: ArrayLike,
    *,
    proposal: Proposal,
    n_steps: int,
    random: np.random.Generator,
) -> SamplingResult:
    """Sample with a proposal that supplies its Hastings correction."""

    current = _prepare_state(initial, n_steps)
    samples = np.empty((n_steps, current.size), dtype=float)
    log_targets = np.empty(n_steps, dtype=float)
    accepted = np.zeros(n_steps - 1, dtype=bool)

    start = perf_counter()
    current_log_target = float(log_target(current))
    if not np.isfinite(current_log_target):
        raise ValueError("initial log target must be finite")
    samples[0] = current
    log_targets[0] = current_log_target
    target_evaluations = 1

    for index in range(1, n_steps):
        candidate, proposal_log_ratio = proposal.draw(current, random)
        candidate = np.asarray(candidate, dtype=float).reshape(-1)
        if candidate.shape != current.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("proposal returned an invalid candidate")
        candidate_log_target = float(log_target(candidate))
        target_evaluations += 1
        if _accept(
            candidate_log_target - current_log_target + proposal_log_ratio,
            random,
        ):
            current = candidate
            current_log_target = candidate_log_target
            accepted[index - 1] = True
        samples[index] = current
        log_targets[index] = current_log_target

    return SamplingResult(
        name="Metropolis-Hastings",
        samples=samples,
        log_targets=log_targets,
        accepted=accepted,
        target_evaluations=target_evaluations,
        elapsed_seconds=perf_counter() - start,
    )


def random_walk_metropolis(
    log_target: LogTarget,
    initial: ArrayLike,
    *,
    proposal_scale: ArrayLike,
    n_steps: int,
    random: np.random.Generator,
) -> SamplingResult:
    """Convenience wrapper for a symmetric Gaussian random walk."""

    state = np.asarray(initial, dtype=float).reshape(-1)
    proposal = GaussianRandomWalkProposal.create(proposal_scale, state.size)
    return metropolis_hastings(
        log_target,
        state,
        proposal=proposal,
        n_steps=n_steps,
        random=random,
    )
