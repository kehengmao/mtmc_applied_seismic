"""Nearest-neighbour Moving Target Monte Carlo implementation."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from .metropolis import (
    LogTarget,
    SamplingResult,
    _accept,
    _prepare_state,
)
from .proposals import Proposal


def nearest_neighbour_mtmc(
    log_target: LogTarget,
    initial: ArrayLike,
    *,
    proposal: Proposal,
    n_steps: int,
    random: np.random.Generator,
    metric_scale: ArrayLike | None = None,
) -> SamplingResult:
    """Sample using the paper's nearest-neighbour MTMC construction.

    The target value assigned to a candidate is the true target value stored at
    its nearest evaluated point. A true evaluation is made only after
    preliminary acceptance. The proposal supplies the complete Hastings
    correction, so symmetric and asymmetric kernels are both supported.
    """

    current = _prepare_state(initial, n_steps)
    if metric_scale is None:
        metric = np.ones(current.size, dtype=float)
    else:
        metric_array = np.asarray(metric_scale, dtype=float)
        if metric_array.ndim == 0:
            metric = np.full(current.size, float(metric_array))
        else:
            metric = np.broadcast_to(
                metric_array.reshape(-1), current.shape
            ).copy()
        if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
            raise ValueError("metric scales must be finite and positive")

    samples = np.empty((n_steps, current.size), dtype=float)
    log_targets = np.empty(n_steps, dtype=float)
    accepted = np.zeros(n_steps - 1, dtype=bool)
    archive_points = np.empty((n_steps, current.size), dtype=float)
    archive_logs = np.empty(n_steps, dtype=float)

    start = perf_counter()
    current_log_target = float(log_target(current))
    if not np.isfinite(current_log_target):
        raise ValueError("initial log target must be finite")

    archive_size = 1
    archive_points[0] = current
    archive_logs[0] = current_log_target
    samples[0] = current
    log_targets[0] = current_log_target

    for index in range(1, n_steps):
        candidate, proposal_log_ratio = proposal.draw(current, random)
        candidate = np.asarray(candidate, dtype=float).reshape(-1)
        if candidate.shape != current.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("proposal returned an invalid candidate")
        scaled_offsets = (archive_points[:archive_size] - candidate) / metric
        squared_distances = np.einsum(
            "ij,ij->i",
            scaled_offsets,
            scaled_offsets,
        )
        nearest_index = int(np.argmin(squared_distances))
        reconstructed_candidate_log = float(archive_logs[nearest_index])

        if _accept(
            reconstructed_candidate_log
            - current_log_target
            + proposal_log_ratio,
            random,
        ):
            candidate_log_target = float(log_target(candidate))
            if not np.isfinite(candidate_log_target):
                raise FloatingPointError("accepted candidate log target must be finite")
            current = candidate
            current_log_target = candidate_log_target
            accepted[index - 1] = True
            archive_points[archive_size] = candidate
            archive_logs[archive_size] = candidate_log_target
            archive_size += 1

        samples[index] = current
        log_targets[index] = current_log_target

    return SamplingResult(
        name="Moving Target Monte Carlo",
        samples=samples,
        log_targets=log_targets,
        accepted=accepted,
        target_evaluations=archive_size,
        elapsed_seconds=perf_counter() - start,
        evaluated_points=archive_points[:archive_size].copy(),
        evaluated_log_targets=archive_logs[:archive_size].copy(),
    )
