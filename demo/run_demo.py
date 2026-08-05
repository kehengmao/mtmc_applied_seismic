"""Run the synthetic seismic MTMC demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .metropolis import SamplingResult, metropolis_hastings
from .mtmc import nearest_neighbour_mtmc
from .proposals import GaussianIndependenceProposal
from .synthetic_model import GaussianLogPosterior, SyntheticProblem, build_synthetic_problem


def _posterior(problem: SyntheticProblem) -> GaussianLogPosterior:
    return GaussianLogPosterior(
        problem.ray_matrix,
        problem.observed_times,
        problem.noise_std,
        problem.prior_mean,
        problem.prior_std,
    )


def _velocity(slowness: np.ndarray) -> np.ndarray:
    return np.reciprocal(np.asarray(slowness, dtype=float))


def _summary(
    result: SamplingResult,
    posterior_mean: np.ndarray,
    burn_in: int,
) -> dict[str, float | int]:
    estimated_slowness = np.mean(result.samples[burn_in:], axis=0)
    estimated_velocity = _velocity(estimated_slowness)
    reference_velocity = _velocity(posterior_mean)
    velocity_rmse = float(
        np.sqrt(np.mean(np.square(estimated_velocity - reference_velocity)))
    )
    return {
        "acceptance_rate": result.acceptance_rate,
        "true_target_evaluations": result.target_evaluations,
        "elapsed_seconds": result.elapsed_seconds,
        "posterior_velocity_rmse_m_per_s": velocity_rmse,
    }


def _save_figure(
    problem: SyntheticProblem,
    mh: SamplingResult,
    mtmc: SamplingResult,
    burn_in: int,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimates = [
        ("True synthetic model", problem.true_velocity),
        ("Analytical posterior mean", _velocity(problem.posterior_mean)),
        ("Metropolis posterior mean", _velocity(np.mean(mh.samples[burn_in:], axis=0))),
        ("MTMC posterior mean", _velocity(np.mean(mtmc.samples[burn_in:], axis=0))),
    ]
    values = np.concatenate([item[1] for item in estimates])
    lower, upper = np.percentile(values, [2, 98])

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (title, model) in zip(axes.flat[:4], estimates, strict=True):
        image = axis.imshow(
            problem.grid.reshape_cells(model),
            origin="lower",
            extent=(0, problem.grid.width, 0, problem.grid.height),
            vmin=lower,
            vmax=upper,
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    figure.colorbar(image, ax=axes.flat[:4].tolist(), label="Velocity [m/s]")

    axes[1, 1].bar(
        ["MH", "MTMC"],
        [mh.target_evaluations, mtmc.target_evaluations],
        color=["#455a64", "#d1495b"],
    )
    axes[1, 1].set_title("True target evaluations")
    axes[1, 1].set_ylabel("Count")

    axes[1, 2].plot(mh.log_targets, label="MH", color="#455a64", alpha=0.8)
    axes[1, 2].plot(mtmc.log_targets, label="MTMC", color="#d1495b", alpha=0.8)
    axes[1, 2].axvline(burn_in, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set_title("Log-target trace")
    axes[1, 2].set_xlabel("Recorded sample")
    axes[1, 2].legend()

    figure.suptitle("Synthetic MTMC seismic demonstration")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--proposal-scale", type=float, default=1.0e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/synthetic_demo"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if not 0 <= args.burn_in < args.steps:
        raise ValueError("burn-in must satisfy 0 <= burn-in < steps")

    problem = build_synthetic_problem(seed=args.seed)
    seed_sequence = np.random.SeedSequence(args.seed)
    mh_seed, mtmc_seed = seed_sequence.spawn(2)

    mh_target = _posterior(problem)
    mtmc_target = _posterior(problem)
    mh_proposal = GaussianIndependenceProposal.create(
        problem.prior_mean,
        args.proposal_scale,
    )
    mtmc_proposal = GaussianIndependenceProposal.create(
        problem.prior_mean,
        args.proposal_scale,
    )
    mh = metropolis_hastings(
        mh_target,
        problem.prior_mean,
        proposal=mh_proposal,
        n_steps=args.steps,
        random=np.random.default_rng(mh_seed),
    )
    mtmc = nearest_neighbour_mtmc(
        mtmc_target,
        problem.prior_mean,
        proposal=mtmc_proposal,
        metric_scale=problem.prior_std,
        n_steps=args.steps,
        random=np.random.default_rng(mtmc_seed),
    )

    metrics = {
        "problem": {
            "model_dimension": problem.grid.n_cells,
            "number_of_rays": int(problem.ray_matrix.shape[0]),
            "noise_std_seconds": problem.noise_std,
            "proposal": "independent Gaussian",
            "proposal_scale": args.proposal_scale,
            "steps": args.steps,
            "burn_in": args.burn_in,
        },
        "metropolis_hastings": _summary(
            mh,
            problem.posterior_mean,
            args.burn_in,
        ),
        "moving_target_monte_carlo": _summary(
            mtmc,
            problem.posterior_mean,
            args.burn_in,
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "samples.npz",
        true_velocity=problem.true_velocity,
        analytical_posterior_mean=problem.posterior_mean,
        mh_samples=mh.samples,
        mtmc_samples=mtmc.samples,
        mtmc_evaluated_points=mtmc.evaluated_points,
    )
    if not args.no_plot:
        _save_figure(
            problem,
            mh,
            mtmc,
            args.burn_in,
            args.output / "summary.png",
        )

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
