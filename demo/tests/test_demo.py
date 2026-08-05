from __future__ import annotations

import unittest

import numpy as np

from demo.metropolis import metropolis_hastings, random_walk_metropolis
from demo.mtmc_reimplementation import nearest_neighbour_mtmc
from demo.proposals import GaussianIndependenceProposal
from demo.synthetic_model import GaussianLogPosterior, build_synthetic_problem


class ForwardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = build_synthetic_problem(seed=17, samples_per_ray=400)

    def test_each_row_preserves_total_ray_length(self) -> None:
        expected = np.linalg.norm(
            self.problem.receivers - self.problem.sources,
            axis=1,
        )
        np.testing.assert_allclose(
            np.sum(self.problem.ray_matrix, axis=1),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_demo_geometry_constrains_every_model_cell(self) -> None:
        self.assertEqual(
            np.linalg.matrix_rank(self.problem.ray_matrix),
            self.problem.grid.n_cells,
        )

    def test_analytical_covariance_is_positive_definite(self) -> None:
        covariance = self.problem.posterior_covariance
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-18)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))


class SamplerTests(unittest.TestCase):
    @staticmethod
    def _target(problem):
        return GaussianLogPosterior(
            problem.ray_matrix,
            problem.observed_times,
            problem.noise_std,
            problem.prior_mean,
            problem.prior_std,
        )

    def setUp(self) -> None:
        self.problem = build_synthetic_problem(seed=23, samples_per_ray=300)

    def test_metropolis_evaluates_every_recorded_step(self) -> None:
        target = self._target(self.problem)
        result = random_walk_metropolis(
            target,
            self.problem.prior_mean,
            proposal_scale=8.0e-7,
            n_steps=120,
            random=np.random.default_rng(1),
        )
        self.assertEqual(result.target_evaluations, 120)
        self.assertEqual(target.evaluations, 120)

    def test_mtmc_evaluates_only_initial_and_accepted_points(self) -> None:
        target = self._target(self.problem)
        proposal = GaussianIndependenceProposal.create(
            self.problem.prior_mean,
            self.problem.prior_std,
        )
        result = nearest_neighbour_mtmc(
            target,
            self.problem.prior_mean,
            proposal=proposal,
            metric_scale=self.problem.prior_std,
            n_steps=120,
            random=np.random.default_rng(2),
        )
        expected = 1 + int(np.count_nonzero(result.accepted))
        self.assertEqual(result.target_evaluations, expected)
        self.assertEqual(target.evaluations, expected)
        self.assertEqual(result.evaluated_points.shape[0], expected)
        self.assertLessEqual(expected, 120)

    def test_mtmc_is_reproducible_for_a_fixed_seed(self) -> None:
        proposal = GaussianIndependenceProposal.create(
            self.problem.prior_mean,
            self.problem.prior_std,
        )
        first = nearest_neighbour_mtmc(
            self._target(self.problem),
            self.problem.prior_mean,
            proposal=proposal,
            n_steps=80,
            random=np.random.default_rng(9),
        )
        second = nearest_neighbour_mtmc(
            self._target(self.problem),
            self.problem.prior_mean,
            proposal=proposal,
            n_steps=80,
            random=np.random.default_rng(9),
        )
        np.testing.assert_array_equal(first.samples, second.samples)
        np.testing.assert_array_equal(first.accepted, second.accepted)

    def test_independence_metropolis_still_evaluates_every_step(self) -> None:
        target = self._target(self.problem)
        proposal = GaussianIndependenceProposal.create(
            self.problem.prior_mean,
            self.problem.prior_std,
        )
        result = metropolis_hastings(
            target,
            self.problem.prior_mean,
            proposal=proposal,
            n_steps=75,
            random=np.random.default_rng(5),
        )
        self.assertEqual(result.target_evaluations, 75)
        self.assertEqual(target.evaluations, 75)


if __name__ == "__main__":
    unittest.main()
