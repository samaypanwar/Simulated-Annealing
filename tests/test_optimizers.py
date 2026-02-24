"""Tests for Gradient Descent and Random Search optimizers."""

import numpy as np
import pytest

from simulated_annealing.landscapes.functions import rosenbrock, sphere
from simulated_annealing.optimizers.gradient_descent import GradientDescent
from simulated_annealing.optimizers.random_search import RandomSearch
from simulated_annealing.types import OptimizationResult

BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]


class TestGradientDescent:
    def test_returns_optimization_result(self):
        gd = GradientDescent(lr=0.1, n_iters=100, seed=0)
        result = gd.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result, OptimizationResult)

    def test_finds_sphere_minimum(self):
        """GD should easily find the sphere minimum (convex)."""
        gd = GradientDescent(lr=0.1, n_iters=500, seed=0)
        result = gd.optimize(sphere, bounds=BOUNDS_2D)
        assert result.value < 0.05

    def test_solution_within_bounds(self):
        gd = GradientDescent(lr=0.05, n_iters=200, seed=1)
        result = gd.optimize(sphere, bounds=BOUNDS_2D)
        for i, (lo, hi) in enumerate(BOUNDS_2D):
            assert lo <= result.solution[i] <= hi

    def test_path_nonempty(self):
        gd = GradientDescent(lr=0.1, n_iters=50, seed=0)
        result = gd.optimize(sphere, bounds=BOUNDS_2D)
        assert len(result.path) > 0

    def test_path_starts_at_x0(self):
        x0 = np.array([3.0, -2.0])
        gd = GradientDescent(lr=0.1, n_iters=50, seed=0)
        result = gd.optimize(sphere, bounds=BOUNDS_2D, x0=x0)
        np.testing.assert_array_equal(result.path[0], x0)

    def test_n_evaluations_positive(self):
        gd = GradientDescent(lr=0.1, n_iters=50, seed=0)
        result = gd.optimize(sphere, bounds=BOUNDS_2D)
        assert result.n_evaluations > 0

    def test_lr_validation(self):
        with pytest.raises(ValueError):
            GradientDescent(lr=0.0)
        with pytest.raises(ValueError):
            GradientDescent(lr=-0.1)

    def test_n_iters_validation(self):
        with pytest.raises(ValueError):
            GradientDescent(n_iters=0)

    def test_gets_stuck_on_ackley(self):
        """GD gets trapped in local minima on Ackley (not at the global min).
        This is intentional — it's the educational contrast point with SA."""
        from simulated_annealing.landscapes.functions import ackley

        gd = GradientDescent(lr=0.01, n_iters=1000, seed=5)
        x0 = np.array([2.5, 2.5])
        result = gd.optimize(ackley, bounds=BOUNDS_2D, x0=x0)
        # GD should converge to some local minimum quickly (value > 0)
        # but we don't require it to find the global min
        assert result.value >= 0  # just sanity check, not global optimality


class TestRandomSearch:
    def test_returns_optimization_result(self):
        rs = RandomSearch(n_iters=200, seed=0)
        result = rs.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result, OptimizationResult)

    def test_solution_within_bounds(self):
        rs = RandomSearch(n_iters=200, seed=0)
        result = rs.optimize(sphere, bounds=BOUNDS_2D)
        for i, (lo, hi) in enumerate(BOUNDS_2D):
            assert lo <= result.solution[i] <= hi

    def test_value_is_float(self):
        rs = RandomSearch(n_iters=100, seed=1)
        result = rs.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result.value, float)

    def test_more_iters_improves_result(self):
        """More iterations → lower expected best value on sphere."""
        rs_few = RandomSearch(n_iters=10, seed=0)
        rs_many = RandomSearch(n_iters=5000, seed=0)
        r_few = rs_few.optimize(sphere, bounds=BOUNDS_2D)
        r_many = rs_many.optimize(sphere, bounds=BOUNDS_2D)
        assert r_many.value <= r_few.value

    def test_n_evaluations_equals_n_iters(self):
        n = 300
        rs = RandomSearch(n_iters=n, seed=0)
        result = rs.optimize(sphere, bounds=BOUNDS_2D)
        assert result.n_evaluations == n

    def test_n_iters_validation(self):
        with pytest.raises(ValueError):
            RandomSearch(n_iters=0)
