"""Tests for the classical SimulatedAnnealing optimizer."""

import math

import numpy as np
import pytest

from simulated_annealing.core.annealer import SimulatedAnnealing
from simulated_annealing.landscapes.functions import sphere, ackley
from simulated_annealing.neighbours.perturbation import GaussianStep
from simulated_annealing.schedules.cooling import GeometricCooling
from simulated_annealing.types import OptimizationResult

BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]


def make_sa(**kwargs) -> SimulatedAnnealing:
    defaults = dict(
        schedule=GeometricCooling(alpha=0.95),
        neighbourhood=GaussianStep(scale=0.5, seed=42),
        initial_temp=10.0,
        final_temp=0.01,
    )
    defaults.update(kwargs)
    return SimulatedAnnealing(**defaults)


class TestReturnType:
    def test_returns_optimization_result(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result, OptimizationResult)

    def test_solution_shape(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.solution.shape == (2,)

    def test_value_is_float(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result.value, float)

    def test_converged_flag(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.converged is True


class TestPathTracking:
    def test_path_nonempty(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert len(result.path) > 0

    def test_path_starts_at_x0(self):
        x0 = np.array([3.0, -3.0])
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D, x0=x0)
        np.testing.assert_array_equal(result.path[0], x0)

    def test_temperatures_match_path_length(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert len(result.temperatures) == len(result.path)

    def test_acceptance_probs_match_path_length(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert len(result.acceptance_probs) == len(result.path)

    def test_acceptance_probs_in_range(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        for p in result.acceptance_probs:
            assert 0.0 < p <= 1.0

    def test_record_every_n_reduces_path(self):
        sa_full = make_sa()
        sa_sparse = SimulatedAnnealing(
            schedule=GeometricCooling(alpha=0.95),
            neighbourhood=GaussianStep(scale=0.5, seed=42),
            initial_temp=10.0,
            final_temp=0.01,
            record_every_n=5,
        )
        r_full = sa_full.optimize(sphere, bounds=BOUNDS_2D)
        r_sparse = sa_sparse.optimize(sphere, bounds=BOUNDS_2D)
        assert len(r_sparse.path) <= len(r_full.path)


class TestOptimality:
    def test_finds_sphere_minimum(self):
        """On the convex sphere, SA should reliably find x ≈ 0."""
        sa = SimulatedAnnealing(
            schedule=GeometricCooling(alpha=0.97),
            neighbourhood=GaussianStep(scale=0.3, seed=0),
            initial_temp=5.0,
            final_temp=1e-4,
        )
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.value < 0.5, f"Expected value < 0.5, got {result.value}"

    def test_solution_within_bounds(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        for i, (lo, hi) in enumerate(BOUNDS_2D):
            assert lo <= result.solution[i] <= hi

    def test_n_evaluations_positive(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.n_evaluations > 0


class TestCustomX0:
    def test_custom_x0_used(self):
        x0 = np.array([1.0, 1.0])
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D, x0=x0)
        np.testing.assert_array_equal(result.path[0], x0)

    def test_none_x0_sampled_in_bounds(self):
        sa = make_sa()
        result = sa.optimize(sphere, bounds=BOUNDS_2D, x0=None)
        for i, (lo, hi) in enumerate(BOUNDS_2D):
            assert lo <= result.path[0][i] <= hi
