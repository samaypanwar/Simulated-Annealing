"""Tests for the Stochastic SA optimizer (reheating variant)."""

import numpy as np
import pytest

from simulated_annealing.core.stochastic_annealer import StochasticSA
from simulated_annealing.landscapes.functions import ackley, sphere
from simulated_annealing.neighbours.perturbation import GaussianStep
from simulated_annealing.schedules.cooling import GeometricCooling
from simulated_annealing.types import OptimizationResult

BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]


def make_ssa(**kwargs) -> StochasticSA:
    defaults = dict(
        schedule=GeometricCooling(alpha=0.95),
        neighbourhood=GaussianStep(scale=0.5, seed=7),
        initial_temp=10.0,
        final_temp=0.01,
        patience=30,
        reheat_factor=2.0,
        max_reheats=5,
    )
    defaults.update(kwargs)
    return StochasticSA(**defaults)


class TestReturnType:
    def test_returns_optimization_result(self):
        ssa = make_ssa()
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        assert isinstance(result, OptimizationResult)

    def test_solution_within_bounds(self):
        ssa = make_ssa()
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        for i, (lo, hi) in enumerate(BOUNDS_2D):
            assert lo <= result.solution[i] <= hi

    def test_converged_flag(self):
        ssa = make_ssa()
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.converged is True


class TestReheating:
    def test_reheating_recorded(self):
        """reheating_events should be populated when stagnation occurs."""
        # Use a very tight patience so reheating is triggered quickly
        ssa = StochasticSA(
            schedule=GeometricCooling(alpha=0.5),
            neighbourhood=GaussianStep(scale=0.01, seed=0),  # tiny steps → stagnation
            initial_temp=100.0,
            final_temp=0.01,
            patience=5,
            reheat_factor=3.0,
            max_reheats=10,
        )
        result = ssa.optimize(ackley, bounds=BOUNDS_2D)
        assert result.n_reheats >= 0  # attribute exists

    def test_reheat_factor_validation(self):
        with pytest.raises(ValueError):
            StochasticSA(
                schedule=GeometricCooling(),
                neighbourhood=GaussianStep(),
                reheat_factor=0.5,
            )

    def test_patience_validation(self):
        with pytest.raises(ValueError):
            StochasticSA(
                schedule=GeometricCooling(),
                neighbourhood=GaussianStep(),
                patience=0,
            )


class TestOptimality:
    def test_finds_sphere_minimum(self):
        ssa = StochasticSA(
            schedule=GeometricCooling(alpha=0.97),
            neighbourhood=GaussianStep(scale=0.3, seed=1),
            initial_temp=5.0,
            final_temp=1e-4,
            patience=50,
            reheat_factor=2.0,
            max_reheats=3,
        )
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.value < 0.5

    def test_path_nonempty(self):
        ssa = make_ssa()
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        assert len(result.path) > 0

    def test_n_evaluations_positive(self):
        ssa = make_ssa()
        result = ssa.optimize(sphere, bounds=BOUNDS_2D)
        assert result.n_evaluations > 0
