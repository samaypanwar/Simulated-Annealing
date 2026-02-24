"""Tests for constraint handling (penalty augmentation)."""

import numpy as np
import pytest

from simulated_annealing.constraints.handler import is_feasible, penalty_augment
from simulated_annealing.core.annealer import SimulatedAnnealing
from simulated_annealing.landscapes.functions import sphere
from simulated_annealing.neighbours.perturbation import GaussianStep
from simulated_annealing.schedules.cooling import GeometricCooling
from simulated_annealing.types import Constraint


class TestPenaltyAugment:
    def test_none_constraints_returns_original(self):
        obj = lambda x: x[0] ** 2
        aug = penalty_augment(obj, None)
        assert aug is obj

    def test_empty_list_returns_original(self):
        obj = lambda x: x[0] ** 2
        aug = penalty_augment(obj, [])
        assert aug is obj

    def test_feasible_point_no_penalty(self):
        """Inside the feasible region the augmented value equals the original."""
        obj = lambda x: x[0] ** 2
        c = Constraint(fn=lambda x: x[0] - 2.0, kind="inequality", penalty=1e4)
        aug = penalty_augment(obj, [c])
        x = np.array([1.0])  # feasible: x[0] - 2 = -1 ≤ 0
        assert aug(x) == pytest.approx(obj(x))

    def test_infeasible_point_penalised(self):
        obj = lambda x: 0.0  # zero objective
        c = Constraint(fn=lambda x: x[0] - 2.0, kind="inequality", penalty=1e4)
        aug = penalty_augment(obj, [c])
        x = np.array([3.0])  # infeasible: x[0] - 2 = 1 > 0
        # penalty = 1e4 * (3 - 2)^2 = 1e4
        assert aug(x) == pytest.approx(1e4)

    def test_equality_constraint_at_zero(self):
        obj = lambda x: 0.0
        c = Constraint(
            fn=lambda x: x[0] ** 2 + x[1] ** 2 - 1.0, kind="equality", penalty=1e3
        )
        aug = penalty_augment(obj, [c])
        x_on = np.array([1.0, 0.0])  # exactly on unit circle → penalty = 0
        x_off = np.array([2.0, 0.0])  # off circle → penalty > 0
        assert aug(x_on) == pytest.approx(0.0, abs=1e-10)
        assert aug(x_off) > 0

    def test_multiple_constraints(self):
        obj = lambda x: 0.0
        c1 = Constraint(fn=lambda x: x[0] - 1.0, kind="inequality", penalty=100.0)
        c2 = Constraint(fn=lambda x: x[1] - 1.0, kind="inequality", penalty=200.0)
        aug = penalty_augment(obj, [c1, c2])
        # Both violated: x = [2, 3]
        x = np.array([2.0, 3.0])
        expected = 100.0 * (2.0 - 1.0) ** 2 + 200.0 * (3.0 - 1.0) ** 2
        assert aug(x) == pytest.approx(expected)


class TestIsFeasible:
    def test_feasible_point(self):
        c = Constraint(fn=lambda x: x[0] - 2.0)
        assert is_feasible(np.array([1.0]), [c]) is True

    def test_infeasible_point(self):
        c = Constraint(fn=lambda x: x[0] - 2.0)
        assert is_feasible(np.array([3.0]), [c]) is False

    def test_equality_feasible(self):
        c = Constraint(fn=lambda x: x[0] - 1.0, kind="equality")
        assert is_feasible(np.array([1.0]), [c]) is True

    def test_equality_infeasible(self):
        c = Constraint(fn=lambda x: x[0] - 1.0, kind="equality")
        assert is_feasible(np.array([1.1]), [c], tol=1e-6) is False


class TestConstrainedOptimization:
    def test_sa_respects_box_constraint(self):
        """SA with a hard upper-bound constraint should stay close to x ≤ 0.5."""
        # Objective: sphere (min at origin)
        # Constraint: x[0] ≤ 0.5 and x[1] ≤ 0.5  (inequality: x - 0.5 ≤ 0)
        constraints = [
            Constraint(fn=lambda x: x[0] - 0.5, kind="inequality", penalty=1e5),
            Constraint(fn=lambda x: x[1] - 0.5, kind="inequality", penalty=1e5),
        ]
        sa = SimulatedAnnealing(
            schedule=GeometricCooling(alpha=0.97),
            neighbourhood=GaussianStep(scale=0.2, seed=42),
            initial_temp=5.0,
            final_temp=1e-4,
        )
        result = sa.optimize(
            sphere, bounds=[(-5.0, 5.0), (-5.0, 5.0)], constraints=constraints
        )
        # The penalty pushes the solution towards (0, 0) but bounded by x ≤ 0.5
        assert result.solution[0] <= 0.6  # allow small numerical slack
        assert result.solution[1] <= 0.6
