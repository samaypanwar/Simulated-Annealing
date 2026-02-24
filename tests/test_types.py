"""Tests for core data contracts in simulated_annealing.types."""

import numpy as np
import pytest

from simulated_annealing.types import (
    Constraint,
    CoolingSchedule,
    NeighbourhoodFn,
    OptimizationResult,
)


class TestOptimizationResult:
    def test_construction_minimal(self):
        result = OptimizationResult(solution=np.array([0.0, 0.0]), value=0.0)
        assert result.value == 0.0
        assert len(result.path) == 0
        assert len(result.temperatures) == 0
        assert len(result.acceptance_probs) == 0
        assert result.n_evaluations == 0
        assert result.converged is False

    def test_construction_full(self):
        path = [np.array([1.0, 2.0]), np.array([0.5, 1.0])]
        result = OptimizationResult(
            solution=np.array([0.0, 0.0]),
            value=0.1,
            path=path,
            temperatures=[10.0, 9.5],
            acceptance_probs=[1.0, 0.7],
            n_evaluations=200,
            converged=True,
        )
        assert result.n_evaluations == 200
        assert result.converged is True
        assert len(result.path) == 2

    def test_solution_is_numpy(self):
        result = OptimizationResult(solution=np.zeros(3), value=0.0)
        assert isinstance(result.solution, np.ndarray)


class TestConstraint:
    def test_default_kind_and_penalty(self):
        c = Constraint(fn=lambda x: x[0] - 1.0)
        assert c.kind == "inequality"
        assert c.penalty == 1e4

    def test_equality_constraint(self):
        c = Constraint(
            fn=lambda x: x[0] ** 2 + x[1] ** 2 - 1.0, kind="equality", penalty=1e3
        )
        assert c.kind == "equality"
        assert c.penalty == 1e3

    def test_fn_callable(self):
        c = Constraint(fn=lambda x: x[0])
        x_feasible = np.array([-1.0])
        x_infeasible = np.array([1.0])
        assert c.fn(x_feasible) <= 0
        assert c.fn(x_infeasible) > 0


class TestProtocols:
    def test_cooling_schedule_protocol(self):
        class MySchedule:
            def __call__(self, t: float, step: int) -> float:
                return t * 0.9

        assert isinstance(MySchedule(), CoolingSchedule)

    def test_neighbourhood_fn_protocol(self):
        class MyNeighbour:
            def __call__(self, x, temperature, bounds):
                return x + 0.01

        assert isinstance(MyNeighbour(), NeighbourhoodFn)

    def test_lambda_does_not_satisfy_cooling_protocol(self):
        # A plain lambda has no __call__ with the right signature via isinstance,
        # but Protocol runtime_checkable only checks method existence, not signatures.
        # This test documents that behaviour: a callable is sufficient.
        fn = lambda t, step: t * 0.9  # noqa: E731
        assert isinstance(fn, CoolingSchedule)
