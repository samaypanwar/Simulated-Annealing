"""Tests for cooling schedule implementations."""

import pytest

from simulated_annealing.schedules.cooling import (
    AdditiveCooling,
    GeometricCooling,
    LogarithmicCooling,
    SlowDecrease,
)


SCHEDULES = [
    GeometricCooling(alpha=0.9),
    AdditiveCooling(delta=0.5),
    LogarithmicCooling(c=1.0),
    SlowDecrease(alpha=0.01),
]


class TestMonotoneDecrease:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_temperature_decreases(self, schedule):
        t = 100.0
        for step in range(1, 20):
            t_next = schedule(t, step)
            assert t_next < t, (
                f"{schedule.__class__.__name__}: t_next={t_next} >= t={t}"
            )
            t = t_next

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_temperature_stays_positive(self, schedule):
        t = 100.0
        for step in range(1, 200):
            t = schedule(t, step)
            assert t > 0, (
                f"{schedule.__class__.__name__}: temperature went non-positive at step {step}"
            )


class TestGeometricCooling:
    def test_exact_value(self):
        s = GeometricCooling(alpha=0.95)
        assert s(100.0, step=1) == pytest.approx(95.0)

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            GeometricCooling(alpha=0.0)
        with pytest.raises(ValueError):
            GeometricCooling(alpha=1.0)
        with pytest.raises(ValueError):
            GeometricCooling(alpha=1.5)

    def test_convergence(self):
        s = GeometricCooling(alpha=0.5)
        t = 1000.0
        for step in range(1, 100):
            t = s(t, step)
        # Temperature floors at _MIN_TEMP = 1e-10 to prevent division by zero
        assert t < 1e-5


class TestAdditiveCooling:
    def test_exact_value(self):
        s = AdditiveCooling(delta=5.0)
        assert s(100.0, step=1) == pytest.approx(95.0)

    def test_clamps_above_zero(self):
        """AdditiveCooling must not return 0 or negative values."""
        s = AdditiveCooling(delta=60.0)
        assert s(50.0, step=1) > 0

    def test_delta_validation(self):
        with pytest.raises(ValueError):
            AdditiveCooling(delta=0.0)
        with pytest.raises(ValueError):
            AdditiveCooling(delta=-1.0)


class TestLogarithmicCooling:
    def test_step_independent_of_initial_call_signature(self):
        s = LogarithmicCooling(c=1.0)
        assert s(100.0, step=1) > 0
        assert s(100.0, step=10) < s(100.0, step=1)

    def test_c_validation(self):
        with pytest.raises(ValueError):
            LogarithmicCooling(c=0.0)
        with pytest.raises(ValueError):
            LogarithmicCooling(c=-1.0)


class TestSlowDecrease:
    def test_exact_formula(self):
        # T / (1 + alpha * T)
        s = SlowDecrease(alpha=0.01)
        t = 100.0
        expected = t / (1 + 0.01 * t)
        assert s(t, step=1) == pytest.approx(expected)

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            SlowDecrease(alpha=0.0)
        with pytest.raises(ValueError):
            SlowDecrease(alpha=-0.5)
