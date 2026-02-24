"""Tests for loss landscape functions."""

import numpy as np
import pytest

from simulated_annealing.landscapes.functions import (
    LANDSCAPES,
    LandscapeInfo,
    ackley,
    rosenbrock,
    sphere,
)


class TestSphere:
    def test_global_minimum_is_zero(self):
        assert sphere(np.zeros(2)) == pytest.approx(0.0)

    def test_global_minimum_ndim(self):
        for d in [1, 2, 5, 10]:
            assert sphere(np.zeros(d)) == pytest.approx(0.0)

    def test_positive_elsewhere(self):
        assert sphere(np.array([1.0, 2.0])) > 0

    def test_symmetric(self):
        x = np.array([1.0, -2.0, 3.0])
        assert sphere(x) == pytest.approx(sphere(-x))

    def test_vectorised_evaluation(self):
        """sphere should work on meshgrid arrays for plotting."""
        xs = np.linspace(-3, 3, 10)
        X, Y = np.meshgrid(xs, xs)
        Z = sphere([X, Y])
        assert Z.shape == X.shape

    def test_gradient_at_minimum(self):
        """Finite-difference gradient should be ~0 at the optimum."""
        eps = 1e-5
        x = np.zeros(2)
        grad = np.array(
            [(sphere(x + eps * e) - sphere(x - eps * e)) / (2 * eps) for e in np.eye(2)]
        )
        np.testing.assert_allclose(grad, 0.0, atol=1e-4)


class TestAckley:
    def test_global_minimum_near_zero(self):
        result = ackley(np.zeros(2))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_multimodal_not_flat(self):
        """Points near the optimum but not at it should have positive values."""
        x = np.array([0.5, 0.5])
        assert ackley(x) > 0

    def test_vectorised(self):
        xs = np.linspace(-5, 5, 8)
        X, Y = np.meshgrid(xs, xs)
        Z = ackley([X, Y])
        assert Z.shape == X.shape

    def test_symmetric(self):
        x = np.array([1.5, -2.3])
        assert ackley(x) == pytest.approx(ackley(-x), rel=1e-6)

    def test_bounded_output(self):
        """Ackley output is bounded between 0 and ~22.7."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform(-5, 5, size=2)
            val = ackley(x)
            assert 0 <= val <= 23.0


class TestRosenbrock:
    def test_global_minimum_at_ones(self):
        assert rosenbrock(np.ones(2)) == pytest.approx(0.0)

    def test_global_minimum_ndim(self):
        for d in [2, 3, 5]:
            assert rosenbrock(np.ones(d)) == pytest.approx(0.0)

    def test_positive_elsewhere(self):
        assert rosenbrock(np.zeros(2)) > 0

    def test_vectorised(self):
        xs = np.linspace(-2, 2, 8)
        X, Y = np.meshgrid(xs, xs)
        Z = rosenbrock([X, Y])
        assert Z.shape == X.shape

    def test_gradient_at_minimum(self):
        eps = 1e-5
        x = np.ones(2)
        grad = np.array(
            [
                (rosenbrock(x + eps * e) - rosenbrock(x - eps * e)) / (2 * eps)
                for e in np.eye(2)
            ]
        )
        np.testing.assert_allclose(grad, 0.0, atol=1e-3)


class TestLandscapeRegistry:
    def test_all_landscapes_present(self):
        assert "sphere" in LANDSCAPES
        assert "ackley" in LANDSCAPES
        assert "rosenbrock" in LANDSCAPES

    def test_landscape_info_fields(self):
        for name, info in LANDSCAPES.items():
            assert isinstance(info, LandscapeInfo)
            assert info.fn is not None
            assert len(info.bounds) >= 1
            assert info.known_minimum is not None
            assert info.known_value is not None

    def test_known_minima_correct(self):
        for name, info in LANDSCAPES.items():
            result = info.fn(np.array(info.known_minimum))
            assert result == pytest.approx(info.known_value, abs=1e-6), (
                f"{name}: expected f({info.known_minimum}) = {info.known_value}, got {result}"
            )
