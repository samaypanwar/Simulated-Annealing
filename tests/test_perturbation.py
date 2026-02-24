"""Tests for neighbourhood / perturbation functions."""

import numpy as np
import pytest

from simulated_annealing.neighbours.perturbation import (
    AdaptiveStep,
    GaussianStep,
    UniformStep,
)

BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]
X0 = np.array([0.0, 0.0])


class TestOutputShape:
    @pytest.mark.parametrize("cls", [GaussianStep, UniformStep, AdaptiveStep])
    def test_output_shape_matches_input(self, cls):
        fn = cls()
        x_new = fn(X0, temperature=1.0, bounds=BOUNDS_2D)
        assert x_new.shape == X0.shape

    @pytest.mark.parametrize("cls", [GaussianStep, UniformStep, AdaptiveStep])
    def test_output_is_numpy(self, cls):
        fn = cls()
        x_new = fn(X0, temperature=1.0, bounds=BOUNDS_2D)
        assert isinstance(x_new, np.ndarray)


class TestBoundsClamping:
    @pytest.mark.parametrize("cls", [GaussianStep, UniformStep, AdaptiveStep])
    def test_result_within_bounds(self, cls):
        fn = cls()
        rng = np.random.default_rng(42)
        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        # Start at a corner to stress clamping
        x = np.array([1.0, 1.0])
        for _ in range(100):
            x_new = fn(x, temperature=10.0, bounds=bounds)
            for i, (lo, hi) in enumerate(bounds):
                assert lo <= x_new[i] <= hi, (
                    f"{cls.__name__}: x_new[{i}]={x_new[i]} outside [{lo}, {hi}]"
                )


class TestGaussianStep:
    def test_scale_controls_step_size(self):
        """Larger scale → larger average displacement."""
        rng = np.random.default_rng(0)
        small = GaussianStep(scale=0.01)
        large = GaussianStep(scale=2.0)
        bounds = [(-100.0, 100.0)] * 2

        displacements_small = []
        displacements_large = []
        for _ in range(500):
            displacements_small.append(np.linalg.norm(small(X0, 1.0, bounds) - X0))
            displacements_large.append(np.linalg.norm(large(X0, 1.0, bounds) - X0))

        assert np.mean(displacements_large) > np.mean(displacements_small)

    def test_different_outputs_on_repeated_calls(self):
        fn = GaussianStep()
        results = [fn(X0, 1.0, BOUNDS_2D) for _ in range(10)]
        # Not all identical
        assert not all(np.allclose(r, results[0]) for r in results[1:])


class TestUniformStep:
    def test_step_within_radius(self):
        """radius is an L-infinity constraint: each dimension stays within ±radius."""
        fn = UniformStep(radius=0.5)
        bounds = [(-100.0, 100.0)] * 2
        for _ in range(200):
            x_new = fn(X0, 1.0, bounds)
            assert np.max(np.abs(x_new - X0)) <= 0.5 + 1e-9


class TestAdaptiveStep:
    def test_high_temp_larger_steps(self):
        """At high temperature, steps should be larger on average than at low."""
        fn = AdaptiveStep(base_scale=0.3)
        bounds = [(-100.0, 100.0)] * 2

        high_temp_displacements = [
            np.linalg.norm(fn(X0, 100.0, bounds)) for _ in range(300)
        ]
        low_temp_displacements = [
            np.linalg.norm(fn(X0, 0.01, bounds)) for _ in range(300)
        ]

        assert np.mean(high_temp_displacements) > np.mean(low_temp_displacements)
