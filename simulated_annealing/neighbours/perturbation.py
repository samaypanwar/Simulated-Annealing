"""
Neighbourhood / perturbation functions.

A perturbation function proposes a new candidate solution by modifying the
current position *x*.  All implementations satisfy the :class:`NeighbourhoodFn`
Protocol and clamp their output to remain within *bounds*.

Perturbation functions
----------------------
GaussianStep    Add Gaussian noise (fixed scale).
UniformStep     Sample uniformly within an L∞ ball.
AdaptiveStep    Gaussian noise whose scale shrinks with temperature — gives
                finer search as the algorithm cools.
"""

from __future__ import annotations

import numpy as np


def _clamp(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return np.clip(x, lo, hi)


class GaussianStep:
    """Perturb each dimension independently with Gaussian noise.

    Parameters
    ----------
    scale:
        Standard deviation of the Gaussian perturbation.  The same scale is
        applied to all dimensions.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(self, scale: float = 0.3, seed: int | None = None) -> None:
        self.scale = scale
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        x: np.ndarray,
        temperature: float,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        noise = self._rng.normal(loc=0.0, scale=self.scale, size=x.shape)
        return _clamp(x + noise, bounds)

    def __repr__(self) -> str:
        return f"GaussianStep(scale={self.scale})"


class UniformStep:
    """Perturb by sampling uniformly inside a hypercube of given radius.

    The perturbation is drawn from Uniform(-radius, +radius) per dimension,
    which produces a flat distribution of step sizes.

    Parameters
    ----------
    radius:
        Half-width of the uniform perturbation interval.
    seed:
        Optional random seed.
    """

    def __init__(self, radius: float = 0.5, seed: int | None = None) -> None:
        self.radius = radius
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        x: np.ndarray,
        temperature: float,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        noise = self._rng.uniform(-self.radius, self.radius, size=x.shape)
        return _clamp(x + noise, bounds)

    def __repr__(self) -> str:
        return f"UniformStep(radius={self.radius})"


class AdaptiveStep:
    """Temperature-adaptive Gaussian perturbation.

    The effective scale is proportional to the current temperature:
        effective_scale = base_scale * temperature

    This means the algorithm explores broadly when the temperature is high
    and switches to fine local search as it cools — mimicking the physical
    annealing process more faithfully.

    Parameters
    ----------
    base_scale:
        Proportionality constant.  A value around 0.1–0.5 works well for
        landscapes with bounds in [-5, 5].
    seed:
        Optional random seed.
    """

    def __init__(self, base_scale: float = 0.3, seed: int | None = None) -> None:
        self.base_scale = base_scale
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        x: np.ndarray,
        temperature: float,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        scale = self.base_scale * temperature
        noise = self._rng.normal(loc=0.0, scale=max(scale, 1e-8), size=x.shape)
        return _clamp(x + noise, bounds)

    def __repr__(self) -> str:
        return f"AdaptiveStep(base_scale={self.base_scale})"
