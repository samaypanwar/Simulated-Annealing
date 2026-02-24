"""
Loss landscape functions for benchmarking optimizers.

Each function accepts either a 1-D ``np.ndarray`` of shape ``(d,)`` for
point-wise evaluation, or a list of ``d`` broadcastable arrays (e.g. from
``np.meshgrid``) for vectorised evaluation used in surface plots.

All functions are defined as *minimisation* targets (return positive values
outside the global minimum).

Landscapes
----------
sphere      Convex bowl.  Global min = 0 at origin.  Easiest possible test.
ackley      Highly multimodal with a flat outer region and a deep central well.
            SA's advantage over gradient descent is most visible here.
rosenbrock  Narrow, curved (banana-shaped) valley.  Global min = 0 at (1,…,1).
            Hard for methods that rely on local gradient information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Landscape functions
# ---------------------------------------------------------------------------


def sphere(x) -> float | np.ndarray:
    """f(x) = sum(x_i^2).  Global minimum: 0 at origin."""
    return sum(xi**2 for xi in x)


def ackley(
    x, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi
) -> float | np.ndarray:
    """Ackley function (2-D by default, vectorised).

    Global minimum: 0 at (0, 0, …, 0).
    Typical bounds: [-5, 5]^d.

    The landscape has exponentially many local minima, which makes it an ideal
    test case for algorithms like SA that accept worse solutions probabilistically.
    """
    d = len(x)
    sum_sq = sum(xi**2 for xi in x)
    sum_cos = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + math.e


def rosenbrock(x, a: float = 1.0, b: float = 100.0) -> float | np.ndarray:
    """Rosenbrock (banana) function.

    f(x) = sum_{i=0}^{d-2} [ (a - x_i)^2 + b*(x_{i+1} - x_i^2)^2 ]

    Global minimum: 0 at (1, 1, …, 1).
    Typical bounds: [-2, 2]^d  (sometimes [-5, 10]^d).

    The narrow, curved valley makes gradient-based methods converge slowly,
    which highlights SA's ability to explore the landscape more freely.
    """
    total = sum(
        (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2 for i in range(len(x) - 1)
    )
    return total


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class LandscapeInfo:
    """Metadata for a landscape function.

    Attributes
    ----------
    fn:
        The callable landscape function.
    bounds:
        Canonical search bounds as ``(low, high)`` used in all examples.
        A single tuple is broadcast to all dimensions.
    known_minimum:
        The global minimiser as a list (for 2-D; generalise as needed).
    known_value:
        The objective value at the global minimiser.
    description:
        One-line human-readable summary.
    """

    fn: Callable
    bounds: list[tuple[float, float]]
    known_minimum: list[float]
    known_value: float
    description: str


LANDSCAPES: dict[str, LandscapeInfo] = {
    "sphere": LandscapeInfo(
        fn=sphere,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        known_minimum=[0.0, 0.0],
        known_value=0.0,
        description="Convex bowl. Trivial global minimum at the origin.",
    ),
    "ackley": LandscapeInfo(
        fn=ackley,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        known_minimum=[0.0, 0.0],
        known_value=0.0,
        description="Highly multimodal with many local minima. SA's showcase.",
    ),
    "rosenbrock": LandscapeInfo(
        fn=rosenbrock,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        known_minimum=[1.0, 1.0],
        known_value=0.0,
        description="Narrow curved valley. Hard for gradient-based methods.",
    ),
}
