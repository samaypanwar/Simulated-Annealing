"""Abstract base class for all optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from simulated_annealing.types import Constraint, OptimizationResult


class BaseOptimizer(ABC):
    """Common interface shared by every optimizer in this package.

    Subclasses must implement :meth:`optimize`.  They are free to add
    constructor parameters as needed.
    """

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        constraints: list[Constraint] | None = None,
        x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Run the optimizer and return the result.

        Parameters
        ----------
        objective:
            The function to **minimise**.  Must accept a 1-D ``np.ndarray``
            and return a scalar float.
        bounds:
            List of ``(low, high)`` pairs, one per dimension.  Defines the
            valid search space.
        constraints:
            Optional list of :class:`~simulated_annealing.types.Constraint`
            objects.  When provided, the optimizer should apply a penalty
            augmentation (or its own feasibility mechanism).
        x0:
            Optional starting point.  If ``None`` the optimizer samples a
            random point inside *bounds*.

        Returns
        -------
        OptimizationResult
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _random_x0(
        bounds: list[tuple[float, float]], rng: np.random.Generator
    ) -> np.ndarray:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return lo + rng.random(len(bounds)) * (hi - lo)

    @staticmethod
    def _clip(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return np.clip(x, lo, hi)
