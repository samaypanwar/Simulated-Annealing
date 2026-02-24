"""
Random Search optimizer.

Random Search is the simplest possible baseline: sample uniformly at random
from the search space and keep track of the best sample found.  Despite its
simplicity, it provides a meaningful lower bound — any optimizer that cannot
outperform pure random sampling on a given problem is useless.

On low-dimensional problems Random Search performs surprisingly well.  As
dimensionality grows its sample efficiency degrades exponentially (curse of
dimensionality), which is where structured methods like SA shine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simulated_annealing.constraints.handler import penalty_augment
from simulated_annealing.core.base_optimizer import BaseOptimizer
from simulated_annealing.types import Constraint, OptimizationResult


class RandomSearch(BaseOptimizer):
    """Uniform random sampling over the search space.

    Parameters
    ----------
    n_iters:
        Number of candidate solutions to evaluate.
    seed:
        Optional random seed.
    """

    def __init__(self, n_iters: int = 1000, seed: int | None = None) -> None:
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        self.n_iters = n_iters
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        constraints: list[Constraint] | None = None,
        x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        obj = penalty_augment(objective, constraints) if constraints else objective

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = len(bounds)

        best_x = lo + self._rng.random(d) * (hi - lo)
        best_f = float(obj(best_x))
        path = [best_x.copy()]

        for _ in range(self.n_iters - 1):
            x = lo + self._rng.random(d) * (hi - lo)
            f = float(obj(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            path.append(x.copy())

        return OptimizationResult(
            solution=best_x,
            value=best_f,
            path=path,
            temperatures=[],
            acceptance_probs=[],
            n_evaluations=self.n_iters,
            converged=True,
        )
