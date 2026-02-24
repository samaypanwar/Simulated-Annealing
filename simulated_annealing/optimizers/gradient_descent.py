"""
Gradient Descent optimizer using finite-difference gradients.

This implementation uses no automatic differentiation — gradients are
estimated via the symmetric finite-difference formula:
    ∂f/∂x_i ≈ [f(x + ε·e_i) - f(x - ε·e_i)] / (2ε)

This makes GD applicable to any black-box objective function, at the cost
of 2d extra function evaluations per gradient step (where d = dimension).

Key limitation (educational purpose)
--------------------------------------
Gradient Descent is a local method: it follows the steepest descent from
the starting point and converges to the nearest local minimum.  On
multimodal landscapes like Ackley, the result is highly sensitive to x₀.
This is the central contrast with SA, which escapes local minima by
probabilistically accepting worse solutions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simulated_annealing.constraints.handler import penalty_augment
from simulated_annealing.core.base_optimizer import BaseOptimizer
from simulated_annealing.types import Constraint, OptimizationResult


class GradientDescent(BaseOptimizer):
    """Finite-difference Gradient Descent with optional momentum.

    Parameters
    ----------
    lr:
        Learning rate (step size).  Must be positive.
    n_iters:
        Maximum number of gradient steps.
    momentum:
        Momentum coefficient in [0, 1).  Set to 0 to disable.
    eps:
        Step size for finite-difference gradient estimation.
    seed:
        Optional random seed for sampling x0.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_iters: int = 1000,
        momentum: float = 0.0,
        eps: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")

        self.lr = lr
        self.n_iters = n_iters
        self.momentum = momentum
        self.eps = eps
        self._rng = np.random.default_rng(seed)

    def _gradient(self, obj: Callable, x: np.ndarray) -> np.ndarray:
        """Symmetric finite-difference gradient estimate."""
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = self.eps
            grad[i] = (obj(x + e) - obj(x - e)) / (2.0 * self.eps)
        return grad

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        constraints: list[Constraint] | None = None,
        x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        obj = penalty_augment(objective, constraints) if constraints else objective

        x = (
            x0.copy().astype(float)
            if x0 is not None
            else self._random_x0(bounds, self._rng)
        )
        f_x = float(obj(x))
        # finite-difference uses 2d evals per step
        n_evals = 1 + 2 * len(x)

        path = [x.copy()]
        velocity = np.zeros_like(x)

        for _ in range(self.n_iters):
            grad = self._gradient(obj, x)
            n_evals += 2 * len(x)

            velocity = self.momentum * velocity - self.lr * grad
            x_new = self._clip(x + velocity, bounds)
            f_new = float(obj(x_new))
            n_evals += 1

            x = x_new
            f_x = f_new
            path.append(x.copy())

        return OptimizationResult(
            solution=x.copy(),
            value=float(obj(x)),
            path=path,
            temperatures=[],
            acceptance_probs=[],
            n_evaluations=n_evals,
            converged=True,
        )
