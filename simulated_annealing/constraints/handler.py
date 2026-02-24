"""
Constraint handling via the penalty method.

The penalty method converts a constrained optimization problem:
    min  f(x)
    s.t. g_i(x) ≤ 0   (inequality)
         h_j(x) = 0   (equality)

into an unconstrained one:
    min  f(x) + sum_i  penalty_i * max(0, g_i(x))^2
              + sum_j  penalty_j * h_j(x)^2

A larger penalty coefficient forces the solver to respect the constraint more
strictly, but can create steep cliffs that make the landscape harder to navigate.
This trade-off is a known limitation of the penalty method and is discussed in
the constraints notebook.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simulated_annealing.types import Constraint


def penalty_augment(
    objective: Callable[[np.ndarray], float],
    constraints: list[Constraint] | None,
) -> Callable[[np.ndarray], float]:
    """Wrap *objective* with a penalty term for each constraint.

    If *constraints* is ``None`` or empty the original function is returned
    unchanged.

    Parameters
    ----------
    objective:
        The original objective function.
    constraints:
        List of :class:`~simulated_annealing.types.Constraint` objects.

    Returns
    -------
    Callable
        An augmented function ``f_aug(x) = f(x) + penalty_sum(x)``.
    """
    if not constraints:
        return objective

    def augmented(x: np.ndarray):
        value = objective(x)
        for c in constraints:
            v = c.fn(x)
            if c.kind == "inequality":
                value = value + c.penalty * np.maximum(0.0, v) ** 2
            else:  # equality
                value = value + c.penalty * v**2
        return value

    return augmented


def is_feasible(
    x: np.ndarray, constraints: list[Constraint], tol: float = 1e-6
) -> bool:
    """Return True if *x* satisfies all constraints within tolerance."""
    for c in constraints:
        v = float(c.fn(x))
        if c.kind == "inequality" and v > tol:
            return False
        if c.kind == "equality" and abs(v) > tol:
            return False
    return True
