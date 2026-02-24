"""
Core data contracts for the simulated_annealing package.

All public types used across the package are defined here to avoid circular imports
and to serve as a single source of truth for the API surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """The outcome of running any optimizer in this package.

    Attributes
    ----------
    solution:
        The best found point in the search space.
    value:
        Objective function value at *solution*.
    path:
        Sequence of accepted positions visited during the run (includes x0).
        May be shorter than the total number of iterations when
        ``record_every_n > 1`` is used.
    temperatures:
        Temperature value recorded at each accepted step (SA-based optimizers).
        Empty for optimizers that do not use a temperature schedule.
    acceptance_probs:
        Acceptance probability recorded at each step (SA-based optimizers).
        1.0 indicates the move was an improvement; values in (0, 1) indicate
        a probabilistic uphill acceptance.
    n_evaluations:
        Total number of objective function calls made during the run.
    converged:
        True if the optimizer's internal stopping criterion was satisfied
        (e.g. temperature reached ``T_final``).
    """

    solution: np.ndarray
    value: float
    path: list[np.ndarray] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    acceptance_probs: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """Represents a single optimization constraint.

    A constraint is expressed as a function *fn* such that:
    - ``fn(x) <= 0``  means *x* is **feasible** (for inequality constraints)
    - ``fn(x) == 0``  means *x* is **feasible** (for equality constraints)

    The penalty method converts the constrained problem into an unconstrained
    one by adding ``penalty * max(0, fn(x))^2`` (inequality) or
    ``penalty * fn(x)^2`` (equality) to the objective.

    Attributes
    ----------
    fn:
        The constraint function.  Returns a scalar; negative / zero → feasible.
    kind:
        ``"inequality"`` (≤ 0) or ``"equality"`` (= 0).
    penalty:
        Coefficient controlling how harshly infeasibility is penalised.
        Larger values force the solver to stay feasible but can create
        steep cliffs in the augmented landscape.
    """

    fn: Callable[[np.ndarray], float]
    kind: Literal["inequality", "equality"] = "inequality"
    penalty: float = 1e4


# ---------------------------------------------------------------------------
# Protocols  (structural typing for callables)
# ---------------------------------------------------------------------------


@runtime_checkable
class CoolingSchedule(Protocol):
    """Protocol for temperature cooling schedules.

    A cooling schedule is any callable that maps the current temperature *t*
    and iteration *step* to the next temperature.  Implementations must
    guarantee the returned value is strictly positive.

    Examples
    --------
    >>> schedule = GeometricCooling(alpha=0.95)
    >>> next_t = schedule(t=100.0, step=1)
    """

    def __call__(self, t: float, step: int) -> float: ...


@runtime_checkable
class NeighbourhoodFn(Protocol):
    """Protocol for neighbourhood / perturbation functions.

    Given the current position *x*, the current temperature *temperature*,
    and the search *bounds*, returns a new candidate position.  The
    implementation is responsible for clamping or reflecting the result
    back inside the bounds.

    Examples
    --------
    >>> perturb = GaussianStep(scale=0.1)
    >>> x_new = perturb(x=np.array([1.0, 2.0]), temperature=5.0, bounds=[(-5, 5), (-5, 5)])
    """

    def __call__(
        self,
        x: np.ndarray,
        temperature: float,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray: ...
