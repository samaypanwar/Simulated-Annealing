"""
Performance metrics for comparing optimizers.

Functions
---------
convergence_rate        Rate at which the objective improves per step.
evals_to_threshold      Number of function evaluations to reach a target value.
benchmark_summary       Run multiple optimizers on a landscape and return a dict.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simulated_annealing.types import OptimizationResult


def evals_to_threshold(
    value_history: list[float],
    threshold: float,
) -> int | None:
    """Return the index of the first step where f ≤ threshold.

    Parameters
    ----------
    value_history:
        Sequence of objective values at each step.
    threshold:
        Target value.

    Returns
    -------
    int or None
        Step index (0-based) where the threshold was first reached, or
        ``None`` if it was never reached.
    """
    for i, v in enumerate(value_history):
        if v <= threshold:
            return i
    return None


def convergence_rate(value_history: list[float]) -> float:
    """Estimate the average improvement per step.

    Computed as (f_0 - f_T) / T where T = len(value_history) - 1.
    Returns 0.0 for constant or single-element sequences.
    """
    if len(value_history) < 2:
        return 0.0
    return (value_history[0] - value_history[-1]) / (len(value_history) - 1)


def benchmark_summary(
    optimizers: dict[str, object],
    objective: Callable,
    bounds: list[tuple[float, float]],
    n_runs: int = 5,
    constraints=None,
) -> dict[str, dict]:
    """Run each optimizer *n_runs* times and aggregate statistics.

    Parameters
    ----------
    optimizers:
        Dict of ``{name: optimizer_instance}``.  Each optimizer must
        implement ``optimize(objective, bounds, constraints) -> OptimizationResult``.
    objective:
        Objective function to minimise.
    bounds:
        Search space bounds.
    n_runs:
        Number of independent runs per optimizer (to average out randomness).
    constraints:
        Optional constraint list passed through to each optimizer.

    Returns
    -------
    dict
        ``{optimizer_name: {"mean_value": ..., "std_value": ...,
           "mean_evals": ..., "best_value": ...}}``
    """
    summary = {}
    for name, opt in optimizers.items():
        values = []
        evals = []
        for _ in range(n_runs):
            result: OptimizationResult = opt.optimize(objective, bounds, constraints)
            values.append(result.value)
            evals.append(result.n_evaluations)
        summary[name] = {
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
            "best_value": float(np.min(values)),
            "mean_evals": float(np.mean(evals)),
        }
    return summary
