"""
Classical Simulated Annealing optimizer.

Algorithm
---------
1. Start at position x₀ (random if not supplied) with temperature T = T_initial.
2. Propose a neighbour x' = neighbourhood(x, T, bounds).
3. Compute ΔE = f(x') - f(x).
4. Accept x' with probability:
       P = 1            if ΔE ≤ 0   (improvement)
       P = exp(-ΔE / T) if ΔE > 0   (probabilistic uphill move)
5. Update temperature: T = schedule(T, step).
6. Repeat until T < T_final.

The acceptance probability for uphill moves is the heart of the algorithm:
at high temperature nearly all moves are accepted (broad exploration), while
at low temperature only improvements are accepted (exploitation).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from simulated_annealing.constraints.handler import penalty_augment
from simulated_annealing.core.base_optimizer import BaseOptimizer
from simulated_annealing.types import (
    Constraint,
    CoolingSchedule,
    NeighbourhoodFn,
    OptimizationResult,
)


class SimulatedAnnealing(BaseOptimizer):
    """Classical Simulated Annealing.

    Parameters
    ----------
    schedule:
        Cooling schedule instance (see :mod:`simulated_annealing.schedules`).
    neighbourhood:
        Perturbation function (see :mod:`simulated_annealing.neighbours`).
    initial_temp:
        Starting temperature.  Should be high enough that most moves are
        accepted at the beginning (acceptance rate ≈ 0.8–0.9 is a rule of
        thumb).
    final_temp:
        Stopping temperature.  The run terminates when T falls below this.
    record_every_n:
        Record the current position to ``path`` every *n* accepted steps.
        Set to 1 (default) to record every step; higher values reduce memory
        usage for long runs.
    seed:
        Optional random seed for reproducibility of the acceptance draws.
    """

    def __init__(
        self,
        schedule: CoolingSchedule,
        neighbourhood: NeighbourhoodFn,
        initial_temp: float = 10.0,
        final_temp: float = 1e-4,
        record_every_n: int = 1,
        seed: int | None = None,
    ) -> None:
        self.schedule = schedule
        self.neighbourhood = neighbourhood
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.record_every_n = record_every_n
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        constraints: list[Constraint] | None = None,
        x0: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Run classical SA and return the result."""
        obj = penalty_augment(objective, constraints) if constraints else objective

        x = x0.copy() if x0 is not None else self._random_x0(bounds, self._rng)
        f_x = float(obj(x))
        n_evals = 1

        best_x = x.copy()
        best_f = f_x

        T = self.initial_temp
        step = 0

        path: list[np.ndarray] = [x.copy()]
        temperatures: list[float] = [T]
        acceptance_probs: list[float] = [1.0]
        value_history: list[float] = [f_x]

        while T > self.final_temp:
            step += 1
            x_new = self.neighbourhood(x, T, bounds)
            f_new = float(obj(x_new))
            n_evals += 2  # one for x_new; current x already evaluated

            delta = f_new - f_x

            if delta <= 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)

            if self._rng.random() < prob:
                x = x_new
                f_x = f_new

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

                if step % self.record_every_n == 0:
                    path.append(x.copy())
                    temperatures.append(T)
                    acceptance_probs.append(prob)
                    value_history.append(f_x)

            T = self.schedule(T, step)

        return OptimizationResult(
            solution=best_x,
            value=best_f,
            path=path,
            temperatures=temperatures,
            acceptance_probs=acceptance_probs,
            value_history=value_history,
            n_evaluations=n_evals,
            converged=True,
        )
