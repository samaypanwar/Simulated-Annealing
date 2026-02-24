"""
Stochastic Simulated Annealing (SSA) with reheating.

Difference from classical SA
-----------------------------
Classical SA cools monotonically until T < T_final.  On highly multimodal
landscapes (e.g. Ackley) it can get trapped in a local minimum once the
temperature is too low to accept uphill moves.

SSA adds a **reheating** mechanism: if the best-known objective value has not
improved for *patience* consecutive accepted steps, the temperature is spiked
back up by *reheat_factor*.  This gives the algorithm another chance to escape
the local basin.  After *max_reheats* reheat events the algorithm terminates.

The reheating idea is sometimes called "simulated annealing with restarts" or
"adaptive SA" in the literature.

The returned :class:`OptimizationResult` is augmented with a custom attribute
``n_reheats`` (number of times reheating was triggered).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass
class _SSAResult(OptimizationResult):
    """OptimizationResult extended with the number of reheating events."""

    n_reheats: int = 0


class StochasticSA(BaseOptimizer):
    """Simulated Annealing with reheating for escaping local minima.

    Parameters
    ----------
    schedule:
        Cooling schedule applied between reheat events.
    neighbourhood:
        Perturbation function.
    initial_temp:
        Starting temperature at the beginning of each cooling phase.
    final_temp:
        Temperature at which a cooling phase ends (triggering reheat or stop).
    patience:
        Number of consecutive accepted steps without improvement before a
        reheat is triggered.
    reheat_factor:
        Multiplier applied to *final_temp* on reheat.  Must be > 1.
        E.g. reheat_factor=2 doubles the current temperature.
    max_reheats:
        Maximum number of reheating events before the run terminates.
    record_every_n:
        Record path entry every *n* accepted steps.
    seed:
        Optional random seed.
    """

    def __init__(
        self,
        schedule: CoolingSchedule,
        neighbourhood: NeighbourhoodFn,
        initial_temp: float = 10.0,
        final_temp: float = 1e-4,
        patience: int = 50,
        reheat_factor: float = 2.0,
        max_reheats: int = 5,
        record_every_n: int = 1,
        seed: int | None = None,
    ) -> None:
        if reheat_factor <= 1.0:
            raise ValueError(f"reheat_factor must be > 1, got {reheat_factor}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.schedule = schedule
        self.neighbourhood = neighbourhood
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.patience = patience
        self.reheat_factor = reheat_factor
        self.max_reheats = max_reheats
        self.record_every_n = record_every_n
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        constraints: list[Constraint] | None = None,
        x0: np.ndarray | None = None,
    ) -> _SSAResult:
        """Run SSA with reheating and return the result."""
        obj = penalty_augment(objective, constraints) if constraints else objective

        x = x0.copy() if x0 is not None else self._random_x0(bounds, self._rng)
        f_x = float(obj(x))
        n_evals = 1

        # Track global best (may differ from current x during exploration)
        best_x = x.copy()
        best_f = f_x

        T = self.initial_temp
        step = 0
        steps_without_improvement = 0
        n_reheats = 0

        path: list[np.ndarray] = [x.copy()]
        temperatures: list[float] = [T]
        acceptance_probs: list[float] = [1.0]

        while True:
            step += 1
            x_new = self.neighbourhood(x, T, bounds)
            f_new = float(obj(x_new))
            n_evals += 2

            delta = f_new - f_x

            if delta <= 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)

            if self._rng.random() < prob:
                x = x_new
                f_x = f_new

                if f_x < best_f:
                    best_f = f_x
                    best_x = x.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if step % self.record_every_n == 0:
                    path.append(x.copy())
                    temperatures.append(T)
                    acceptance_probs.append(prob)

            T = self.schedule(T, step)

            # --- Reheating trigger ---
            if T <= self.final_temp or steps_without_improvement >= self.patience:
                if n_reheats >= self.max_reheats or T <= self.final_temp:
                    break
                # Reheat: spike temperature and reset stagnation counter
                T = min(T * self.reheat_factor, self.initial_temp)
                steps_without_improvement = 0
                n_reheats += 1
                step = 0  # reset step counter so schedule restarts from step 1

        return _SSAResult(
            solution=best_x,
            value=best_f,
            path=path,
            temperatures=temperatures,
            acceptance_probs=acceptance_probs,
            n_evaluations=n_evals,
            converged=True,
            n_reheats=n_reheats,
        )
