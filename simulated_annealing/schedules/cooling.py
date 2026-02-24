"""
Cooling schedules for Simulated Annealing.

A cooling schedule controls how the temperature *T* evolves over time.
The temperature governs the probability of accepting a worse solution:
    P(accept) = exp(-ΔE / T)
As T → 0 the algorithm becomes purely greedy (hill-climbing).

All schedule objects are callable and satisfy the :class:`CoolingSchedule`
Protocol: ``schedule(t: float, step: int) -> float``.

Schedules
---------
GeometricCooling    T_{k+1} = α · T_k                     (most common)
AdditiveCooling     T_{k+1} = T_k − δ  (clamped > ε)
LogarithmicCooling  T_{k+1} = c / log(1 + step)           (theoretically optimal)
SlowDecrease        T_{k+1} = T_k / (1 + α · T_k)
"""

from __future__ import annotations

import math


_MIN_TEMP = 1e-10  # floor to prevent division by zero


class GeometricCooling:
    """Multiply temperature by a constant factor each step.

    T_{k+1} = alpha * T_k

    This is the most widely used schedule.  With alpha = 0.95, the
    temperature halves roughly every 14 steps.

    Parameters
    ----------
    alpha:
        Multiplicative decay factor. Must be in the open interval (0, 1).
    """

    def __init__(self, alpha: float = 0.95) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def __call__(self, t: float, step: int) -> float:
        return max(t * self.alpha, _MIN_TEMP)

    def __repr__(self) -> str:
        return f"GeometricCooling(alpha={self.alpha})"


class AdditiveCooling:
    """Subtract a fixed amount from the temperature each step.

    T_{k+1} = max(T_k − delta, ε)

    Simple but requires careful tuning: if delta is too large relative to
    the initial temperature the run may end prematurely.

    Parameters
    ----------
    delta:
        Amount to subtract each step. Must be positive.
    """

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = delta

    def __call__(self, t: float, step: int) -> float:
        return max(t - self.delta, _MIN_TEMP)

    def __repr__(self) -> str:
        return f"AdditiveCooling(delta={self.delta})"


class LogarithmicCooling:
    """Logarithmic cooling schedule (Geman & Geman, 1984).

    T_k = c / log(1 + k)

    This schedule is theoretically guaranteed to find the global optimum
    as k → ∞ (given the right c), but converges very slowly in practice.
    It is included here for educational comparison.

    Parameters
    ----------
    c:
        Scale constant. Must be positive.
    """

    def __init__(self, c: float = 1.0) -> None:
        if c <= 0.0:
            raise ValueError(f"c must be positive, got {c}")
        self.c = c

    def __call__(self, t: float, step: int) -> float:
        return max(self.c / math.log(1 + step), _MIN_TEMP)

    def __repr__(self) -> str:
        return f"LogarithmicCooling(c={self.c})"


class SlowDecrease:
    """Slow-decrease (Aarts & Korst) cooling schedule.

    T_{k+1} = T_k / (1 + alpha * T_k)

    Produces a slower cooling curve than geometric, lying between
    geometric and logarithmic in terms of convergence speed.

    Parameters
    ----------
    alpha:
        Positive control parameter. Larger values cool faster.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def __call__(self, t: float, step: int) -> float:
        return max(t / (1.0 + self.alpha * t), _MIN_TEMP)

    def __repr__(self) -> str:
        return f"SlowDecrease(alpha={self.alpha})"
