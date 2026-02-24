"""
simulated_annealing
===================
Educational and practical package for Simulated Annealing optimisation.

Quick-start
-----------
>>> from simulated_annealing import SimulatedAnnealing, StochasticSA
>>> from simulated_annealing.landscapes import ackley
>>> from simulated_annealing.schedules import GeometricCooling
>>> from simulated_annealing.neighbours import GaussianStep
>>>
>>> result = SimulatedAnnealing(
...     schedule=GeometricCooling(alpha=0.95),
...     neighbourhood=GaussianStep(scale=0.3),
...     initial_temp=10.0,
...     final_temp=1e-4,
... ).optimize(ackley, bounds=[(-5, 5), (-5, 5)])
>>> print(result.solution, result.value)
"""

from simulated_annealing.types import (
    Constraint,
    CoolingSchedule,
    NeighbourhoodFn,
    OptimizationResult,
)

# Deferred imports (populated once submodules are implemented)
try:
    from simulated_annealing.core.annealer import SimulatedAnnealing
except ImportError:
    pass

try:
    from simulated_annealing.core.stochastic_annealer import StochasticSA
except ImportError:
    pass

__all__ = [
    "Constraint",
    "CoolingSchedule",
    "NeighbourhoodFn",
    "OptimizationResult",
    "SimulatedAnnealing",
    "StochasticSA",
]
