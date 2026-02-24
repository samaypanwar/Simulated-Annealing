# Simulated Annealing

An educational Python package for understanding and using Simulated Annealing (SA), with comparisons against Gradient Descent and Random Search across three benchmark loss landscapes.

## What's Inside

| Component | Description |
|---|---|
| `simulated_annealing/` | Importable package with a clean, usable API |
| `notebooks/` | 5 Jupyter notebooks forming a guided tutorial |
| `tests/` | 109 unit tests (TDD) |

## Notebooks

| Notebook | Topic |
|---|---|
| `01_introduction.ipynb` | What is SA? Metallurgical analogy, acceptance probability |
| `02_loss_landscapes.ipynb` | Sphere, Ackley, Rosenbrock — 3D plots and topology |
| `03_sa_deep_dive.ipynb` | Path tracing, cooling schedule comparison, SA vs SSA |
| `04_comparison.ipynb` | SA vs Gradient Descent vs Random Search benchmark |
| `05_constraints.ipynb` | Constrained optimization via the penalty method |

## Quick Start

```python
from simulated_annealing import SimulatedAnnealing
from simulated_annealing.landscapes import ackley
from simulated_annealing.schedules import GeometricCooling
from simulated_annealing.neighbours import GaussianStep

result = SimulatedAnnealing(
    schedule=GeometricCooling(alpha=0.97),
    neighbourhood=GaussianStep(scale=0.5),
    initial_temp=10.0,
    final_temp=1e-4,
).optimize(ackley, bounds=[(-5, 5), (-5, 5)])

print(result.solution, result.value)
```

## Package Structure

```
simulated_annealing/
  types.py              # OptimizationResult, Constraint, Protocols
  core/
    annealer.py         # Classical SA
    stochastic_annealer.py  # SA with reheating
    base_optimizer.py   # ABC
  schedules/
    cooling.py          # Geometric, Additive, Logarithmic, SlowDecrease
  landscapes/
    functions.py        # sphere, ackley, rosenbrock + LANDSCAPES registry
  neighbours/
    perturbation.py     # GaussianStep, UniformStep, AdaptiveStep
  constraints/
    handler.py          # penalty_augment, is_feasible
  optimizers/
    gradient_descent.py # Finite-difference GD
    random_search.py    # Uniform random search
  utils/
    visualization.py    # Plotting functions
    metrics.py          # benchmark_summary, convergence_rate
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Constrained Optimization

```python
from simulated_annealing.types import Constraint
import numpy as np

# Stay inside the unit disk
constraint = Constraint(
    fn=lambda x: x[0]**2 + x[1]**2 - 1.0,
    kind="inequality",
    penalty=1e4,
)

result = SimulatedAnnealing(...).optimize(
    objective, bounds=[(-5,5),(-5,5)], constraints=[constraint]
)
```
