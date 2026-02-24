"""
Visualization utilities for loss landscapes and optimizer paths.

All plotting functions return the ``matplotlib.figure.Figure`` object so
callers can either display it in a notebook (``plt.show()``) or save it
(``fig.savefig(...)``).  Functions never call ``plt.show()`` directly —
that is left to the caller.

Functions
---------
plot_landscape_3d        Surface plot of a 2-D objective function.
plot_landscape_contour   Filled contour plot (top-down view).
plot_path_on_contour     Contour plot with optimizer path overlaid.
plot_convergence         Objective value vs. iteration for one or more runs.
plot_temperature_schedule  Temperature decay curves for SA schedules.
plot_acceptance_prob     Acceptance probability over the course of an SA run.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from simulated_annealing.types import OptimizationResult


def _make_grid(
    fn: Callable,
    bounds: list[tuple[float, float]],
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a meshgrid and evaluate *fn* on it."""
    xs = np.linspace(bounds[0][0], bounds[0][1], n_points)
    ys = np.linspace(bounds[1][0], bounds[1][1], n_points)
    X, Y = np.meshgrid(xs, ys)
    Z = fn([X, Y])
    return X, Y, Z


def plot_landscape_3d(
    fn: Callable,
    bounds: list[tuple[float, float]],
    title: str = "Objective Landscape",
    n_points: int = 80,
    cmap: str = "viridis",
    known_minimum: list[float] | None = None,
) -> Figure:
    """3-D surface plot of a 2-D objective function.

    Parameters
    ----------
    fn:
        2-D objective function accepting a list ``[X, Y]`` of meshgrid arrays.
    bounds:
        ``[(x_lo, x_hi), (y_lo, y_hi)]``.
    title:
        Plot title.
    n_points:
        Grid resolution along each axis.
    cmap:
        Matplotlib colormap name.
    known_minimum:
        If provided, marks the global minimum with a red dot.
    """
    X, Y, Z = _make_grid(fn, bounds, n_points)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x)")
    ax.set_title(title)

    if known_minimum is not None:
        z_min = float(fn(np.array(known_minimum)))
        ax.scatter(
            *known_minimum, z_min, color="red", s=80, zorder=5, label="Global min"
        )
        ax.legend()

    fig.tight_layout()
    return fig


def plot_landscape_contour(
    fn: Callable,
    bounds: list[tuple[float, float]],
    title: str = "Objective Contour",
    n_points: int = 200,
    levels: int = 30,
    cmap: str = "viridis",
    known_minimum: list[float] | None = None,
) -> Figure:
    """Filled contour plot (bird's-eye view) of a 2-D objective."""
    X, Y, Z = _make_grid(fn, bounds, n_points)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label="f(x)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)

    if known_minimum is not None:
        ax.scatter(
            *known_minimum, color="red", s=100, marker="*", zorder=5, label="Global min"
        )
        ax.legend()

    fig.tight_layout()
    return fig


def plot_path_on_contour(
    fn: Callable,
    bounds: list[tuple[float, float]],
    results: dict[str, OptimizationResult],
    title: str = "Optimizer Paths",
    n_points: int = 200,
    levels: int = 30,
    cmap: str = "gray",
    known_minimum: list[float] | None = None,
    max_path_points: int = 500,
) -> Figure:
    """Contour map with the path taken by one or more optimizers overlaid.

    Parameters
    ----------
    fn:
        Objective function (2-D, accepts meshgrid lists).
    bounds:
        Search space bounds.
    results:
        Dict mapping optimizer name → :class:`OptimizationResult`.  Each
        result's ``path`` attribute is plotted as a trajectory.
    title:
        Plot title.
    max_path_points:
        Downsample paths longer than this value for readability.
    """
    X, Y, Z = _make_grid(fn, bounds, n_points)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.7)
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.2, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label="f(x)")

    colors = plt.cm.tab10.colors
    for idx, (name, result) in enumerate(results.items()):
        path = result.path
        if len(path) > max_path_points:
            step = len(path) // max_path_points
            path = path[::step]

        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            color = colors[idx % len(colors)]
            ax.plot(xs, ys, "-", color=color, alpha=0.7, linewidth=1.2, label=name)
            ax.scatter(xs[0], ys[0], color=color, s=80, marker="o", zorder=5)  # start
            ax.scatter(
                result.solution[0],
                result.solution[1],
                color=color,
                s=120,
                marker="X",
                zorder=6,
            )  # end

    if known_minimum is not None:
        ax.scatter(
            *known_minimum, color="red", s=150, marker="*", zorder=7, label="Global min"
        )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_convergence(
    results: dict[str, OptimizationResult],
    title: str = "Convergence",
    log_scale: bool = False,
) -> Figure:
    """Plot objective value at each accepted step for one or more optimizer runs.

    Parameters
    ----------
    results:
        Dict mapping optimizer name → :class:`OptimizationResult`.
        Uses ``value_history`` (the accepted objective value at each recorded
        step) when available.  Falls back to plotting the final value as a
        flat reference line for results that pre-date this field.
    title:
        Plot title.
    log_scale:
        If True, use a log scale on the y-axis.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        if result.value_history:
            n_pts = len(result.value_history)
            ax.plot(
                range(n_pts),
                result.value_history,
                "-",
                color=color,
                alpha=0.8,
                linewidth=1.2,
                label=f"{name}  (best f={result.value:.4f})",
            )
        else:
            n_pts = len(result.path)
            if n_pts > 0:
                ax.plot(
                    range(n_pts),
                    [result.value] * n_pts,
                    "--",
                    color=color,
                    alpha=0.4,
                    linewidth=0.8,
                )
            ax.scatter(
                n_pts - 1 if n_pts else 0,
                result.value,
                color=color,
                s=80,
                zorder=5,
                label=f"{name}  (f={result.value:.4f})",
            )
        ax.axhline(result.value, color=color, linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Accepted steps recorded")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_value_history(
    value_histories: dict[str, list[float]],
    title: str = "Objective Value History",
    log_scale: bool = False,
) -> Figure:
    """Plot pre-computed objective value sequences (one per optimizer).

    Parameters
    ----------
    value_histories:
        Dict of ``{optimizer_name: [f(x_0), f(x_1), ...]}``.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for idx, (name, values) in enumerate(value_histories.items()):
        color = colors[idx % len(colors)]
        ax.plot(values, color=color, linewidth=1.5, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_temperature_schedule(
    schedules: dict[str, object],
    initial_temp: float = 10.0,
    n_steps: int = 300,
    title: str = "Cooling Schedules",
) -> Figure:
    """Plot temperature decay curves for multiple cooling schedules.

    Parameters
    ----------
    schedules:
        Dict mapping schedule name → schedule callable.
    initial_temp:
        Starting temperature.
    n_steps:
        Number of steps to simulate.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for idx, (name, schedule) in enumerate(schedules.items()):
        color = colors[idx % len(colors)]
        temps = [initial_temp]
        t = initial_temp
        for step in range(1, n_steps + 1):
            t = schedule(t, step)
            temps.append(t)
        ax.plot(temps, color=color, linewidth=2, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature T")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_acceptance_probability(
    result: OptimizationResult,
    title: str = "Acceptance Probability over Time",
    optimizer_name: str = "SA",
) -> Figure:
    """Plot the acceptance probability at each recorded step of an SA run.

    A probability of 1.0 means the move was an improvement; values < 1
    indicate probabilistic uphill acceptance.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    steps = list(range(len(result.acceptance_probs)))

    axes[0].plot(
        steps, result.acceptance_probs, alpha=0.6, linewidth=0.8, color="steelblue"
    )
    axes[0].set_ylabel("Acceptance prob")
    axes[0].set_title(f"{title} — {optimizer_name}")
    axes[0].set_ylim(0, 1.05)

    if result.temperatures:
        axes[1].plot(steps, result.temperatures, color="tomato", linewidth=1.5)
        axes[1].set_ylabel("Temperature T")
    else:
        axes[1].set_visible(False)

    axes[-1].set_xlabel("Accepted step")
    fig.tight_layout()
    return fig
