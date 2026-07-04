"""
Dimension-general diagnostics for SU(2) lattice gauge theory.

The Clay Yang-Mills problem concerns four-dimensional quantum Yang-Mills theory.
This project begins with small finite lattices, but the code is designed to
support arbitrary-dimensional periodic hypercubic lattices.

This module provides tools for comparing simulations across 2D, 3D, and 4D
lattices.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import time

import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis
from ymlab.plaquette import average_plaquette
from ymlab.wilson_action import wilson_action, number_of_plaquettes


@dataclass
class DimensionSummary:
    """Summary of one dimension/lattice-shape experiment."""

    shape: tuple[int, ...]
    dimension: int
    beta: float
    sweeps: int
    epsilon: float
    sites: int
    links: int
    plaquettes: int
    final_action: float
    action_density_per_site: float
    action_density_per_plaquette: float
    final_average_plaquette: float
    mean_acceptance_rate: float
    runtime_seconds: float


def theoretical_plaquettes_per_site(dimension: int) -> int:
    """
    Return the number of coordinate plaquette planes per lattice site.

    In d dimensions, each site has C(d, 2) positively oriented plaquette planes.
    """
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")

    return comb(dimension, 2)


def run_dimension_summary(
    shape: tuple[int, ...],
    beta: float,
    sweeps: int,
    epsilon: float,
    seed: int,
    burn_in: int = 0,
) -> DimensionSummary:
    """
    Run one SU(2) simulation and summarize dimension-dependent diagnostics.
    """
    if len(shape) < 2:
        raise ValueError("shape must have dimension at least 2.")

    start = time.perf_counter()

    lattice = Lattice(shape=shape, cold_start=True, seed=seed)

    result = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=1,
        burn_in=burn_in,
    )

    runtime = time.perf_counter() - start

    final_action = wilson_action(lattice, beta)
    sites = lattice.number_of_sites()
    links = lattice.number_of_links()
    plaquettes = number_of_plaquettes(lattice)

    return DimensionSummary(
        shape=shape,
        dimension=len(shape),
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        sites=sites,
        links=links,
        plaquettes=plaquettes,
        final_action=float(final_action),
        action_density_per_site=float(final_action / sites),
        action_density_per_plaquette=float(final_action / plaquettes),
        final_average_plaquette=float(average_plaquette(lattice)),
        mean_acceptance_rate=float(np.mean(result.acceptance_rates)),
        runtime_seconds=float(runtime),
    )


def run_dimension_comparison(
    shapes: list[tuple[int, ...]],
    beta: float,
    sweeps: int,
    epsilon: float,
    seed: int,
    burn_in: int = 0,
) -> list[DimensionSummary]:
    """
    Run dimension comparison across multiple lattice shapes.
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required.")

    return [
        run_dimension_summary(
            shape=shape,
            beta=beta,
            sweeps=sweeps,
            epsilon=epsilon,
            seed=seed + i,
            burn_in=burn_in,
        )
        for i, shape in enumerate(shapes)
    ]


def dimension_summaries_as_rows(
    summaries: list[DimensionSummary],
) -> list[dict[str, float | int | str]]:
    """
    Convert dimension summaries into table rows.
    """
    rows: list[dict[str, float | int | str]] = []

    for summary in summaries:
        rows.append(
            {
                "shape": str(summary.shape),
                "dimension": summary.dimension,
                "beta": summary.beta,
                "sweeps": summary.sweeps,
                "sites": summary.sites,
                "links": summary.links,
                "plaquettes": summary.plaquettes,
                "final_action": summary.final_action,
                "action_density_per_site": summary.action_density_per_site,
                "action_density_per_plaquette": summary.action_density_per_plaquette,
                "final_average_plaquette": summary.final_average_plaquette,
                "mean_acceptance_rate": summary.mean_acceptance_rate,
                "runtime_seconds": summary.runtime_seconds,
            }
        )

    return rows
