"""
Performance benchmarking utilities for SU(2) lattice gauge theory.

This module measures how simulation cost scales with lattice size and dimension.
It is not about proving Yang-Mills theory; it is about making the computational
framework more transparent and professionally measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.wilson_action import number_of_plaquettes


@dataclass
class PerformanceResult:
    """Performance result for one lattice benchmark."""

    shape: tuple[int, ...]
    dimension: int
    sites: int
    links: int
    plaquettes: int
    beta: float
    epsilon: float
    sweeps: int
    total_runtime_seconds: float
    average_sweep_seconds: float
    link_updates_per_second: float
    plaquettes_per_second: float
    mean_acceptance_rate: float


def benchmark_lattice(
    shape: tuple[int, ...],
    beta: float,
    epsilon: float,
    sweeps: int,
    seed: int = 2026,
) -> PerformanceResult:
    """
    Benchmark Metropolis sweep performance on one lattice shape.
    """
    if len(shape) < 2:
        raise ValueError("shape must have dimension at least 2.")

    if sweeps <= 0:
        raise ValueError("sweeps must be positive.")

    lattice = Lattice(shape=shape, cold_start=True, seed=seed)

    sites = lattice.number_of_sites()
    links = lattice.number_of_links()
    plaquettes = number_of_plaquettes(lattice)

    acceptance_rates = []

    start = time.perf_counter()

    for _ in range(sweeps):
        acceptance_rates.append(
            metropolis_sweep(
                lattice=lattice,
                beta=beta,
                epsilon=epsilon,
            )
        )

    total_runtime = time.perf_counter() - start
    average_sweep = total_runtime / sweeps

    total_link_updates = links * sweeps
    total_plaquettes_touched = plaquettes * sweeps

    return PerformanceResult(
        shape=shape,
        dimension=len(shape),
        sites=sites,
        links=links,
        plaquettes=plaquettes,
        beta=beta,
        epsilon=epsilon,
        sweeps=sweeps,
        total_runtime_seconds=float(total_runtime),
        average_sweep_seconds=float(average_sweep),
        link_updates_per_second=float(total_link_updates / total_runtime),
        plaquettes_per_second=float(total_plaquettes_touched / total_runtime),
        mean_acceptance_rate=float(np.mean(acceptance_rates)),
    )


def benchmark_many(
    shapes: list[tuple[int, ...]],
    beta: float,
    epsilon: float,
    sweeps: int,
    seed: int = 2026,
) -> list[PerformanceResult]:
    """
    Benchmark multiple lattice shapes.
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required.")

    results = []

    for i, shape in enumerate(shapes):
        results.append(
            benchmark_lattice(
                shape=shape,
                beta=beta,
                epsilon=epsilon,
                sweeps=sweeps,
                seed=seed + i,
            )
        )

    return results


def performance_results_as_rows(
    results: list[PerformanceResult],
) -> list[dict[str, float | int | str]]:
    """
    Convert performance results to table rows.
    """
    rows: list[dict[str, float | int | str]] = []

    for result in results:
        rows.append(
            {
                "shape": str(result.shape),
                "dimension": result.dimension,
                "sites": result.sites,
                "links": result.links,
                "plaquettes": result.plaquettes,
                "beta": result.beta,
                "epsilon": result.epsilon,
                "sweeps": result.sweeps,
                "total_runtime_seconds": result.total_runtime_seconds,
                "average_sweep_seconds": result.average_sweep_seconds,
                "link_updates_per_second": result.link_updates_per_second,
                "plaquettes_per_second": result.plaquettes_per_second,
                "mean_acceptance_rate": result.mean_acceptance_rate,
            }
        )

    return rows
