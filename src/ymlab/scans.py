"""
Parameter scans for SU(2) lattice gauge theory.

This module supports:
1. Beta scans: varying the coupling parameter beta.
2. Finite-size scans: varying lattice size.
3. Combined beta/size experiments.

These scans are important because a serious numerical Yang-Mills project should
not report only one isolated simulation. It should examine how observables
change across parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.ensemble import BootstrapResult, bootstrap_mean, run_independent_chain


@dataclass
class ScanPoint:
    """One point in a beta/size scan."""

    shape: tuple[int, ...]
    beta: float
    sweeps: int
    epsilon: float
    seeds: list[int]
    action: BootstrapResult
    plaquette: BootstrapResult
    acceptance: BootstrapResult


@dataclass
class ScanResult:
    """Container for all scan points."""

    points: list[ScanPoint]


def run_scan_point(
    shape: tuple[int, ...],
    beta: float,
    sweeps: int,
    epsilon: float,
    seeds: list[int],
    n_bootstrap: int = 1000,
) -> ScanPoint:
    """
    Run multiple independent chains for one beta/shape setting.
    """
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")

    summaries = [
        run_independent_chain(
            shape=shape,
            beta=beta,
            sweeps=sweeps,
            epsilon=epsilon,
            seed=seed,
            cold_start=True,
        )
        for seed in seeds
    ]

    actions = np.asarray([summary.final_action for summary in summaries], dtype=float)
    plaquettes = np.asarray(
        [summary.final_average_plaquette for summary in summaries],
        dtype=float,
    )
    acceptances = np.asarray(
        [summary.mean_acceptance_rate for summary in summaries],
        dtype=float,
    )

    seed_base = int(100000 * beta) + sum(shape) + seeds[0]

    return ScanPoint(
        shape=shape,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        seeds=seeds,
        action=bootstrap_mean(
            actions,
            n_bootstrap=n_bootstrap,
            seed=seed_base + 1,
        ),
        plaquette=bootstrap_mean(
            plaquettes,
            n_bootstrap=n_bootstrap,
            seed=seed_base + 2,
        ),
        acceptance=bootstrap_mean(
            acceptances,
            n_bootstrap=n_bootstrap,
            seed=seed_base + 3,
        ),
    )


def run_beta_scan(
    shape: tuple[int, ...],
    betas: list[float],
    sweeps: int,
    epsilon: float,
    seeds: list[int],
    n_bootstrap: int = 1000,
) -> ScanResult:
    """
    Run a beta scan at fixed lattice shape.
    """
    if len(betas) == 0:
        raise ValueError("At least one beta is required.")

    points = [
        run_scan_point(
            shape=shape,
            beta=beta,
            sweeps=sweeps,
            epsilon=epsilon,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        for beta in betas
    ]

    return ScanResult(points=points)


def run_finite_size_scan(
    shapes: list[tuple[int, ...]],
    beta: float,
    sweeps: int,
    epsilon: float,
    seeds: list[int],
    n_bootstrap: int = 1000,
) -> ScanResult:
    """
    Run a finite-size scan at fixed beta.
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required.")

    points = [
        run_scan_point(
            shape=shape,
            beta=beta,
            sweeps=sweeps,
            epsilon=epsilon,
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        for shape in shapes
    ]

    return ScanResult(points=points)


def run_beta_size_grid(
    shapes: list[tuple[int, ...]],
    betas: list[float],
    sweeps: int,
    epsilon: float,
    seeds: list[int],
    n_bootstrap: int = 1000,
) -> ScanResult:
    """
    Run a full grid over lattice shape and beta.
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required.")

    if len(betas) == 0:
        raise ValueError("At least one beta is required.")

    points = []

    for shape in shapes:
        for beta in betas:
            points.append(
                run_scan_point(
                    shape=shape,
                    beta=beta,
                    sweeps=sweeps,
                    epsilon=epsilon,
                    seeds=seeds,
                    n_bootstrap=n_bootstrap,
                )
            )

    return ScanResult(points=points)


def scan_result_table(result: ScanResult) -> list[dict[str, float | str]]:
    """
    Convert a scan result into a list of table rows.
    """
    rows: list[dict[str, float | str]] = []

    for point in result.points:
        rows.append(
            {
                "shape": str(point.shape),
                "volume": float(np.prod(point.shape)),
                "beta": float(point.beta),
                "action_mean": point.action.mean,
                "action_se": point.action.standard_error,
                "plaquette_mean": point.plaquette.mean,
                "plaquette_se": point.plaquette.standard_error,
                "acceptance_mean": point.acceptance.mean,
                "acceptance_se": point.acceptance.standard_error,
            }
        )

    return rows
