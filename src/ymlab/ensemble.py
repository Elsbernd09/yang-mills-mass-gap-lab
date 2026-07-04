"""
Ensemble statistics for SU(2) lattice gauge theory.

Real lattice gauge theory studies do not rely on one Markov chain. They run
ensembles of independent or approximately independent configurations, measure
observables, and estimate uncertainty.

This module implements:
1. Bootstrap mean/error estimation.
2. Independent Metropolis chain execution.
3. Ensemble summaries for average plaquette and Wilson action.

These tools make the numerical side of the project more credible and
scientifically honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis
from ymlab.plaquette import average_plaquette
from ymlab.wilson_action import wilson_action


@dataclass
class BootstrapResult:
    """Bootstrap estimate for a scalar statistic."""

    mean: float
    standard_error: float
    lower_ci: float
    upper_ci: float
    bootstrap_samples: np.ndarray


@dataclass
class ChainSummary:
    """Summary of one independent Markov chain."""

    seed: int
    final_action: float
    final_average_plaquette: float
    mean_acceptance_rate: float


@dataclass
class EnsembleResult:
    """Summary of an ensemble of independent chains."""

    chain_summaries: list[ChainSummary]
    action_bootstrap: BootstrapResult
    plaquette_bootstrap: BootstrapResult
    acceptance_bootstrap: BootstrapResult


def bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """
    Estimate the mean and uncertainty of a one-dimensional sample by bootstrap.

    Parameters
    ----------
    values:
        One-dimensional array of measured values.
    n_bootstrap:
        Number of bootstrap resamples.
    confidence_level:
        Confidence interval level, such as 0.95.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
        Mean, standard error, confidence interval, and bootstrap samples.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("values cannot be empty.")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(values)

    bootstrap_samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_samples[i] = float(np.mean(sample))

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_samples, alpha / 2.0))
    upper = float(np.quantile(bootstrap_samples, 1.0 - alpha / 2.0))

    return BootstrapResult(
        mean=float(np.mean(values)),
        standard_error=float(np.std(bootstrap_samples, ddof=1)),
        lower_ci=lower,
        upper_ci=upper,
        bootstrap_samples=bootstrap_samples,
    )


def run_independent_chain(
    shape: tuple[int, ...],
    beta: float,
    sweeps: int,
    epsilon: float,
    seed: int,
    cold_start: bool = True,
) -> ChainSummary:
    """
    Run one independent SU(2) Markov chain and summarize final observables.
    """
    lattice = Lattice(shape=shape, cold_start=cold_start, seed=seed)

    result = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=1,
    )

    return ChainSummary(
        seed=seed,
        final_action=float(wilson_action(lattice, beta)),
        final_average_plaquette=float(average_plaquette(lattice)),
        mean_acceptance_rate=float(np.mean(result.acceptance_rates)),
    )


def run_ensemble(
    shape: tuple[int, ...],
    beta: float,
    sweeps: int,
    epsilon: float,
    seeds: list[int],
    n_bootstrap: int = 1000,
    cold_start: bool = True,
) -> EnsembleResult:
    """
    Run several independent Markov chains and estimate ensemble uncertainty.
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
            cold_start=cold_start,
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

    return EnsembleResult(
        chain_summaries=summaries,
        action_bootstrap=bootstrap_mean(
            actions,
            n_bootstrap=n_bootstrap,
            seed=seeds[0] + 10000,
        ),
        plaquette_bootstrap=bootstrap_mean(
            plaquettes,
            n_bootstrap=n_bootstrap,
            seed=seeds[0] + 20000,
        ),
        acceptance_bootstrap=bootstrap_mean(
            acceptances,
            n_bootstrap=n_bootstrap,
            seed=seeds[0] + 30000,
        ),
    )


def summarize_bootstrap(name: str, result: BootstrapResult) -> str:
    """
    Return a formatted string for a bootstrap result.
    """
    return (
        f"{name}: mean={result.mean:.8f}, "
        f"SE={result.standard_error:.8f}, "
        f"95% CI=[{result.lower_ci:.8f}, {result.upper_ci:.8f}]"
    )
