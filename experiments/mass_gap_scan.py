"""
Exploratory mass-gap-style scan for SU(2) lattice gauge theory.

This experiment:
1. Evolves a finite SU(2) lattice using Metropolis-Hastings.
2. Measures a plaquette-based time-slice observable.
3. Computes its connected autocorrelation.
4. Estimates effective masses from correlation decay.
5. Fits an exponential decay model when possible.

This is not a proof of the Yang-Mills mass gap. It is a finite-lattice
numerical diagnostic for mass-gap-like behavior.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ymlab.correlations import (
    plaquette_time_series,
    normalized_connected_autocorrelation,
)
from ymlab.lattice import Lattice
from ymlab.mass_gap import effective_mass, fit_exponential_mass, exponential_decay
from ymlab.monte_carlo import run_metropolis


def main() -> None:
    beta = 2.0
    sweeps = 80
    epsilon = 0.15

    lattice = Lattice(shape=(8, 8), cold_start=True, seed=2026)

    print("Running SU(2) lattice simulation for mass-gap-style diagnostics...")
    print(f"Shape: {lattice.shape}")
    print(f"Beta: {beta}")
    print(f"Sweeps: {sweeps}")
    print(f"Epsilon: {epsilon}")
    print()

    run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=10,
    )

    print("Computing plaquette time-slice observable...")
    series = plaquette_time_series(
        lattice=lattice,
        time_direction=0,
        mu=0,
        nu=1,
    )

    print("Time-slice observable:")
    for i, value in enumerate(series):
        print(f"t={i:02d}: {value:.8f}")

    correlation = normalized_connected_autocorrelation(series)

    print()
    print("Normalized connected autocorrelation:")
    for i, value in enumerate(correlation):
        print(f"lag={i:02d}: {value:.8f}")

    mass_result = effective_mass(correlation)

    print()
    print("Effective mass estimates:")
    if len(mass_result.effective_masses) == 0:
        print("No positive consecutive correlation values available.")
    else:
        for lag, mass in zip(mass_result.lags, mass_result.effective_masses):
            print(f"lag={lag:02d}: m_eff={mass:.8f}")

    fit_result = None
    try:
        fit_result = fit_exponential_mass(
            correlation=correlation,
            min_lag=0,
            max_lag=min(4, len(correlation) - 1),
        )

        print()
        print("Exponential fit:")
        print(f"Amplitude: {fit_result.amplitude:.8f}")
        print(f"Estimated mass: {fit_result.mass:.8f}")

    except ValueError as error:
        print()
        print(f"Exponential fit skipped: {error}")

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    lags = np.arange(len(correlation))

    plt.figure()
    plt.plot(lags, correlation, marker="o")
    plt.xlabel("Lag")
    plt.ylabel("Normalized connected autocorrelation")
    plt.title("Plaquette Correlation Function")
    plt.tight_layout()
    plt.savefig(figures_dir / "plaquette_correlation.png", dpi=200)

    if len(mass_result.effective_masses) > 0:
        plt.figure()
        plt.plot(mass_result.lags, mass_result.effective_masses, marker="o")
        plt.xlabel("Lag")
        plt.ylabel("Effective mass")
        plt.title("Effective Mass Estimate")
        plt.tight_layout()
        plt.savefig(figures_dir / "effective_mass.png", dpi=200)

    if fit_result is not None:
        positive_mask = correlation > 0
        fit_lags = lags[positive_mask]

        if len(fit_lags) > 0:
            plt.figure()
            plt.scatter(lags, correlation, label="Correlation")
            plt.plot(
                fit_lags,
                exponential_decay(
                    fit_lags,
                    fit_result.amplitude,
                    fit_result.mass,
                ),
                label="Exponential fit",
            )
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.title("Exponential Mass-Gap-Style Fit")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "mass_gap_exponential_fit.png", dpi=200)

    print()
    print("Saved figures:")
    print("results/figures/plaquette_correlation.png")
    print("results/figures/effective_mass.png")
    print("results/figures/mass_gap_exponential_fit.png")


if __name__ == "__main__":
    main()
