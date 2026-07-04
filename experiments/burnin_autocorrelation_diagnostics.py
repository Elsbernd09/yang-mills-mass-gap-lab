"""
Burn-in and autocorrelation diagnostics for SU(2) lattice gauge theory.

This experiment:
1. Runs a longer Metropolis chain.
2. Discards an initial burn-in period.
3. Measures Wilson action and average plaquette.
4. Computes autocorrelation diagnostics.
5. Estimates effective sample size.

This makes the Monte Carlo methodology more professional and transparent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ymlab.diagnostics import diagnose_autocorrelation
from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis


def print_diagnostics(name: str, diagnostics) -> None:
    print(f"{name}:")
    print(f"  Integrated autocorrelation time: {diagnostics.integrated_autocorrelation_time:.6f}")
    print(f"  Effective sample size:           {diagnostics.effective_sample_size:.6f}")
    print()


def main() -> None:
    shape = (8, 8)
    beta = 2.0
    sweeps = 200
    burn_in = 50
    epsilon = 0.18
    measurement_interval = 1

    lattice = Lattice(shape=shape, cold_start=True, seed=2026)

    print("Running SU(2) burn-in/autocorrelation diagnostic experiment...")
    print(f"Shape: {shape}")
    print(f"Beta: {beta}")
    print(f"Total sweeps: {sweeps}")
    print(f"Burn-in sweeps: {burn_in}")
    print(f"Measurement interval: {measurement_interval}")
    print(f"Epsilon: {epsilon}")
    print()

    result = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=measurement_interval,
        burn_in=burn_in,
    )

    actions = np.asarray(result.actions, dtype=float)
    plaquettes = np.asarray(result.average_plaquettes, dtype=float)
    acceptances = np.asarray(result.acceptance_rates, dtype=float)

    action_diag = diagnose_autocorrelation(actions, burn_in=0, max_lag=40)
    plaquette_diag = diagnose_autocorrelation(plaquettes, burn_in=0, max_lag=40)

    print("Measurement summary after burn-in:")
    print(f"Recorded measurements:           {len(actions)}")
    print(f"Mean Wilson action:              {np.mean(actions):.8f}")
    print(f"Mean average plaquette:          {np.mean(plaquettes):.8f}")
    print(f"Mean acceptance rate:            {np.mean(acceptances):.8f}")
    print()

    print_diagnostics("Wilson action autocorrelation", action_diag)
    print_diagnostics("Average plaquette autocorrelation", plaquette_diag)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweep_axis = np.arange(1, len(actions) + 1)

    plt.figure()
    plt.plot(sweep_axis, actions)
    plt.xlabel("Measurement index after burn-in")
    plt.ylabel("Wilson action")
    plt.title("Wilson Action After Burn-In")
    plt.tight_layout()
    plt.savefig(figures_dir / "burnin_action_trace.png", dpi=200)

    plt.figure()
    plt.plot(sweep_axis, plaquettes)
    plt.xlabel("Measurement index after burn-in")
    plt.ylabel("Average plaquette")
    plt.title("Average Plaquette After Burn-In")
    plt.tight_layout()
    plt.savefig(figures_dir / "burnin_plaquette_trace.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(len(action_diag.autocorrelation)), action_diag.autocorrelation)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Wilson Action Autocorrelation")
    plt.tight_layout()
    plt.savefig(figures_dir / "action_autocorrelation.png", dpi=200)

    plt.figure()
    plt.plot(np.arange(len(plaquette_diag.autocorrelation)), plaquette_diag.autocorrelation)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Average Plaquette Autocorrelation")
    plt.tight_layout()
    plt.savefig(figures_dir / "plaquette_autocorrelation.png", dpi=200)

    print("Saved figures:")
    print("results/figures/burnin_action_trace.png")
    print("results/figures/burnin_plaquette_trace.png")
    print("results/figures/action_autocorrelation.png")
    print("results/figures/plaquette_autocorrelation.png")


if __name__ == "__main__":
    main()
