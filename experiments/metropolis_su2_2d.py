"""
Metropolis-Hastings SU(2) lattice gauge theory experiment.

This experiment starts from a cold SU(2) lattice and evolves the configuration
using Metropolis-Hastings updates. It records:

1. Wilson action
2. Acceptance rate
3. Average plaquette

The goal is to demonstrate that the code can generate a sequence of lattice
gauge configurations rather than merely evaluating one fixed configuration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis


def main() -> None:
    beta = 2.0
    sweeps = 50
    epsilon = 0.15
    measurement_interval = 1

    lattice = Lattice(shape=(6, 6), cold_start=True, seed=2026)

    print("Running SU(2) Metropolis-Hastings simulation...")
    print(f"Shape: {lattice.shape}")
    print(f"Beta: {beta}")
    print(f"Sweeps: {sweeps}")
    print(f"Epsilon: {epsilon}")
    print()

    result = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=measurement_interval,
    )

    for i, (action, acceptance, plaquette) in enumerate(
        zip(result.actions, result.acceptance_rates, result.average_plaquettes),
        start=1,
    ):
        print(
            f"Sweep {i:03d} | "
            f"Action: {action:10.6f} | "
            f"Acceptance: {acceptance:6.3f} | "
            f"Average plaquette: {plaquette: .6f}"
        )

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweeps_axis = list(range(1, len(result.actions) + 1))

    plt.figure()
    plt.plot(sweeps_axis, result.actions)
    plt.xlabel("Sweep")
    plt.ylabel("Wilson action")
    plt.title("SU(2) Wilson Action During Metropolis Evolution")
    plt.tight_layout()
    plt.savefig(figures_dir / "metropolis_action.png", dpi=200)

    plt.figure()
    plt.plot(sweeps_axis, result.acceptance_rates)
    plt.xlabel("Sweep")
    plt.ylabel("Acceptance rate")
    plt.title("Metropolis Acceptance Rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "metropolis_acceptance.png", dpi=200)

    plt.figure()
    plt.plot(sweeps_axis, result.average_plaquettes)
    plt.xlabel("Sweep")
    plt.ylabel("Average plaquette")
    plt.title("Average Plaquette During SU(2) Evolution")
    plt.tight_layout()
    plt.savefig(figures_dir / "metropolis_average_plaquette.png", dpi=200)

    print()
    print("Saved figures:")
    print("results/figures/metropolis_action.png")
    print("results/figures/metropolis_acceptance.png")
    print("results/figures/metropolis_average_plaquette.png")


if __name__ == "__main__":
    main()
