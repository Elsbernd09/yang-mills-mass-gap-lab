"""
Wilson loop experiment for SU(2) lattice gauge theory.

This experiment:
1. Generates an SU(2) lattice configuration using Metropolis-Hastings.
2. Measures rectangular Wilson loops of different sizes.
3. Saves a simple area-vs-log-loop plot.

In lattice gauge theory, Wilson loop decay is connected to confinement.
A rough area-law pattern has the form

    W(R, T) ~ exp(-sigma R T)

where sigma is interpreted as a string-tension-like quantity.

This script is exploratory and finite-dimensional. It does not prove confinement
or the Yang-Mills mass gap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis
from ymlab.observables import wilson_loop_table


def main() -> None:
    beta = 2.0
    sweeps = 40
    epsilon = 0.15

    lattice = Lattice(shape=(6, 6), cold_start=True, seed=2026)

    print("Thermalizing SU(2) lattice with Metropolis-Hastings...")
    run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=10,
    )

    print("Measuring Wilson loops...")
    table = wilson_loop_table(
        lattice=lattice,
        mu=0,
        nu=1,
        max_width=3,
        max_height=3,
    )

    print()
    print("Width Height Area Perimeter WilsonLoop")
    print("-" * 50)

    for row in table:
        print(
            f"{int(row['width']):5d} "
            f"{int(row['height']):6d} "
            f"{int(row['area']):4d} "
            f"{int(row['perimeter']):9d} "
            f"{row['value']: .8f}"
        )

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    positive_rows = [row for row in table if row["value"] > 0]

    if positive_rows:
        areas = np.array([row["area"] for row in positive_rows])
        log_values = np.log(np.array([row["value"] for row in positive_rows]))

        plt.figure()
        plt.scatter(areas, log_values)
        plt.xlabel("Loop area")
        plt.ylabel("log Wilson loop")
        plt.title("Exploratory Wilson Loop Area Decay")
        plt.tight_layout()
        plt.savefig(figures_dir / "wilson_loop_area_decay.png", dpi=200)

        print()
        print("Saved figure:")
        print("results/figures/wilson_loop_area_decay.png")
    else:
        print()
        print("No positive Wilson loop values available for log plot.")


if __name__ == "__main__":
    main()
