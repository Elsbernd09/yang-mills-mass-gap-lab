"""
Beta and finite-size scan for SU(2) lattice gauge theory.

This experiment runs a small grid over:

- beta values,
- lattice sizes.

For each setting, it runs multiple independent Markov chains and reports
bootstrap uncertainty estimates for:

1. Final Wilson action
2. Final average plaquette
3. Mean acceptance rate

This is a finite-lattice numerical experiment, not a continuum proof.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.scans import run_beta_size_grid, scan_result_table


def main() -> None:
    shapes = [(4, 4), (6, 6), (8, 8)]
    betas = [1.0, 1.5, 2.0, 2.5]
    sweeps = 30
    epsilon = 0.18
    seeds = [2026, 2027, 2028]
    n_bootstrap = 500

    print("Running SU(2) beta/finite-size scan...")
    print(f"Shapes: {shapes}")
    print(f"Betas: {betas}")
    print(f"Sweeps per chain: {sweeps}")
    print(f"Seeds per point: {seeds}")
    print()

    result = run_beta_size_grid(
        shapes=shapes,
        betas=betas,
        sweeps=sweeps,
        epsilon=epsilon,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )

    rows = scan_result_table(result)

    print("Scan results:")
    print("-" * 120)
    print(
        "Shape      Volume   Beta    Plaquette Mean   Plaquette SE     "
        "Action Mean      Action SE       Acceptance Mean"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['shape']:<10} "
            f"{row['volume']:7.0f} "
            f"{row['beta']:6.2f} "
            f"{row['plaquette_mean']:16.8f} "
            f"{row['plaquette_se']:14.8f} "
            f"{row['action_mean']:16.8f} "
            f"{row['action_se']:14.8f} "
            f"{row['acceptance_mean']:16.8f}"
        )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "beta_finite_size_scan.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved data: {csv_path}")

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot average plaquette vs beta for each lattice size.
    plt.figure()

    for shape in shapes:
        shape_rows = [row for row in rows if row["shape"] == str(shape)]
        beta_values = np.array([row["beta"] for row in shape_rows], dtype=float)
        plaquette_values = np.array(
            [row["plaquette_mean"] for row in shape_rows],
            dtype=float,
        )
        plaquette_errors = np.array(
            [row["plaquette_se"] for row in shape_rows],
            dtype=float,
        )

        plt.errorbar(
            beta_values,
            plaquette_values,
            yerr=plaquette_errors,
            marker="o",
            capsize=3,
            label=f"L={shape[0]}",
        )

    plt.xlabel("Beta")
    plt.ylabel("Average plaquette")
    plt.title("Average Plaquette vs Beta Across Lattice Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "plaquette_vs_beta_finite_size.png", dpi=200)

    # Plot Wilson action per volume vs beta.
    plt.figure()

    for shape in shapes:
        shape_rows = [row for row in rows if row["shape"] == str(shape)]
        beta_values = np.array([row["beta"] for row in shape_rows], dtype=float)
        action_density = np.array(
            [row["action_mean"] / row["volume"] for row in shape_rows],
            dtype=float,
        )

        plt.plot(beta_values, action_density, marker="o", label=f"L={shape[0]}")

    plt.xlabel("Beta")
    plt.ylabel("Wilson action / volume")
    plt.title("Action Density vs Beta Across Lattice Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "action_density_vs_beta.png", dpi=200)

    # Plot acceptance vs beta.
    plt.figure()

    for shape in shapes:
        shape_rows = [row for row in rows if row["shape"] == str(shape)]
        beta_values = np.array([row["beta"] for row in shape_rows], dtype=float)
        acceptance_values = np.array(
            [row["acceptance_mean"] for row in shape_rows],
            dtype=float,
        )

        plt.plot(beta_values, acceptance_values, marker="o", label=f"L={shape[0]}")

    plt.xlabel("Beta")
    plt.ylabel("Mean acceptance rate")
    plt.title("Metropolis Acceptance vs Beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "acceptance_vs_beta.png", dpi=200)

    print("Saved figures:")
    print("results/figures/plaquette_vs_beta_finite_size.png")
    print("results/figures/action_density_vs_beta.png")
    print("results/figures/acceptance_vs_beta.png")


if __name__ == "__main__":
    main()
