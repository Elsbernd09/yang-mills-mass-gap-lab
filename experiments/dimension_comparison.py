"""
2D / 3D / 4D SU(2) lattice gauge theory comparison.

This experiment demonstrates that the project is not restricted to a 2D toy
model. It compares finite periodic SU(2) lattices in:

- 2D
- 3D
- 4D

The 4D lattice is intentionally small because computational cost grows quickly.
This is still a finite-lattice numerical experiment, not a continuum proof.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.dimensional_analysis import (
    run_dimension_comparison,
    dimension_summaries_as_rows,
    theoretical_plaquettes_per_site,
)


def main() -> None:
    shapes = [
        (8, 8),
        (4, 4, 4),
        (3, 3, 3, 3),
    ]

    beta = 2.0
    sweeps = 30
    burn_in = 5
    epsilon = 0.16
    seed = 2026

    print("Running 2D / 3D / 4D SU(2) dimension comparison...")
    print(f"Shapes: {shapes}")
    print(f"Beta: {beta}")
    print(f"Sweeps: {sweeps}")
    print(f"Burn-in: {burn_in}")
    print(f"Epsilon: {epsilon}")
    print()

    summaries = run_dimension_comparison(
        shapes=shapes,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        seed=seed,
        burn_in=burn_in,
    )

    rows = dimension_summaries_as_rows(summaries)

    print("Dimension comparison:")
    print("-" * 140)
    print(
        "Shape          Dim  Sites  Links  Plaquettes  C(d,2)  "
        "Action/Site   Action/Plaq   Avg Plaquette   Acceptance   Runtime(s)"
    )
    print("-" * 140)

    for row in rows:
        planes = theoretical_plaquettes_per_site(int(row["dimension"]))

        print(
            f"{row['shape']:<14} "
            f"{row['dimension']:>3d} "
            f"{row['sites']:>6d} "
            f"{row['links']:>6d} "
            f"{row['plaquettes']:>10d} "
            f"{planes:>7d} "
            f"{row['action_density_per_site']:>12.8f} "
            f"{row['action_density_per_plaquette']:>13.8f} "
            f"{row['final_average_plaquette']:>15.8f} "
            f"{row['mean_acceptance_rate']:>12.8f} "
            f"{row['runtime_seconds']:>11.4f}"
        )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "dimension_comparison.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    dimensions = np.asarray([row["dimension"] for row in rows], dtype=int)
    action_site = np.asarray([row["action_density_per_site"] for row in rows], dtype=float)
    action_plaq = np.asarray(
        [row["action_density_per_plaquette"] for row in rows],
        dtype=float,
    )
    plaquette = np.asarray([row["final_average_plaquette"] for row in rows], dtype=float)
    runtime = np.asarray([row["runtime_seconds"] for row in rows], dtype=float)

    plt.figure()
    plt.plot(dimensions, action_site, marker="o")
    plt.xlabel("Dimension")
    plt.ylabel("Wilson action / site")
    plt.title("Action Density per Site Across Dimensions")
    plt.xticks(dimensions)
    plt.tight_layout()
    plt.savefig(figures_dir / "dimension_action_per_site.png", dpi=200)

    plt.figure()
    plt.plot(dimensions, action_plaq, marker="o")
    plt.xlabel("Dimension")
    plt.ylabel("Wilson action / plaquette")
    plt.title("Action Density per Plaquette Across Dimensions")
    plt.xticks(dimensions)
    plt.tight_layout()
    plt.savefig(figures_dir / "dimension_action_per_plaquette.png", dpi=200)

    plt.figure()
    plt.plot(dimensions, plaquette, marker="o")
    plt.xlabel("Dimension")
    plt.ylabel("Average plaquette")
    plt.title("Average Plaquette Across Dimensions")
    plt.xticks(dimensions)
    plt.tight_layout()
    plt.savefig(figures_dir / "dimension_average_plaquette.png", dpi=200)

    plt.figure()
    plt.plot(dimensions, runtime, marker="o")
    plt.xlabel("Dimension")
    plt.ylabel("Runtime seconds")
    plt.title("Runtime Across Dimensions")
    plt.xticks(dimensions)
    plt.tight_layout()
    plt.savefig(figures_dir / "dimension_runtime.png", dpi=200)

    print()
    print(f"Saved data: {csv_path}")
    print("Saved figures:")
    print("results/figures/dimension_action_per_site.png")
    print("results/figures/dimension_action_per_plaquette.png")
    print("results/figures/dimension_average_plaquette.png")
    print("results/figures/dimension_runtime.png")


if __name__ == "__main__":
    main()
