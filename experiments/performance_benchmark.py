"""
Performance benchmark for the Yang-Mills Mass Gap Laboratory.

This experiment measures runtime scaling across small 2D, 3D, and 4D SU(2)
lattice gauge theory simulations.

The goal is to make the computational cost transparent.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.performance import benchmark_many, performance_results_as_rows


def main() -> None:
    shapes = [
        (4, 4),
        (6, 6),
        (8, 8),
        (3, 3, 3),
        (4, 4, 4),
        (2, 2, 2, 2),
        (3, 3, 3, 3),
    ]

    beta = 2.0
    epsilon = 0.16
    sweeps = 5
    seed = 2026

    print("Running performance benchmark...")
    print(f"Beta: {beta}")
    print(f"Epsilon: {epsilon}")
    print(f"Sweeps per shape: {sweeps}")
    print()

    results = benchmark_many(
        shapes=shapes,
        beta=beta,
        epsilon=epsilon,
        sweeps=sweeps,
        seed=seed,
    )

    rows = performance_results_as_rows(results)

    print("Performance results:")
    print("-" * 150)
    print(
        "Shape          Dim  Sites  Links  Plaquettes  AvgSweep(s)   "
        "Updates/s       Plaquettes/s    Accept"
    )
    print("-" * 150)

    for row in rows:
        print(
            f"{row['shape']:<14} "
            f"{row['dimension']:>3d} "
            f"{row['sites']:>6d} "
            f"{row['links']:>6d} "
            f"{row['plaquettes']:>10d} "
            f"{row['average_sweep_seconds']:>12.6f} "
            f"{row['link_updates_per_second']:>12.2f} "
            f"{row['plaquettes_per_second']:>14.2f} "
            f"{row['mean_acceptance_rate']:>9.6f}"
        )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "performance_benchmark.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sites = np.asarray([row["sites"] for row in rows], dtype=float)
    links = np.asarray([row["links"] for row in rows], dtype=float)
    plaquettes = np.asarray([row["plaquettes"] for row in rows], dtype=float)
    sweep_times = np.asarray([row["average_sweep_seconds"] for row in rows], dtype=float)
    updates_per_second = np.asarray(
        [row["link_updates_per_second"] for row in rows],
        dtype=float,
    )
    dimensions = np.asarray([row["dimension"] for row in rows], dtype=int)

    plt.figure()
    plt.scatter(links, sweep_times)
    plt.xlabel("Number of links")
    plt.ylabel("Average sweep time (seconds)")
    plt.title("Metropolis Sweep Time vs Number of Links")
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_sweep_time_vs_links.png", dpi=200)

    plt.figure()
    plt.scatter(plaquettes, sweep_times)
    plt.xlabel("Number of plaquettes")
    plt.ylabel("Average sweep time (seconds)")
    plt.title("Metropolis Sweep Time vs Number of Plaquettes")
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_sweep_time_vs_plaquettes.png", dpi=200)

    plt.figure()
    plt.scatter(dimensions, sweep_times)
    plt.xlabel("Dimension")
    plt.ylabel("Average sweep time (seconds)")
    plt.title("Sweep Time Across Dimensions")
    plt.xticks(sorted(set(dimensions)))
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_sweep_time_by_dimension.png", dpi=200)

    plt.figure()
    plt.scatter(sites, updates_per_second)
    plt.xlabel("Number of sites")
    plt.ylabel("Link updates per second")
    plt.title("Update Throughput vs Lattice Sites")
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_updates_per_second.png", dpi=200)

    print()
    print(f"Saved data: {csv_path}")
    print("Saved figures:")
    print("results/figures/performance_sweep_time_vs_links.png")
    print("results/figures/performance_sweep_time_vs_plaquettes.png")
    print("results/figures/performance_sweep_time_by_dimension.png")
    print("results/figures/performance_updates_per_second.png")


if __name__ == "__main__":
    main()
