"""
Creutz ratio and string-tension-style scan for SU(2) lattice gauge theory.

This experiment:
1. Runs SU(2) Metropolis simulations at multiple beta values.
2. Measures rectangular Wilson loops.
3. Computes Creutz ratios.
4. Estimates a string-tension-like quantity from valid Creutz ratios.
5. Saves a beta-vs-string-tension-style plot.

This is an exploratory finite-lattice diagnostic. It does not prove
confinement or the Yang-Mills mass gap.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.creutz import (
    creutz_ratio_table,
    creutz_results_as_rows,
    estimate_string_tension,
    valid_creutz_values,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis


def main() -> None:
    shape = (8, 8)
    betas = [1.0, 1.5, 2.0, 2.5]
    sweeps = 80
    burn_in = 20
    epsilon = 0.18
    max_width = 4
    max_height = 4

    summary_rows = []
    all_rows = []

    print("Running Creutz ratio / string-tension-style scan...")
    print(f"Shape: {shape}")
    print(f"Betas: {betas}")
    print(f"Sweeps: {sweeps}")
    print(f"Burn-in: {burn_in}")
    print(f"Loop max size: {max_width} x {max_height}")
    print()

    for beta in betas:
        lattice = Lattice(shape=shape, cold_start=True, seed=int(1000 * beta) + 2026)

        print(f"Simulating beta={beta:.2f}...")
        result = run_metropolis(
            lattice=lattice,
            beta=beta,
            sweeps=sweeps,
            epsilon=epsilon,
            measurement_interval=1,
            burn_in=burn_in,
        )

        creutz_results = creutz_ratio_table(
            lattice=lattice,
            mu=0,
            nu=1,
            max_width=max_width,
            max_height=max_height,
        )

        values = valid_creutz_values(creutz_results)

        if len(values) > 0:
            string_tension_estimate = estimate_string_tension(creutz_results)
            string_tension_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            string_tension_estimate = float("nan")
            string_tension_std = float("nan")

        mean_acceptance = float(np.mean(result.acceptance_rates))

        print(f"  valid Creutz ratios: {len(values)}")
        print(f"  string-tension-style estimate: {string_tension_estimate:.8f}")
        print(f"  mean acceptance: {mean_acceptance:.8f}")
        print()

        summary_rows.append(
            {
                "beta": beta,
                "shape": str(shape),
                "valid_creutz_count": len(values),
                "string_tension_estimate": string_tension_estimate,
                "string_tension_std": string_tension_std,
                "mean_acceptance": mean_acceptance,
            }
        )

        for row in creutz_results_as_rows(creutz_results):
            row = dict(row)
            row["beta"] = beta
            row["shape"] = str(shape)
            all_rows.append(row)

    print("Summary:")
    print("-" * 90)
    print("Beta    ValidCount    StringTensionEstimate    StdAcrossRatios    Acceptance")
    print("-" * 90)

    for row in summary_rows:
        print(
            f"{row['beta']:4.1f} "
            f"{row['valid_creutz_count']:11d} "
            f"{row['string_tension_estimate']:24.8f} "
            f"{row['string_tension_std']:17.8f} "
            f"{row['mean_acceptance']:13.8f}"
        )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "creutz_string_tension_summary.csv"
    detail_path = data_dir / "creutz_ratio_details.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with detail_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    beta_values = np.asarray([row["beta"] for row in summary_rows], dtype=float)
    tension_values = np.asarray(
        [row["string_tension_estimate"] for row in summary_rows],
        dtype=float,
    )
    tension_errors = np.asarray(
        [row["string_tension_std"] for row in summary_rows],
        dtype=float,
    )

    finite_mask = np.isfinite(tension_values)

    plt.figure()
    plt.errorbar(
        beta_values[finite_mask],
        tension_values[finite_mask],
        yerr=tension_errors[finite_mask],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Beta")
    plt.ylabel("Creutz-ratio string-tension-style estimate")
    plt.title("Exploratory String-Tension-Style Estimate vs Beta")
    plt.tight_layout()
    plt.savefig(figures_dir / "creutz_string_tension_vs_beta.png", dpi=200)

    print()
    print("Saved data:")
    print(summary_path)
    print(detail_path)
    print("Saved figure:")
    print("results/figures/creutz_string_tension_vs_beta.png")


if __name__ == "__main__":
    main()
