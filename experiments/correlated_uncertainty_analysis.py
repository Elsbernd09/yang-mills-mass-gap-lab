"""
Correlation-aware uncertainty analysis for SU(2) Monte Carlo measurements.

This experiment compares:

1. Naive individual-sample bootstrap.
2. Circular moving-block bootstrap.
3. Delete-one jackknife.
4. Delete-one-block jackknife.
5. Block-size sensitivity.

The primary observable is the average plaquette measured along a post-burn-in
Markov chain.

The goal is to quantify how uncertainty estimates change when temporal
correlation is treated more carefully.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.diagnostics import diagnose_autocorrelation
from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis
from ymlab.resampling import (
    block_size_sensitivity,
    delete_one_block_jackknife_mean,
    delete_one_jackknife_mean,
    moving_block_bootstrap_mean,
    naive_bootstrap_mean,
)


def main() -> None:
    shape = (8, 8)
    beta = 2.0
    sweeps = 300
    burn_in = 75
    epsilon = 0.18
    measurement_interval = 1
    seed = 2026
    n_bootstrap = 2000

    print("Correlation-Aware Uncertainty Analysis")
    print("=" * 88)
    print(f"Lattice shape:             {shape}")
    print(f"Beta:                      {beta}")
    print(f"Total sweeps:              {sweeps}")
    print(f"Burn-in sweeps:            {burn_in}")
    print(f"Measurement interval:      {measurement_interval}")
    print(f"Proposal epsilon:           {epsilon}")
    print(f"Bootstrap resamples:        {n_bootstrap}")
    print()

    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    print("Generating post-burn-in SU(2) Markov-chain measurements...")

    result = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=measurement_interval,
        burn_in=burn_in,
    )

    plaquettes = np.asarray(
        result.average_plaquettes,
        dtype=float,
    )

    print(f"Recorded measurements:     {len(plaquettes)}")
    print(f"Mean average plaquette:     {np.mean(plaquettes):.12f}")
    print()

    diagnostics = diagnose_autocorrelation(
        values=plaquettes,
        burn_in=0,
        max_lag=min(60, len(plaquettes) - 1),
    )

    tau_int = diagnostics.integrated_autocorrelation_time
    effective_n = diagnostics.effective_sample_size

    print("Autocorrelation diagnostics")
    print("-" * 88)
    print(f"Integrated autocorrelation time: {tau_int:.8f}")
    print(f"Effective sample size:           {effective_n:.8f}")
    print()

    suggested_block_size = max(
        2,
        int(np.ceil(2.0 * tau_int)),
    )
    suggested_block_size = min(
        suggested_block_size,
        max(2, len(plaquettes) // 4),
    )

    candidate_block_sizes = sorted(
        {
            1,
            2,
            4,
            suggested_block_size,
            8,
            16,
            32,
        }
    )

    candidate_block_sizes = [
        block_size
        for block_size in candidate_block_sizes
        if block_size <= len(plaquettes) // 2
    ]

    print(f"Suggested block size from 2*tau_int: {suggested_block_size}")
    print(f"Block sizes tested:                    {candidate_block_sizes}")
    print()

    naive = naive_bootstrap_mean(
        values=plaquettes,
        n_bootstrap=n_bootstrap,
        seed=seed + 100,
    )

    blocked = moving_block_bootstrap_mean(
        values=plaquettes,
        block_size=suggested_block_size,
        n_bootstrap=n_bootstrap,
        seed=seed + 200,
    )

    jackknife = delete_one_jackknife_mean(
        values=plaquettes,
    )

    block_jackknife = delete_one_block_jackknife_mean(
        values=plaquettes,
        block_size=suggested_block_size,
    )

    estimates = [
        naive,
        blocked,
        jackknife,
        block_jackknife,
    ]

    print("Uncertainty estimator comparison")
    print("-" * 122)
    print(
        f"{'Method':<40}"
        f"{'Estimate':>18}"
        f"{'Std Error':>18}"
        f"{'Lower 95% CI':>20}"
        f"{'Upper 95% CI':>20}"
    )
    print("-" * 122)

    for estimate in estimates:
        print(
            f"{estimate.method:<40}"
            f"{estimate.estimate:>18.12f}"
            f"{estimate.standard_error:>18.12f}"
            f"{estimate.lower_ci:>20.12f}"
            f"{estimate.upper_ci:>20.12f}"
        )

    sensitivity = block_size_sensitivity(
        values=plaquettes,
        block_sizes=candidate_block_sizes,
        n_bootstrap=1000,
        seed=seed + 300,
    )

    print()
    print("Moving-block bootstrap block-size sensitivity")
    print("-" * 88)
    print(
        f"{'Block Size':>12}"
        f"{'Approx Blocks':>16}"
        f"{'Std Error':>20}"
        f"{'Lower CI':>20}"
        f"{'Upper CI':>20}"
    )
    print("-" * 88)

    for point in sensitivity:
        print(
            f"{point.block_size:>12d}"
            f"{point.number_of_blocks:>16d}"
            f"{point.standard_error:>20.12f}"
            f"{point.lower_ci:>20.12f}"
            f"{point.upper_ci:>20.12f}"
        )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        data_dir / "correlated_uncertainty_comparison.csv"
    )
    sensitivity_path = (
        data_dir / "block_size_sensitivity.csv"
    )

    comparison_rows = [
        {
            "method": estimate.method,
            "estimate": estimate.estimate,
            "standard_error": estimate.standard_error,
            "lower_ci": estimate.lower_ci,
            "upper_ci": estimate.upper_ci,
            "tau_int": tau_int,
            "effective_sample_size": effective_n,
        }
        for estimate in estimates
    ]

    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    sensitivity_rows = [
        {
            "block_size": point.block_size,
            "number_of_blocks": point.number_of_blocks,
            "standard_error": point.standard_error,
            "lower_ci": point.lower_ci,
            "upper_ci": point.upper_ci,
        }
        for point in sensitivity
    ]

    with sensitivity_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(sensitivity_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(
        np.arange(len(plaquettes)),
        plaquettes,
    )
    plt.xlabel("Measurement index")
    plt.ylabel("Average plaquette")
    plt.title("Post-Burn-In Average Plaquette Trace")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "correlated_uncertainty_plaquette_trace.png",
        dpi=200,
    )

    plt.figure()
    plt.plot(
        np.arange(len(diagnostics.autocorrelation)),
        diagnostics.autocorrelation,
        marker="o",
    )
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Average Plaquette Autocorrelation")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "correlated_uncertainty_autocorrelation.png",
        dpi=200,
    )

    block_sizes = np.asarray(
        [point.block_size for point in sensitivity],
        dtype=int,
    )
    standard_errors = np.asarray(
        [point.standard_error for point in sensitivity],
        dtype=float,
    )

    plt.figure()
    plt.plot(
        block_sizes,
        standard_errors,
        marker="o",
    )
    plt.xlabel("Moving-block bootstrap block size")
    plt.ylabel("Estimated standard error")
    plt.title("Block-Size Sensitivity of Plaquette Uncertainty")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "block_size_uncertainty_sensitivity.png",
        dpi=200,
    )

    method_names = [
        estimate.method
        for estimate in estimates
    ]
    method_errors = [
        estimate.standard_error
        for estimate in estimates
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(
        np.arange(len(method_names)),
        method_errors,
    )
    plt.xticks(
        np.arange(len(method_names)),
        method_names,
        rotation=25,
        ha="right",
    )
    plt.ylabel("Estimated standard error")
    plt.title("Uncertainty Estimator Comparison")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "uncertainty_estimator_comparison.png",
        dpi=200,
    )

    print()
    print("Saved data:")
    print(comparison_path)
    print(sensitivity_path)
    print()
    print("Saved figures:")
    print(
        "results/figures/"
        "correlated_uncertainty_plaquette_trace.png"
    )
    print(
        "results/figures/"
        "correlated_uncertainty_autocorrelation.png"
    )
    print(
        "results/figures/"
        "block_size_uncertainty_sensitivity.png"
    )
    print(
        "results/figures/"
        "uncertainty_estimator_comparison.png"
    )


if __name__ == "__main__":
    main()
