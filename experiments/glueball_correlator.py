"""
Gauge-invariant scalar glueball-style Euclidean correlator experiment.

This experiment constructs a simple scalar spatial-plaquette operator O(t) on
each Euclidean time slice of a finite 3D SU(2) lattice.

The workflow is:

1. Thermalize a lattice.
2. Advance the Markov chain between measurements.
3. Measure O(t) on many gauge configurations.
4. Build the ensemble raw correlator <O(t) O(t + tau)>.
5. Subtract <O>^2 to form the connected correlator.
6. Normalize by C(0).
7. Save measurements and figures.

This is an exploratory finite-lattice spectroscopy pipeline. It does not prove
the Yang-Mills mass gap and is not a continuum glueball calculation.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.glueball import (
    ensemble_connected_correlator,
    normalized_connected_correlator,
    scalar_glueball_time_series,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep


def main() -> None:
    shape = (8, 6, 6)
    time_direction = 0
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 80
    number_of_configurations = 120
    sweeps_between_measurements = 3
    seed = 2026

    print("Scalar Glueball-Style Euclidean Correlator")
    print("=" * 92)
    print(f"Lattice shape:                  {shape}")
    print(f"Euclidean time direction:       {time_direction}")
    print(f"Beta:                           {beta}")
    print(f"Proposal epsilon:               {epsilon}")
    print(f"Thermalization sweeps:          {thermalization_sweeps}")
    print(f"Measured configurations:        {number_of_configurations}")
    print(
        "Sweeps between measurements:    "
        f"{sweeps_between_measurements}"
    )
    print()

    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    print("Thermalizing lattice...")

    thermal_acceptance = []

    for _ in range(thermalization_sweeps):
        thermal_acceptance.append(
            metropolis_sweep(
                lattice=lattice,
                beta=beta,
                epsilon=epsilon,
            )
        )

    print(
        "Mean thermalization acceptance: "
        f"{np.mean(thermal_acceptance):.8f}"
    )
    print()

    operator_ensemble = []
    measurement_acceptance = []

    print("Collecting gauge-invariant operator measurements...")

    for configuration_index in range(
        number_of_configurations
    ):
        for _ in range(sweeps_between_measurements):
            measurement_acceptance.append(
                metropolis_sweep(
                    lattice=lattice,
                    beta=beta,
                    epsilon=epsilon,
                )
            )

        operator_series = scalar_glueball_time_series(
            lattice=lattice,
            time_direction=time_direction,
        )

        operator_ensemble.append(
            operator_series
        )

        if (
            configuration_index == 0
            or (configuration_index + 1) % 20 == 0
        ):
            print(
                "  collected configuration "
                f"{configuration_index + 1:03d}/"
                f"{number_of_configurations:03d}"
            )

    ensemble = np.asarray(
        operator_ensemble,
        dtype=float,
    )

    result = ensemble_connected_correlator(
        operator_series_ensemble=ensemble,
    )

    normalized = normalized_connected_correlator(
        result.correlation
    )

    print()
    print("Measurement summary")
    print("-" * 92)
    print(
        f"Operator ensemble shape:        {ensemble.shape}"
    )
    print(
        "Mean measurement acceptance:   "
        f"{np.mean(measurement_acceptance):.8f}"
    )
    print(
        f"Mean scalar operator <O>:        "
        f"{result.mean_operator:.12f}"
    )
    print()

    print("Connected Euclidean correlator")
    print("-" * 92)
    print(
        f"{'Lag':>8}"
        f"{'Raw Correlation':>24}"
        f"{'Connected C(t)':>24}"
        f"{'Normalized C(t)':>24}"
    )
    print("-" * 92)

    for lag, raw, connected, norm in zip(
        result.lags,
        result.raw_correlation,
        result.correlation,
        normalized,
    ):
        print(
            f"{lag:>8d}"
            f"{raw:>24.12e}"
            f"{connected:>24.12e}"
            f"{norm:>24.12e}"
        )

    symmetry_errors = []

    for lag in range(1, result.temporal_extent):
        partner = (
            result.temporal_extent - lag
        ) % result.temporal_extent

        symmetry_errors.append(
            abs(
                result.correlation[lag]
                - result.correlation[partner]
            )
        )

    max_symmetry_error = (
        max(symmetry_errors)
        if symmetry_errors
        else 0.0
    )

    print()
    print(
        "Maximum periodic correlator symmetry error: "
        f"{max_symmetry_error:.12e}"
    )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    ensemble_path = (
        data_dir / "scalar_glueball_operator_ensemble.csv"
    )
    correlator_path = (
        data_dir / "scalar_glueball_correlator.csv"
    )

    with ensemble_path.open("w", newline="") as file:
        writer = csv.writer(file)

        header = [
            "configuration"
        ] + [
            f"t_{t}"
            for t in range(
                result.temporal_extent
            )
        ]

        writer.writerow(header)

        for index, series in enumerate(ensemble):
            writer.writerow(
                [index] + list(series)
            )

    correlator_rows = [
        {
            "lag": int(lag),
            "raw_correlation": float(raw),
            "connected_correlation": float(connected),
            "normalized_correlation": float(norm),
        }
        for lag, raw, connected, norm in zip(
            result.lags,
            result.raw_correlation,
            result.correlation,
            normalized,
        )
    ]

    with correlator_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                correlator_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(correlator_rows)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure()
    plt.plot(
        result.lags,
        result.correlation,
        marker="o",
    )
    plt.xlabel("Euclidean time lag")
    plt.ylabel("Connected correlation")
    plt.title(
        "Scalar Glueball-Style Connected Correlator"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "scalar_glueball_connected_correlator.png",
        dpi=200,
    )

    plt.figure()
    plt.plot(
        result.lags,
        normalized,
        marker="o",
    )
    plt.xlabel("Euclidean time lag")
    plt.ylabel("C(t) / C(0)")
    plt.title(
        "Normalized Scalar Glueball-Style Correlator"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "scalar_glueball_normalized_correlator.png",
        dpi=200,
    )

    plt.figure()
    plt.imshow(
        ensemble,
        aspect="auto",
    )
    plt.xlabel("Euclidean time slice")
    plt.ylabel("Configuration index")
    plt.title(
        "Scalar Glueball-Style Operator Ensemble"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "scalar_glueball_operator_ensemble.png",
        dpi=200,
    )

    print()
    print("Saved data:")
    print(ensemble_path)
    print(correlator_path)
    print()
    print("Saved figures:")
    print(
        "results/figures/"
        "scalar_glueball_connected_correlator.png"
    )
    print(
        "results/figures/"
        "scalar_glueball_normalized_correlator.png"
    )
    print(
        "results/figures/"
        "scalar_glueball_operator_ensemble.png"
    )


if __name__ == "__main__":
    main()
