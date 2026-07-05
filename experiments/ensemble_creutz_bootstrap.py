"""
Post-thermalization ensemble Wilson-loop and block-bootstrap Creutz analysis.

The experiment measures a complete rectangular Wilson-loop basis on many
Markov-chain configurations.

The configuration-level loop matrix is block bootstrapped.

Every bootstrap replicate reconstructs:

1. mean Wilson loops,
2. all supported Creutz ratios.

The nonlinear ratio is therefore evaluated after joint resampling rather than
through independent loop error propagation.

The output is a finite-lattice string-tension-style diagnostic.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.diagnostics import (
    diagnose_autocorrelation,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import (
    metropolis_sweep,
)
from ymlab.wilson_ensemble import (
    block_bootstrap_creutz_ratios,
    create_rectangular_loop_basis,
    measure_wilson_loop_basis,
    square_creutz_plateau_values,
)


def collect_loop_ensemble(
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    number_of_configurations,
    sweeps_between_measurements,
    basis,
    mu,
    nu,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    thermal_acceptance = []
    measurement_acceptance = []

    print(
        "Thermalizing SU(2) lattice..."
    )

    for _ in range(
        thermalization_sweeps
    ):
        thermal_acceptance.append(
            metropolis_sweep(
                lattice=lattice,
                beta=beta,
                epsilon=epsilon,
            )
        )

    measurements = []

    print(
        "Collecting post-thermalization Wilson-loop ensemble..."
    )

    for configuration_index in range(
        number_of_configurations
    ):
        for _ in range(
            sweeps_between_measurements
        ):
            measurement_acceptance.append(
                metropolis_sweep(
                    lattice=lattice,
                    beta=beta,
                    epsilon=epsilon,
                )
            )

        measurements.append(
            measure_wilson_loop_basis(
                lattice=lattice,
                basis=basis,
                mu=mu,
                nu=nu,
            )
        )

        if (
            configuration_index == 0
            or (
                configuration_index + 1
            ) % 50 == 0
        ):
            print(
                "  collected configuration "
                f"{configuration_index + 1:04d}/"
                f"{number_of_configurations:04d}"
            )

    return (
        np.asarray(
            measurements,
            dtype=float,
        ),
        float(
            np.mean(
                thermal_acceptance
            )
        ),
        float(
            np.mean(
                measurement_acceptance
            )
        ),
    )


def main() -> None:
    shape = (
        8,
        8,
    )

    beta = 2.0
    epsilon = 0.18

    thermalization_sweeps = 120
    number_of_configurations = 500
    sweeps_between_measurements = 2

    maximum_width = 4
    maximum_height = 4

    mu = 0
    nu = 1

    n_bootstrap = 1500
    seed = 2026

    basis = create_rectangular_loop_basis(
        maximum_width=maximum_width,
        maximum_height=maximum_height,
    )

    print(
        "Ensemble Wilson-Loop and Block-Bootstrap Creutz Analysis"
    )

    print(
        "=" * 128
    )

    print(
        f"Lattice shape:                    {shape}"
    )

    print(
        f"Beta:                             {beta}"
    )

    print(
        f"Proposal epsilon:                 {epsilon}"
    )

    print(
        f"Thermalization sweeps:            "
        f"{thermalization_sweeps}"
    )

    print(
        f"Measured configurations:          "
        f"{number_of_configurations}"
    )

    print(
        "Sweeps between measurements:      "
        f"{sweeps_between_measurements}"
    )

    print(
        f"Wilson-loop basis:                "
        f"{basis.labels}"
    )

    print(
        f"Bootstrap replicates:             "
        f"{n_bootstrap}"
    )

    print()

    (
        loop_ensemble,
        thermal_acceptance,
        measurement_acceptance,
    ) = collect_loop_ensemble(
        shape=shape,
        beta=beta,
        epsilon=epsilon,
        thermalization_sweeps=(
            thermalization_sweeps
        ),
        number_of_configurations=(
            number_of_configurations
        ),
        sweeps_between_measurements=(
            sweeps_between_measurements
        ),
        basis=basis,
        mu=mu,
        nu=nu,
        seed=seed,
    )

    plaquette_column = basis.index(
        (
            1,
            1,
        )
    )

    w11_diagnostics = (
        diagnose_autocorrelation(
            loop_ensemble[
                :,
                plaquette_column,
            ]
        )
    )

    tau_int = float(
        w11_diagnostics
        .integrated_autocorrelation_time
    )

    suggested_block_size = max(
        1,
        int(
            np.ceil(
                2.0
                * tau_int
            )
        ),
    )

    suggested_block_size = min(
        suggested_block_size,
        number_of_configurations,
    )

    print()

    print(
        "Autocorrelation-informed block selection"
    )

    print(
        "-" * 128
    )

    print(
        f"W(1,1) integrated autocorrelation time: "
        f"{tau_int:.8f}"
    )

    print(
        f"Suggested block size ceil(2 tau):       "
        f"{suggested_block_size}"
    )

    print()

    print(
        "Running circular block bootstrap..."
    )

    bootstrap = block_bootstrap_creutz_ratios(
        loop_ensemble=loop_ensemble,
        basis=basis,
        n_bootstrap=n_bootstrap,
        block_size=suggested_block_size,
        seed=seed + 10000,
    )

    loop_standard_errors = np.std(
        bootstrap.loop_mean_samples,
        axis=0,
        ddof=1,
    )

    print()

    print(
        "Wilson-loop ensemble summary"
    )

    print(
        "-" * 128
    )

    print(
        f"{'Loop':<12}"
        f"{'Mean':>20}"
        f"{'Bootstrap SE':>20}"
        f"{'Relative SE':>20}"
    )

    print(
        "-" * 128
    )

    for index, (
        shape_value,
        label,
    ) in enumerate(
        zip(
            basis.shapes,
            basis.labels,
        )
    ):
        mean = (
            bootstrap.central_loop_means[
                index
            ]
        )

        standard_error = (
            loop_standard_errors[
                index
            ]
        )

        relative_error = (
            standard_error
            / abs(
                mean
            )
            if mean != 0.0
            else np.nan
        )

        print(
            f"{label:<12}"
            f"{mean:>20.10e}"
            f"{standard_error:>20.10e}"
            f"{relative_error:>20.10e}"
        )

    print()

    print(
        "Block-bootstrap Creutz-ratio summary"
    )

    print(
        "-" * 128
    )

    print(
        f"{'Creutz':<12}"
        f"{'Central':>16}"
        f"{'Boot mean':>16}"
        f"{'Boot SE':>16}"
        f"{'95% lower':>16}"
        f"{'95% upper':>16}"
        f"{'Valid N':>12}"
        f"{'Valid frac':>14}"
    )

    print(
        "-" * 128
    )

    for point in bootstrap.creutz_points:
        label = (
            f"chi({point.width},{point.height})"
        )

        print(
            f"{label:<12}"
            f"{point.central_value:>16.8f}"
            f"{point.bootstrap_mean:>16.8f}"
            f"{point.standard_error:>16.8f}"
            f"{point.lower_95:>16.8f}"
            f"{point.upper_95:>16.8f}"
            f"{point.finite_replicates:>12d}"
            f"{point.valid_fraction:>14.8f}"
        )

    square_values = square_creutz_plateau_values(
        bootstrap,
        minimum_valid_fraction=0.80,
    )

    print()

    print(
        "Square Creutz string-tension-style diagnostic"
    )

    print(
        "-" * 128
    )

    print(
        f"Valid square Creutz values:        "
        f"{square_values}"
    )

    if len(
        square_values
    ) > 0:
        print(
            f"Mean valid square Creutz value:    "
            f"{np.mean(square_values):.12f}"
        )

        if len(
            square_values
        ) > 1:
            print(
                f"Std across square Creutz values:   "
                f"{np.std(square_values, ddof=1):.12f}"
            )

    print()

    print(
        "Simulation acceptance"
    )

    print(
        "-" * 128
    )

    print(
        f"Mean thermal acceptance:           "
        f"{thermal_acceptance:.8f}"
    )

    print(
        f"Mean measurement acceptance:       "
        f"{measurement_acceptance:.8f}"
    )

    data_dir = Path(
        "results/data"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = Path(
        "results/figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loop_path = (
        data_dir
        / "ensemble_wilson_loop_summary.csv"
    )

    loop_rows = []

    for index, (
        width,
        height,
    ) in enumerate(
        basis.shapes
    ):
        mean = (
            bootstrap.central_loop_means[
                index
            ]
        )

        standard_error = (
            loop_standard_errors[
                index
            ]
        )

        loop_rows.append(
            {
                "width": width,
                "height": height,
                "label": basis.labels[
                    index
                ],
                "mean": mean,
                "bootstrap_standard_error": (
                    standard_error
                ),
                "relative_standard_error": (
                    standard_error
                    / abs(
                        mean
                    )
                    if mean != 0.0
                    else np.nan
                ),
                "block_size": (
                    suggested_block_size
                ),
                "bootstrap_replicates": (
                    n_bootstrap
                ),
            }
        )

    with loop_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                loop_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            loop_rows
        )

    creutz_path = (
        data_dir
        / "ensemble_creutz_bootstrap_summary.csv"
    )

    creutz_rows = []

    for point in bootstrap.creutz_points:
        creutz_rows.append(
            {
                "width": point.width,
                "height": point.height,
                "central_value": (
                    point.central_value
                ),
                "bootstrap_mean": (
                    point.bootstrap_mean
                ),
                "standard_error": (
                    point.standard_error
                ),
                "lower_95": (
                    point.lower_95
                ),
                "median": (
                    point.median
                ),
                "upper_95": (
                    point.upper_95
                ),
                "finite_replicates": (
                    point.finite_replicates
                ),
                "requested_replicates": (
                    point.requested_replicates
                ),
                "valid_fraction": (
                    point.valid_fraction
                ),
                "block_size": (
                    suggested_block_size
                ),
            }
        )

    with creutz_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                creutz_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            creutz_rows
        )

    areas = np.asarray(
        [
            width
            * height
            for width, height in basis.shapes
        ],
        dtype=float,
    )

    means = np.asarray(
        bootstrap.central_loop_means,
        dtype=float,
    )

    plt.figure()

    positive = (
        means > 0.0
    )

    plt.errorbar(
        areas[
            positive
        ],
        means[
            positive
        ],
        yerr=loop_standard_errors[
            positive
        ],
        marker="o",
        linestyle="none",
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Rectangular loop area R T"
    )

    plt.ylabel(
        "Mean Wilson loop"
    )

    plt.title(
        "Post-Thermalization Ensemble Wilson Loops"
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "ensemble_wilson_loops_vs_area.png",
        dpi=200,
    )

    plt.figure()

    square_points = [
        point
        for point in bootstrap.creutz_points
        if (
            point.width == point.height
            and point.valid_fraction >= 0.80
            and np.isfinite(
                point.bootstrap_mean
            )
            and np.isfinite(
                point.standard_error
            )
        )
    ]

    if len(
        square_points
    ) > 0:
        sizes = np.asarray(
            [
                point.width
                for point in square_points
            ],
            dtype=int,
        )

        values = np.asarray(
            [
                point.bootstrap_mean
                for point in square_points
            ],
            dtype=float,
        )

        errors = np.asarray(
            [
                point.standard_error
                for point in square_points
            ],
            dtype=float,
        )

        plt.errorbar(
            sizes,
            values,
            yerr=errors,
            marker="o",
            linestyle="none",
        )

    plt.xlabel(
        "Square Creutz size R = T"
    )

    plt.ylabel(
        "Creutz ratio"
    )

    plt.title(
        "Block-Bootstrap Square Creutz Diagnostics"
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "ensemble_square_creutz_bootstrap.png",
        dpi=200,
    )

    print()

    print(
        "Saved data:"
    )

    print(
        loop_path
    )

    print(
        creutz_path
    )

    print()

    print(
        "Saved figures:"
    )

    print(
        "results/figures/"
        "ensemble_wilson_loops_vs_area.png"
    )

    print(
        "results/figures/"
        "ensemble_square_creutz_bootstrap.png"
    )


if __name__ == "__main__":
    main()
