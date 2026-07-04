"""
Covariance-aware correlated spectroscopy fit experiment.

The experiment measures a smeared scalar gauge-invariant operator on a finite
3D SU(2) lattice.

It then:

1. Builds the ensemble connected Euclidean correlator.
2. Resamples complete configurations.
3. Generates bootstrap connected-correlator replicates.
4. Estimates the time-time covariance matrix.
5. Measures covariance correlations and eigenspectrum.
6. Applies diagonal shrinkage.
7. Constructs a regularized covariance pseudoinverse.
8. Scans positive periodic-cosh fit windows.
9. Compares uncorrelated and correlated fits.

The correlated objective is

    chi^2
        =
        [C - f]^T Sigma^{-1} [C - f].

This is exploratory finite-lattice spectroscopy.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.covariance_fits import (
    bootstrap_connected_correlators,
    estimate_covariance,
    fit_correlated_periodic_cosh,
    regularized_inverse_covariance,
    shrink_covariance_to_diagonal,
)
from ymlab.glueball import (
    ensemble_connected_correlator,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.operator_basis import (
    create_smearing_basis,
    measure_operator_basis,
)
from ymlab.spectroscopy import (
    fit_periodic_cosh,
    periodic_cosh_correlator,
)


def collect_operator_ensemble(
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    number_of_configurations,
    sweeps_between_measurements,
    basis,
    selected_operator_index,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    thermal_acceptance = []

    print("Thermalizing lattice...")

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
    measurement_acceptance = []

    print(
        "Collecting selected smeared operator..."
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

        basis_measurement = measure_operator_basis(
            lattice=lattice,
            basis=basis,
        )

        measurements.append(
            basis_measurement[
                selected_operator_index
            ]
        )

        if (
            configuration_index == 0
            or (
                configuration_index + 1
            ) % 25 == 0
        ):
            print(
                "  collected configuration "
                f"{configuration_index + 1:03d}/"
                f"{number_of_configurations:03d}"
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


def scan_correlated_windows(
    correlation,
    covariance,
    shrinkage,
    relative_cutoff,
):
    temporal_extent = len(
        correlation
    )

    candidates = []

    maximum_stop = (
        temporal_extent // 2
        + 1
    )

    for fit_start in [
        1,
        2,
    ]:
        for fit_stop in range(
            fit_start + 3,
            maximum_stop + 1,
        ):
            try:
                fit = fit_correlated_periodic_cosh(
                    correlation=correlation,
                    covariance=covariance,
                    fit_start=fit_start,
                    fit_stop=fit_stop,
                    shrinkage=shrinkage,
                    relative_cutoff=relative_cutoff,
                )
            except ValueError:
                continue

            if not fit.success:
                continue

            score = (
                fit.reduced_chi_squared
                if np.isfinite(
                    fit.reduced_chi_squared
                )
                else np.inf
            )

            candidates.append(
                (
                    abs(
                        score - 1.0
                    ),
                    -(
                        fit.fit_stop
                        - fit.fit_start
                    ),
                    fit,
                )
            )

    if len(candidates) == 0:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return candidates[0][2]


def main() -> None:
    shape = (12, 5, 5)
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 90
    number_of_configurations = 220
    sweeps_between_measurements = 3
    n_bootstrap = 600
    shrinkage = 0.10
    relative_cutoff = 1e-8
    seed = 2026

    basis = create_smearing_basis(
        smearing_levels=[
            0,
            2,
            5,
            10,
        ],
        alpha=0.5,
        time_direction=0,
    )

    selected_operator_index = 2
    selected_operator_name = basis.names[
        selected_operator_index
    ]

    print(
        "Covariance-Aware Correlated Spectroscopy"
    )
    print("=" * 108)
    print(
        f"Lattice shape:                  {shape}"
    )
    print(
        f"Beta:                           {beta}"
    )
    print(
        f"Proposal epsilon:               {epsilon}"
    )
    print(
        f"Thermalization sweeps:          "
        f"{thermalization_sweeps}"
    )
    print(
        f"Measured configurations:        "
        f"{number_of_configurations}"
    )
    print(
        "Sweeps between measurements:    "
        f"{sweeps_between_measurements}"
    )
    print(
        f"Selected operator:              "
        f"{selected_operator_name}"
    )
    print(
        f"Bootstrap replicates:           "
        f"{n_bootstrap}"
    )
    print(
        f"Covariance shrinkage:           "
        f"{shrinkage:.4f}"
    )
    print(
        f"Inverse relative cutoff:        "
        f"{relative_cutoff:.3e}"
    )
    print()

    (
        ensemble,
        thermal_acceptance,
        measurement_acceptance,
    ) = collect_operator_ensemble(
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
        selected_operator_index=(
            selected_operator_index
        ),
        seed=seed,
    )

    correlator = (
        ensemble_connected_correlator(
            ensemble
        )
    )

    print()
    print(
        "Generating configuration-bootstrap "
        "correlator replicates..."
    )

    bootstrap_replicates = (
        bootstrap_connected_correlators(
            operator_series_ensemble=ensemble,
            n_bootstrap=n_bootstrap,
            seed=seed + 5000,
        )
    )

    covariance_estimate = (
        estimate_covariance(
            bootstrap_replicates
        )
    )

    shrunk_covariance = (
        shrink_covariance_to_diagonal(
            covariance=(
                covariance_estimate.covariance
            ),
            shrinkage=shrinkage,
        )
    )

    inverse_result = (
        regularized_inverse_covariance(
            covariance=shrunk_covariance,
            relative_cutoff=relative_cutoff,
            absolute_cutoff=1e-18,
        )
    )

    correlated_fit = scan_correlated_windows(
        correlation=correlator.correlation,
        covariance=(
            covariance_estimate.covariance
        ),
        shrinkage=shrinkage,
        relative_cutoff=relative_cutoff,
    )

    uncorrelated_fit = None

    maximum_stop = (
        correlator.temporal_extent // 2
        + 1
    )

    uncorrelated_candidates = []

    for fit_start in [
        1,
        2,
    ]:
        for fit_stop in range(
            fit_start + 2,
            maximum_stop + 1,
        ):
            try:
                candidate = fit_periodic_cosh(
                    correlation=(
                        correlator.correlation
                    ),
                    fit_start=fit_start,
                    fit_stop=fit_stop,
                )
            except ValueError:
                continue

            if not candidate.success:
                continue

            residuals = (
                candidate.fit_values
                - candidate.model_values
            )

            mse = float(
                np.mean(
                    residuals ** 2
                )
            )

            uncorrelated_candidates.append(
                (
                    mse,
                    -len(
                        candidate.fit_values
                    ),
                    candidate,
                )
            )

    if len(
        uncorrelated_candidates
    ) > 0:
        uncorrelated_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        uncorrelated_fit = (
            uncorrelated_candidates[0][2]
        )

    print()
    print("Simulation summary")
    print("-" * 108)
    print(
        f"Operator ensemble shape:        "
        f"{ensemble.shape}"
    )
    print(
        f"Mean thermal acceptance:        "
        f"{thermal_acceptance:.8f}"
    )
    print(
        f"Mean measurement acceptance:    "
        f"{measurement_acceptance:.8f}"
    )
    print(
        f"Mean scalar operator:            "
        f"{correlator.mean_operator:.12e}"
    )
    print()

    print("Correlator covariance diagnostics")
    print("-" * 108)
    print(
        f"Covariance numerical rank:       "
        f"{covariance_estimate.numerical_rank}"
    )
    print(
        f"Raw covariance condition number: "
        f"{covariance_estimate.condition_number:.12e}"
    )
    print(
        f"Regularized inverse rank:        "
        f"{inverse_result.retained_rank}"
    )
    print(
        f"Inverse covariance cutoff:       "
        f"{inverse_result.eigenvalue_cutoff:.12e}"
    )
    print(
        f"Regularized condition number:    "
        f"{inverse_result.condition_number:.12e}"
    )
    print(
        "Covariance/inverse projection error: "
        f"{inverse_result.identity_projection_error:.12e}"
    )
    print()

    print("Covariance eigenvalues")
    print("-" * 108)

    for index, eigenvalue in enumerate(
        covariance_estimate.eigenvalues
    ):
        print(
            f"eigenvalue {index:>2d}: "
            f"{eigenvalue:>24.12e}"
        )

    print()
    print("Fit comparison")
    print("-" * 108)

    if uncorrelated_fit is None:
        print(
            "Uncorrelated fit: no valid positive window."
        )
    else:
        print(
            "Uncorrelated fit"
        )
        print(
            f"  fit window:                  "
            f"[{uncorrelated_fit.fit_start}, "
            f"{uncorrelated_fit.fit_stop})"
        )
        print(
            f"  fitted mass:                 "
            f"{uncorrelated_fit.mass:.12f}"
        )
        print(
            f"  covariance mass error:        "
            f"{uncorrelated_fit.mass_error:.12f}"
        )

    print()

    if correlated_fit is None:
        print(
            "Correlated fit: no valid positive window."
        )
        print(
            "No covariance-aware mass estimate "
            "will be fabricated."
        )
    else:
        print(
            "Correlated fit"
        )
        print(
            f"  fit window:                  "
            f"[{correlated_fit.fit_start}, "
            f"{correlated_fit.fit_stop})"
        )
        print(
            f"  fitted amplitude:            "
            f"{correlated_fit.amplitude:.12e}"
        )
        print(
            f"  amplitude error:             "
            f"{correlated_fit.amplitude_error:.12e}"
        )
        print(
            f"  fitted mass:                 "
            f"{correlated_fit.mass:.12f}"
        )
        print(
            f"  correlated mass error:       "
            f"{correlated_fit.mass_error:.12f}"
        )
        print(
            f"  chi-squared:                 "
            f"{correlated_fit.chi_squared:.12f}"
        )
        print(
            f"  effective degrees of freedom:"
            f" {correlated_fit.effective_degrees_of_freedom}"
        )
        print(
            f"  reduced chi-squared:         "
            f"{correlated_fit.reduced_chi_squared:.12f}"
        )
        print(
            f"  fit covariance rank:         "
            f"{correlated_fit.covariance_rank}"
        )
        print(
            f"  fit covariance condition:    "
            f"{correlated_fit.covariance_condition_number:.12e}"
        )
        print(
            f"  optimizer:                   "
            f"{correlated_fit.optimizer_message}"
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

    covariance_path = (
        data_dir
        / "correlator_covariance_matrix.csv"
    )

    correlation_path = (
        data_dir
        / "correlator_time_correlation_matrix.csv"
    )

    eigenvalue_path = (
        data_dir
        / "correlator_covariance_eigenvalues.csv"
    )

    np.savetxt(
        covariance_path,
        covariance_estimate.covariance,
        delimiter=",",
    )

    np.savetxt(
        correlation_path,
        covariance_estimate.correlation,
        delimiter=",",
    )

    with eigenvalue_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "index",
                "covariance_eigenvalue",
            ]
        )

        for index, eigenvalue in enumerate(
            covariance_estimate.eigenvalues
        ):
            writer.writerow(
                [
                    index,
                    float(
                        eigenvalue
                    ),
                ]
            )

    fit_path = (
        data_dir
        / "correlated_spectroscopy_fit.csv"
    )

    fit_rows = []

    if uncorrelated_fit is not None:
        fit_rows.append(
            {
                "fit_type": "uncorrelated",
                "fit_start": (
                    uncorrelated_fit.fit_start
                ),
                "fit_stop": (
                    uncorrelated_fit.fit_stop
                ),
                "mass": (
                    uncorrelated_fit.mass
                ),
                "mass_error": (
                    uncorrelated_fit.mass_error
                ),
                "chi_squared": "",
                "degrees_of_freedom": "",
                "reduced_chi_squared": "",
            }
        )

    if correlated_fit is not None:
        fit_rows.append(
            {
                "fit_type": "correlated",
                "fit_start": (
                    correlated_fit.fit_start
                ),
                "fit_stop": (
                    correlated_fit.fit_stop
                ),
                "mass": (
                    correlated_fit.mass
                ),
                "mass_error": (
                    correlated_fit.mass_error
                ),
                "chi_squared": (
                    correlated_fit.chi_squared
                ),
                "degrees_of_freedom": (
                    correlated_fit.effective_degrees_of_freedom
                ),
                "reduced_chi_squared": (
                    correlated_fit.reduced_chi_squared
                ),
            }
        )

    if len(
        fit_rows
    ) > 0:
        with fit_path.open(
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    fit_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                fit_rows
            )

    plt.figure()
    plt.imshow(
        covariance_estimate.correlation,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.colorbar(
        label="Correlation coefficient"
    )
    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Euclidean time lag"
    )
    plt.title(
        "Bootstrap Correlator Time-Time Correlation Matrix"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "correlator_time_correlation_matrix.png",
        dpi=200,
    )

    plt.figure()
    plt.semilogy(
        np.arange(
            len(
                covariance_estimate.eigenvalues
            )
        ),
        np.maximum(
            covariance_estimate.eigenvalues,
            np.finfo(float).tiny,
        ),
        marker="o",
    )
    plt.xlabel(
        "Covariance eigenvalue index"
    )
    plt.ylabel(
        "Covariance eigenvalue"
    )
    plt.title(
        "Correlator Covariance Eigenspectrum"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "correlator_covariance_eigenspectrum.png",
        dpi=200,
    )

    plt.figure()

    lags = correlator.lags

    errors = (
        covariance_estimate.standard_deviations
    )

    plt.errorbar(
        lags,
        correlator.correlation,
        yerr=errors,
        marker="o",
        linestyle="none",
        label="Connected correlator",
    )

    if correlated_fit is not None:
        dense_lags = np.linspace(
            correlated_fit.fit_start,
            correlated_fit.fit_stop - 1,
            300,
        )

        dense_model = periodic_cosh_correlator(
            lag=dense_lags,
            amplitude=(
                correlated_fit.amplitude
            ),
            mass=correlated_fit.mass,
            temporal_extent=(
                correlator.temporal_extent
            ),
        )

        plt.plot(
            dense_lags,
            dense_model,
            label="Correlated periodic-cosh fit",
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Connected correlation"
    )
    plt.title(
        "Covariance-Aware Scalar Spectroscopy Fit"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "correlated_spectroscopy_fit.png",
        dpi=200,
    )

    print()
    print("Saved data:")
    print(covariance_path)
    print(correlation_path)
    print(eigenvalue_path)

    if len(
        fit_rows
    ) > 0:
        print(fit_path)

    print()
    print("Saved figures:")
    print(
        "results/figures/"
        "correlator_time_correlation_matrix.png"
    )
    print(
        "results/figures/"
        "correlator_covariance_eigenspectrum.png"
    )
    print(
        "results/figures/"
        "correlated_spectroscopy_fit.png"
    )


if __name__ == "__main__":
    main()
