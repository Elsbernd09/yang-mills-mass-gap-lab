"""
Covariance-aware fits of matched GEVP principal correlators.

The experiment combines:

1. multi-operator scalar measurements,
2. regularized GEVP spectroscopy,
3. circular block bootstrap,
4. metric-overlap state matching,
5. principal-correlator time-time covariance,
6. covariance shrinkage,
7. regularized inverse covariance,
8. normalized periodic principal-state fitting,
9. bootstrap mass refitting.

Each variational state is treated independently after central/bootstrap state
matching.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.gevp_bootstrap import (
    bootstrap_gevp,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import (
    metropolis_sweep,
)
from ymlab.operator_basis import (
    create_smearing_basis,
    measure_operator_basis,
)
from ymlab.principal_fits import (
    bootstrap_refit_principal_mass,
    normalized_periodic_principal_correlator,
    principal_state_covariance,
    scan_principal_fit_windows,
)


def collect_operator_ensemble(
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    number_of_configurations,
    sweeps_between_measurements,
    basis,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    thermal_acceptance = []
    measurement_acceptance = []
    measurements = []

    print(
        "Thermalizing lattice..."
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

    print(
        "Collecting multi-operator ensemble..."
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
            measure_operator_basis(
                lattice=lattice,
                basis=basis,
            )
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


def main() -> None:
    shape = (
        12,
        5,
        5,
    )

    beta = 2.0
    epsilon = 0.18

    thermalization_sweeps = 100
    number_of_configurations = 220
    sweeps_between_measurements = 3

    reference_time = 1
    relative_cutoff = 1e-8
    absolute_cutoff = 1e-14

    n_bootstrap = 450
    block_size = 5

    shrinkage = 0.10
    covariance_relative_cutoff = 1e-8

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

    print(
        "Correlated Principal-State Spectroscopy"
    )
    print("=" * 116)

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
        f"Measured configurations:          "
        f"{number_of_configurations}"
    )

    print(
        f"Operator basis:                   "
        f"{basis.names}"
    )

    print(
        f"GEVP reference time:              "
        f"{reference_time}"
    )

    print(
        f"Bootstrap replicates:             "
        f"{n_bootstrap}"
    )

    print(
        f"Bootstrap block size:             "
        f"{block_size}"
    )

    print(
        f"Principal covariance shrinkage:   "
        f"{shrinkage}"
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
        seed=seed,
    )

    print()

    print(
        "Bootstrapping and state-matching "
        "the full GEVP pipeline..."
    )

    bootstrap = bootstrap_gevp(
        operator_ensemble=ensemble,
        n_bootstrap=n_bootstrap,
        reference_time=reference_time,
        relative_cutoff=relative_cutoff,
        absolute_cutoff=absolute_cutoff,
        mode="circular_block",
        block_size=block_size,
        seed=seed + 10000,
    )

    central = bootstrap.central_result

    print()

    print(
        "Simulation and bootstrap summary"
    )

    print("-" * 116)

    print(
        f"Operator ensemble shape:          "
        f"{ensemble.shape}"
    )

    print(
        f"Mean thermal acceptance:          "
        f"{thermal_acceptance:.8f}"
    )

    print(
        f"Mean measurement acceptance:      "
        f"{measurement_acceptance:.8f}"
    )

    print(
        f"Central retained rank:            "
        f"{bootstrap.central_rank}"
    )

    print(
        f"Accepted bootstrap replicates:    "
        f"{bootstrap.accepted_replicates}/"
        f"{bootstrap.requested_replicates}"
    )

    print(
        f"Rank-rejected replicates:         "
        f"{bootstrap.rejected_rank_replicates}"
    )

    print(
        f"Numerically rejected replicates:  "
        f"{bootstrap.rejected_numerical_replicates}"
    )

    print()

    state_results = []

    for state in range(
        bootstrap.central_rank
    ):
        print(
            "=" * 116
        )

        print(
            f"Variational state {state}"
        )

        print(
            "=" * 116
        )

        covariance_result = (
            principal_state_covariance(
                principal_samples=(
                    bootstrap.principal_samples
                ),
                state=state,
            )
        )

        central_principal = (
            central.principal_correlators[
                :,
                state,
            ]
        )

        covariance = (
            covariance_result
            .estimate
            .covariance
        )

        fits = scan_principal_fit_windows(
            principal_correlator=(
                central_principal
            ),
            covariance=covariance,
            reference_time=reference_time,
            shrinkage=shrinkage,
            relative_cutoff=(
                covariance_relative_cutoff
            ),
            minimum_points=3,
        )

        print(
            f"Finite covariance replicates:     "
            f"{covariance_result.finite_replicates}"
        )

        print(
            f"Raw covariance rank:              "
            f"{covariance_result.estimate.numerical_rank}"
        )

        print(
            f"Raw covariance condition number:  "
            f"{covariance_result.estimate.condition_number:.12e}"
        )

        if len(
            fits
        ) == 0:
            print(
                "No valid positive correlated fit window."
            )

            print(
                "No principal-state mass estimate "
                "will be fabricated."
            )

            state_results.append(
                {
                    "state": state,
                    "fit_available": False,
                    "fit_start": "",
                    "fit_stop": "",
                    "central_mass": "",
                    "local_mass_error": "",
                    "chi_squared": "",
                    "effective_dof": "",
                    "reduced_chi_squared": "",
                    "bootstrap_accepted": "",
                    "bootstrap_rejected": "",
                    "bootstrap_mean_mass": "",
                    "bootstrap_standard_error": "",
                    "bootstrap_lower_95": "",
                    "bootstrap_median_mass": "",
                    "bootstrap_upper_95": "",
                }
            )

            print()

            continue

        best_fit = fits[0]

        print(
            f"Selected fit window:              "
            f"[{best_fit.fit_start}, "
            f"{best_fit.fit_stop})"
        )

        print(
            f"Correlated central mass:          "
            f"{best_fit.mass:.12f}"
        )

        print(
            f"Local curvature mass error:       "
            f"{best_fit.mass_error:.12f}"
        )

        print(
            f"Chi-squared:                      "
            f"{best_fit.chi_squared:.12f}"
        )

        print(
            f"Effective degrees of freedom:     "
            f"{best_fit.effective_degrees_of_freedom}"
        )

        print(
            f"Reduced chi-squared:              "
            f"{best_fit.reduced_chi_squared:.12f}"
        )

        print(
            f"Fit covariance rank:              "
            f"{best_fit.covariance_rank}"
        )

        print(
            f"Fit covariance condition number:  "
            f"{best_fit.covariance_condition_number:.12e}"
        )

        state_samples = (
            bootstrap.principal_samples[
                :,
                :,
                state,
            ]
        )

        try:
            refit_summary = (
                bootstrap_refit_principal_mass(
                    principal_samples=(
                        state_samples
                    ),
                    central_fit=best_fit,
                    covariance=covariance,
                    shrinkage=shrinkage,
                    relative_cutoff=(
                        covariance_relative_cutoff
                    ),
                )
            )
        except ValueError as error:
            print(
                "Bootstrap principal-state refit "
                f"failed: {error}"
            )

            refit_summary = None

        if refit_summary is not None:
            print()

            print(
                "Bootstrap fixed-window mass refit"
            )

            print(
                f"  accepted fits:                  "
                f"{refit_summary.accepted_fits}/"
                f"{refit_summary.requested_fits}"
            )

            print(
                f"  rejected fits:                  "
                f"{refit_summary.rejected_fits}"
            )

            print(
                f"  bootstrap mean mass:            "
                f"{refit_summary.mean_mass:.12f}"
            )

            print(
                f"  bootstrap standard error:       "
                f"{refit_summary.standard_error:.12f}"
            )

            print(
                f"  bootstrap 95% interval:         "
                f"[{refit_summary.lower_95:.12f}, "
                f"{refit_summary.upper_95:.12f}]"
            )

            print(
                f"  bootstrap median mass:          "
                f"{refit_summary.median_mass:.12f}"
            )

        state_results.append(
            {
                "state": state,
                "fit_available": True,
                "fit_start": (
                    best_fit.fit_start
                ),
                "fit_stop": (
                    best_fit.fit_stop
                ),
                "central_mass": (
                    best_fit.mass
                ),
                "local_mass_error": (
                    best_fit.mass_error
                ),
                "chi_squared": (
                    best_fit.chi_squared
                ),
                "effective_dof": (
                    best_fit.effective_degrees_of_freedom
                ),
                "reduced_chi_squared": (
                    best_fit.reduced_chi_squared
                ),
                "bootstrap_accepted": (
                    refit_summary.accepted_fits
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_rejected": (
                    refit_summary.rejected_fits
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_mean_mass": (
                    refit_summary.mean_mass
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_standard_error": (
                    refit_summary.standard_error
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_lower_95": (
                    refit_summary.lower_95
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_median_mass": (
                    refit_summary.median_mass
                    if refit_summary is not None
                    else ""
                ),
                "bootstrap_upper_95": (
                    refit_summary.upper_95
                    if refit_summary is not None
                    else ""
                ),
            }
        )

        print()

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

    summary_path = (
        data_dir
        / "correlated_principal_state_fits.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                state_results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            state_results
        )

    for state in range(
        bootstrap.central_rank
    ):
        row = state_results[
            state
        ]

        if not row[
            "fit_available"
        ]:
            continue

        covariance_result = (
            principal_state_covariance(
                principal_samples=(
                    bootstrap.principal_samples
                ),
                state=state,
            )
        )

        fits = scan_principal_fit_windows(
            principal_correlator=(
                central.principal_correlators[
                    :,
                    state,
                ]
            ),
            covariance=(
                covariance_result
                .estimate
                .covariance
            ),
            reference_time=reference_time,
            shrinkage=shrinkage,
            relative_cutoff=(
                covariance_relative_cutoff
            ),
            minimum_points=3,
        )

        best_fit = fits[0]

        plt.figure()

        principal_samples = (
            bootstrap.principal_samples[
                :,
                :,
                state,
            ]
        )

        mean = np.mean(
            principal_samples,
            axis=0,
        )

        standard_error = np.std(
            principal_samples,
            axis=0,
            ddof=1,
        )

        plt.errorbar(
            central.lags,
            mean,
            yerr=standard_error,
            marker="o",
            linestyle="none",
            label="Bootstrap principal correlator",
        )

        dense_lags = np.linspace(
            best_fit.fit_start,
            best_fit.fit_stop - 1,
            300,
        )

        dense_model = (
            normalized_periodic_principal_correlator(
                lag=dense_lags,
                mass=best_fit.mass,
                temporal_extent=(
                    best_fit.temporal_extent
                ),
                reference_time=(
                    best_fit.reference_time
                ),
            )
        )

        plt.plot(
            dense_lags,
            dense_model,
            label="Correlated normalized periodic fit",
        )

        plt.xlabel(
            "Euclidean time lag"
        )

        plt.ylabel(
            "Principal correlator"
        )

        plt.title(
            "Correlated Principal-State Fit "
            f"for Variational State {state}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            figures_dir
            / (
                "correlated_principal_fit_state_"
                f"{state}.png"
            ),
            dpi=200,
        )

    available_rows = [
        row
        for row in state_results
        if row[
            "fit_available"
        ]
    ]

    if len(
        available_rows
    ) > 0:
        states = np.asarray(
            [
                row[
                    "state"
                ]
                for row in available_rows
            ],
            dtype=int,
        )

        masses = np.asarray(
            [
                row[
                    "bootstrap_mean_mass"
                ]
                if row[
                    "bootstrap_mean_mass"
                ] != ""
                else row[
                    "central_mass"
                ]
                for row in available_rows
            ],
            dtype=float,
        )

        errors = np.asarray(
            [
                row[
                    "bootstrap_standard_error"
                ]
                if row[
                    "bootstrap_standard_error"
                ] != ""
                else row[
                    "local_mass_error"
                ]
                for row in available_rows
            ],
            dtype=float,
        )

        plt.figure()

        plt.errorbar(
            states,
            masses,
            yerr=errors,
            marker="o",
            linestyle="none",
        )

        plt.xlabel(
            "Matched variational state"
        )

        plt.ylabel(
            "Lattice mass estimate"
        )

        plt.title(
            "Correlated Variational-State Mass Estimates"
        )

        plt.tight_layout()

        plt.savefig(
            figures_dir
            / "correlated_principal_state_masses.png",
            dpi=200,
        )

    print(
        f"Saved fit summary: {summary_path}"
    )

    print()

    print(
        "Saved available principal-state fit figures "
        "in results/figures/."
    )


if __name__ == "__main__":
    main()
