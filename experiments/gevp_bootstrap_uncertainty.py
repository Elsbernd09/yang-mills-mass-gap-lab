"""
Bootstrap uncertainty analysis for the complete GEVP spectroscopy pipeline.

The experiment constructs a multi-smearing scalar operator ensemble and then:

1. computes the central connected correlator matrix,
2. solves the central regularized GEVP,
3. performs circular block bootstrap resampling of configurations,
4. rebuilds C_ij(t) independently for every replicate,
5. resolves the reference metric and GEVP,
6. rejects fixed-rank-incompatible replicates,
7. matches bootstrap variational states to central states,
8. constructs principal-correlator uncertainty bands,
9. constructs variational effective-mass uncertainty distributions,
10. measures bootstrap state-overlap stability.

This is finite-lattice variational uncertainty analysis.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.gevp_bootstrap import (
    bootstrap_gevp,
    summarize_bootstrap_distribution,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.operator_basis import (
    create_smearing_basis,
    measure_operator_basis,
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
    shape = (10, 5, 5)
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 90
    number_of_configurations = 180
    sweeps_between_measurements = 3

    reference_time = 1
    relative_cutoff = 1e-8
    absolute_cutoff = 1e-14

    n_bootstrap = 400
    block_size = 5
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
        "Full GEVP Bootstrap and Variational-State Uncertainty"
    )
    print("=" * 112)
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
        f"Operator basis:                   "
        f"{basis.names}"
    )
    print(
        f"GEVP reference time:              "
        f"{reference_time}"
    )
    print(
        f"GEVP relative cutoff:             "
        f"{relative_cutoff:.3e}"
    )
    print(
        f"Bootstrap mode:                   "
        "circular_block"
    )
    print(
        f"Bootstrap block size:             "
        f"{block_size}"
    )
    print(
        f"Requested bootstrap replicates:   "
        f"{n_bootstrap}"
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
        "Bootstrapping complete correlator-matrix "
        "and GEVP pipeline..."
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

    principal_summary = (
        summarize_bootstrap_distribution(
            bootstrap.principal_samples,
            confidence_level=0.95,
        )
    )

    log_mass_summary = (
        summarize_bootstrap_distribution(
            bootstrap.log_mass_samples,
            confidence_level=0.95,
        )
    )

    arccosh_mass_summary = (
        summarize_bootstrap_distribution(
            bootstrap.arccosh_mass_samples,
            confidence_level=0.95,
        )
    )

    overlap_summary = (
        summarize_bootstrap_distribution(
            bootstrap.overlap_samples,
            confidence_level=0.95,
        )
    )

    central = bootstrap.central_result

    acceptance_fraction = (
        bootstrap.accepted_replicates
        / bootstrap.requested_replicates
    )

    print()
    print("Simulation summary")
    print("-" * 112)
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
        f"Central GEVP retained rank:       "
        f"{bootstrap.central_rank}"
    )
    print()

    print("Bootstrap acceptance diagnostics")
    print("-" * 112)
    print(
        f"Requested replicates:             "
        f"{bootstrap.requested_replicates}"
    )
    print(
        f"Accepted replicates:              "
        f"{bootstrap.accepted_replicates}"
    )
    print(
        f"Rejected rank replicates:         "
        f"{bootstrap.rejected_rank_replicates}"
    )
    print(
        f"Rejected numerical replicates:    "
        f"{bootstrap.rejected_numerical_replicates}"
    )
    print(
        f"Acceptance fraction:              "
        f"{acceptance_fraction:.8f}"
    )
    print(
        f"Bootstrap mode:                   "
        f"{bootstrap.bootstrap_mode}"
    )
    print(
        f"Block size:                       "
        f"{bootstrap.block_size}"
    )
    print()

    print("Principal-correlator bootstrap summary")
    print("-" * 112)

    for state in range(
        bootstrap.central_rank
    ):
        print()
        print(f"State {state}")
        print(
            f"{'lag':>6}"
            f"{'central':>18}"
            f"{'bootstrap mean':>20}"
            f"{'std error':>18}"
            f"{'95% lower':>18}"
            f"{'95% upper':>18}"
            f"{'finite N':>12}"
        )

        for lag in central.lags:
            print(
                f"{lag:>6d}"
                f"{central.principal_correlators[lag, state]:>18.8e}"
                f"{principal_summary.mean[lag, state]:>20.8e}"
                f"{principal_summary.standard_error[lag, state]:>18.8e}"
                f"{principal_summary.lower[lag, state]:>18.8e}"
                f"{principal_summary.upper[lag, state]:>18.8e}"
                f"{principal_summary.finite_counts[lag, state]:>12d}"
            )

    print()
    print("Bootstrap state-overlap stability")
    print("-" * 112)

    for state in range(
        bootstrap.central_rank
    ):
        values = bootstrap.overlap_samples[
            :,
            :,
            state,
        ]

        print(
            f"State {state}: "
            f"mean overlap = "
            f"{np.mean(values):.8f}, "
            f"minimum overlap = "
            f"{np.min(values):.8f}, "
            f"median overlap = "
            f"{np.median(values):.8f}"
        )

    print()
    print("Log-ratio variational mass uncertainty")
    print("-" * 112)

    for state in range(
        bootstrap.central_rank
    ):
        print()
        print(f"State {state}")
        print(
            f"{'lag':>6}"
            f"{'mean mass':>18}"
            f"{'std error':>18}"
            f"{'95% lower':>18}"
            f"{'95% upper':>18}"
            f"{'finite N':>12}"
        )

        for lag in central.lags:
            count = (
                log_mass_summary.finite_counts[
                    lag,
                    state,
                ]
            )

            if count == 0:
                continue

            print(
                f"{lag:>6d}"
                f"{log_mass_summary.mean[lag, state]:>18.8e}"
                f"{log_mass_summary.standard_error[lag, state]:>18.8e}"
                f"{log_mass_summary.lower[lag, state]:>18.8e}"
                f"{log_mass_summary.upper[lag, state]:>18.8e}"
                f"{count:>12d}"
            )

    print()
    print("Arccosh variational mass uncertainty")
    print("-" * 112)

    for state in range(
        bootstrap.central_rank
    ):
        print()
        print(f"State {state}")
        print(
            f"{'lag':>6}"
            f"{'mean mass':>18}"
            f"{'std error':>18}"
            f"{'95% lower':>18}"
            f"{'95% upper':>18}"
            f"{'finite N':>12}"
        )

        for lag in central.lags:
            count = (
                arccosh_mass_summary.finite_counts[
                    lag,
                    state,
                ]
            )

            if count == 0:
                continue

            print(
                f"{lag:>6d}"
                f"{arccosh_mass_summary.mean[lag, state]:>18.8e}"
                f"{arccosh_mass_summary.standard_error[lag, state]:>18.8e}"
                f"{arccosh_mass_summary.lower[lag, state]:>18.8e}"
                f"{arccosh_mass_summary.upper[lag, state]:>18.8e}"
                f"{count:>12d}"
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

    principal_path = (
        data_dir
        / "gevp_bootstrap_principal_summary.csv"
    )

    principal_rows = []

    for lag in central.lags:
        for state in range(
            bootstrap.central_rank
        ):
            principal_rows.append(
                {
                    "lag": int(lag),
                    "state": state,
                    "central": float(
                        central.principal_correlators[
                            lag,
                            state,
                        ]
                    ),
                    "bootstrap_mean": float(
                        principal_summary.mean[
                            lag,
                            state,
                        ]
                    ),
                    "standard_error": float(
                        principal_summary.standard_error[
                            lag,
                            state,
                        ]
                    ),
                    "lower_95": float(
                        principal_summary.lower[
                            lag,
                            state,
                        ]
                    ),
                    "median": float(
                        principal_summary.median[
                            lag,
                            state,
                        ]
                    ),
                    "upper_95": float(
                        principal_summary.upper[
                            lag,
                            state,
                        ]
                    ),
                    "finite_count": int(
                        principal_summary.finite_counts[
                            lag,
                            state,
                        ]
                    ),
                }
            )

    with principal_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                principal_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            principal_rows
        )

    mass_path = (
        data_dir
        / "gevp_bootstrap_mass_summary.csv"
    )

    mass_rows = []

    for estimator_name, summary in [
        (
            "log_ratio",
            log_mass_summary,
        ),
        (
            "arccosh",
            arccosh_mass_summary,
        ),
    ]:
        for lag in central.lags:
            for state in range(
                bootstrap.central_rank
            ):
                count = int(
                    summary.finite_counts[
                        lag,
                        state,
                    ]
                )

                mass_rows.append(
                    {
                        "estimator": estimator_name,
                        "lag": int(lag),
                        "state": state,
                        "mean": (
                            float(
                                summary.mean[
                                    lag,
                                    state,
                                ]
                            )
                            if count > 0
                            else ""
                        ),
                        "standard_error": (
                            float(
                                summary.standard_error[
                                    lag,
                                    state,
                                ]
                            )
                            if count > 1
                            else ""
                        ),
                        "lower_95": (
                            float(
                                summary.lower[
                                    lag,
                                    state,
                                ]
                            )
                            if count > 0
                            else ""
                        ),
                        "median": (
                            float(
                                summary.median[
                                    lag,
                                    state,
                                ]
                            )
                            if count > 0
                            else ""
                        ),
                        "upper_95": (
                            float(
                                summary.upper[
                                    lag,
                                    state,
                                ]
                            )
                            if count > 0
                            else ""
                        ),
                        "finite_count": count,
                    }
                )

    with mass_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                mass_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            mass_rows
        )

    overlap_path = (
        data_dir
        / "gevp_bootstrap_state_overlap_summary.csv"
    )

    overlap_rows = []

    for lag in central.lags:
        for state in range(
            bootstrap.central_rank
        ):
            overlap_rows.append(
                {
                    "lag": int(lag),
                    "state": state,
                    "mean_overlap": float(
                        overlap_summary.mean[
                            lag,
                            state,
                        ]
                    ),
                    "standard_error": float(
                        overlap_summary.standard_error[
                            lag,
                            state,
                        ]
                    ),
                    "lower_95": float(
                        overlap_summary.lower[
                            lag,
                            state,
                        ]
                    ),
                    "median": float(
                        overlap_summary.median[
                            lag,
                            state,
                        ]
                    ),
                    "upper_95": float(
                        overlap_summary.upper[
                            lag,
                            state,
                        ]
                    ),
                }
            )

    with overlap_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                overlap_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            overlap_rows
        )

    plt.figure()

    for state in range(
        bootstrap.central_rank
    ):
        plt.plot(
            central.lags,
            central.principal_correlators[
                :,
                state,
            ],
            marker="o",
            label=f"central state {state}",
        )

        plt.fill_between(
            central.lags,
            principal_summary.lower[
                :,
                state,
            ],
            principal_summary.upper[
                :,
                state,
            ],
            alpha=0.2,
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Principal correlator"
    )
    plt.title(
        "GEVP Principal Correlators with 95% Bootstrap Intervals"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_bootstrap_principal_intervals.png",
        dpi=200,
    )

    plt.figure()

    for state in range(
        bootstrap.central_rank
    ):
        finite = (
            log_mass_summary.finite_counts[
                :,
                state,
            ] > 1
        )

        plt.errorbar(
            central.lags[
                finite
            ],
            log_mass_summary.mean[
                finite,
                state,
            ],
            yerr=(
                log_mass_summary.standard_error[
                    finite,
                    state,
                ]
            ),
            marker="o",
            linestyle="none",
            label=f"state {state}",
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Log-ratio variational effective mass"
    )
    plt.title(
        "Bootstrap GEVP Effective-Mass Uncertainty"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_bootstrap_log_mass_uncertainty.png",
        dpi=200,
    )

    plt.figure()

    for state in range(
        bootstrap.central_rank
    ):
        plt.plot(
            central.lags,
            overlap_summary.mean[
                :,
                state,
            ],
            marker="o",
            label=f"state {state}",
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Mean central/bootstrap metric overlap"
    )
    plt.title(
        "Variational State-Matching Stability"
    )
    plt.ylim(
        0.0,
        1.05,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_bootstrap_state_overlap.png",
        dpi=200,
    )

    print()
    print("Saved data:")
    print(principal_path)
    print(mass_path)
    print(overlap_path)
    print()
    print("Saved figures:")
    print(
        "results/figures/"
        "gevp_bootstrap_principal_intervals.png"
    )
    print(
        "results/figures/"
        "gevp_bootstrap_log_mass_uncertainty.png"
    )
    print(
        "results/figures/"
        "gevp_bootstrap_state_overlap.png"
    )


if __name__ == "__main__":
    main()
