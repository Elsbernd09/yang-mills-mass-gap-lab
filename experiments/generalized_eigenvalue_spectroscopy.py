"""
Regularized generalized-eigenvalue spectroscopy experiment.

The experiment constructs a multi-operator scalar basis from several spatial
smearing levels and estimates the connected Euclidean correlator matrix.

It then solves

    C(t) v_n = lambda_n(t, t0) C(t0) v_n

using a regularized whitening procedure.

The analysis reports:

1. Reference-metric spectrum.
2. Retained numerical rank.
3. Reference metric condition number.
4. Principal correlators.
5. Log-ratio variational effective masses.
6. Arccosh variational effective masses.
7. Reference-time normalization errors.
8. Generalized eigenvector operator components.

This is exploratory finite-lattice variational spectroscopy.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.operator_basis import (
    create_smearing_basis,
    ensemble_correlator_matrix,
    measure_operator_basis,
    symmetrize_correlator_matrix,
)
from ymlab.variational import (
    metric_orthonormality_error,
    principal_arccosh_effective_masses,
    principal_log_effective_masses,
    solve_regularized_gevp,
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
        "Collecting variational operator ensemble..."
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
            ) % 20 == 0
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
    thermalization_sweeps = 80
    number_of_configurations = 160
    sweeps_between_measurements = 3
    seed = 2026

    reference_time = 1
    relative_cutoff = 1e-8
    absolute_cutoff = 1e-14

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
        "Regularized Generalized-Eigenvalue Spectroscopy"
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
        f"Operator basis:                 "
        f"{basis.names}"
    )
    print(
        f"Reference time t0:              "
        f"{reference_time}"
    )
    print(
        f"Relative metric cutoff:         "
        f"{relative_cutoff:.3e}"
    )
    print(
        f"Absolute metric cutoff:         "
        f"{absolute_cutoff:.3e}"
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

    matrix_result = (
        ensemble_correlator_matrix(
            ensemble
        )
    )

    symmetric_matrices = (
        symmetrize_correlator_matrix(
            matrix_result.correlation_matrices
        )
    )

    gevp = solve_regularized_gevp(
        correlation_matrices=(
            symmetric_matrices
        ),
        reference_time=reference_time,
        relative_cutoff=relative_cutoff,
        absolute_cutoff=absolute_cutoff,
        track_states=True,
    )

    log_masses = (
        principal_log_effective_masses(
            gevp.principal_correlators
        )
    )

    arccosh_masses = (
        principal_arccosh_effective_masses(
            gevp.principal_correlators
        )
    )

    metric_error = (
        metric_orthonormality_error(
            vectors=(
                gevp.generalized_eigenvectors[
                    reference_time
                ]
            ),
            reference_matrix=(
                symmetric_matrices[
                    reference_time
                ]
            ),
        )
    )

    metric = gevp.reference_metric

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
    print()

    print("Reference correlator metric")
    print("-" * 108)
    print(
        f"Original operator dimension:    "
        f"{metric.original_dimension}"
    )
    print(
        f"Retained numerical rank:        "
        f"{metric.retained_rank}"
    )
    print(
        f"Reference eigenvalue cutoff:    "
        f"{metric.eigenvalue_cutoff:.12e}"
    )
    print(
        f"Retained condition number:      "
        f"{metric.condition_number:.12e}"
    )
    print(
        "Whitened reference identity error: "
        f"{gevp.maximum_reference_identity_error:.12e}"
    )
    print(
        "Generalized-vector metric error:   "
        f"{metric_error:.12e}"
    )
    print()

    print("Reference metric eigenvalues")
    print("-" * 108)

    for index, eigenvalue in enumerate(
        metric.eigenvalues
    ):
        retained = bool(
            metric.retained_mask[
                index
            ]
        )

        print(
            f"metric eigenvalue {index:>2d}: "
            f"{eigenvalue:>24.12e}    "
            f"{'RETAINED' if retained else 'REMOVED'}"
        )

    print()
    print("Principal correlators")
    print("-" * 108)

    header = (
        f"{'lag':>8}"
        + "".join(
            f"{'lambda_' + str(state):>24}"
            for state in range(
                gevp.retained_rank
            )
        )
    )

    print(header)
    print("-" * 108)

    for lag in gevp.lags:
        print(
            f"{lag:>8d}"
            + "".join(
                f"{value:>24.12e}"
                for value in (
                    gevp.principal_correlators[
                        lag
                    ]
                )
            )
        )

    print()
    print("Log-ratio variational effective masses")
    print("-" * 108)

    print(header.replace(
        "lambda_",
        "mass_",
    ))

    for lag in gevp.lags:
        values = []

        for mass in log_masses[
            lag
        ]:
            values.append(
                f"{mass:>24.12e}"
                if np.isfinite(
                    mass
                )
                else f"{'NaN':>24}"
            )

        print(
            f"{lag:>8d}"
            + "".join(
                values
            )
        )

    print()
    print("Arccosh variational effective masses")
    print("-" * 108)

    print(header.replace(
        "lambda_",
        "mass_",
    ))

    for lag in gevp.lags:
        values = []

        for mass in arccosh_masses[
            lag
        ]:
            values.append(
                f"{mass:>24.12e}"
                if np.isfinite(
                    mass
                )
                else f"{'NaN':>24}"
            )

        print(
            f"{lag:>8d}"
            + "".join(
                values
            )
        )

    print()
    print(
        "Generalized eigenvector components at reference time"
    )
    print("-" * 108)

    reference_vectors = (
        gevp.generalized_eigenvectors[
            reference_time
        ]
    )

    for state in range(
        gevp.retained_rank
    ):
        print(
            f"State {state}"
        )

        for operator_index, name in enumerate(
            basis.names
        ):
            print(
                f"  {name:<30}"
                f"{reference_vectors[operator_index, state]:>24.12e}"
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
        / "gevp_principal_correlators.csv"
    )

    with principal_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            ["lag"]
            + [
                f"principal_{state}"
                for state in range(
                    gevp.retained_rank
                )
            ]
        )

        for lag in gevp.lags:
            writer.writerow(
                [int(lag)]
                + list(
                    gevp.principal_correlators[
                        lag
                    ]
                )
            )

    mass_path = (
        data_dir
        / "gevp_effective_masses.csv"
    )

    mass_rows = []

    for lag in gevp.lags:
        for state in range(
            gevp.retained_rank
        ):
            mass_rows.append(
                {
                    "lag": int(lag),
                    "state": state,
                    "log_effective_mass": (
                        float(
                            log_masses[
                                lag,
                                state,
                            ]
                        )
                        if np.isfinite(
                            log_masses[
                                lag,
                                state,
                            ]
                        )
                        else ""
                    ),
                    "arccosh_effective_mass": (
                        float(
                            arccosh_masses[
                                lag,
                                state,
                            ]
                        )
                        if np.isfinite(
                            arccosh_masses[
                                lag,
                                state,
                            ]
                        )
                        else ""
                    ),
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

    metric_path = (
        data_dir
        / "gevp_reference_metric.csv"
    )

    with metric_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "metric_index",
                "eigenvalue",
                "retained",
            ]
        )

        for index, eigenvalue in enumerate(
            metric.eigenvalues
        ):
            writer.writerow(
                [
                    index,
                    float(
                        eigenvalue
                    ),
                    bool(
                        metric.retained_mask[
                            index
                        ]
                    ),
                ]
            )

    plt.figure()

    for state in range(
        gevp.retained_rank
    ):
        plt.plot(
            gevp.lags,
            gevp.principal_correlators[
                :,
                state,
            ],
            marker="o",
            label=f"state {state}",
        )

    plt.axvline(
        reference_time,
        linestyle="--",
        label="reference time",
    )
    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Principal correlator"
    )
    plt.title(
        "Regularized GEVP Principal Correlators"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_principal_correlators.png",
        dpi=200,
    )

    plt.figure()

    for state in range(
        gevp.retained_rank
    ):
        finite = np.isfinite(
            log_masses[
                :,
                state,
            ]
        )

        plt.plot(
            gevp.lags[finite],
            log_masses[
                finite,
                state,
            ],
            marker="o",
            label=f"state {state}",
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Log-ratio effective mass"
    )
    plt.title(
        "GEVP Variational Effective Masses"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_log_effective_masses.png",
        dpi=200,
    )

    plt.figure()

    for state in range(
        gevp.retained_rank
    ):
        finite = np.isfinite(
            arccosh_masses[
                :,
                state,
            ]
        )

        plt.plot(
            gevp.lags[finite],
            arccosh_masses[
                finite,
                state,
            ],
            marker="o",
            label=f"state {state}",
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Arccosh effective mass"
    )
    plt.title(
        "GEVP Periodic Effective Masses"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "gevp_arccosh_effective_masses.png",
        dpi=200,
    )

    print()
    print("Saved data:")
    print(principal_path)
    print(mass_path)
    print(metric_path)
    print()
    print("Saved figures:")
    print(
        "results/figures/"
        "gevp_principal_correlators.png"
    )
    print(
        "results/figures/"
        "gevp_log_effective_masses.png"
    )
    print(
        "results/figures/"
        "gevp_arccosh_effective_masses.png"
    )


if __name__ == "__main__":
    main()
