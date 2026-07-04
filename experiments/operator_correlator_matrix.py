"""
Multi-operator scalar correlator-matrix experiment.

This experiment constructs an operator basis using several spatial smearing
levels measured on the same SU(2) gauge configurations.

It then estimates

    C_ij(tau)
        =
        <O_i(t) O_j(t + tau)>
        -
        <O_i><O_j>.

The resulting correlator matrix prepares the project for a generalized
eigenvalue analysis.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.operator_basis import (
    correlation_matrix_eigenvalues,
    create_smearing_basis,
    ensemble_correlator_matrix,
    maximum_matrix_asymmetry,
    measure_operator_basis,
    symmetrize_correlator_matrix,
)


def collect_basis_ensemble(
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
        "Collecting multi-operator measurements..."
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
    shape = (8, 5, 5)
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 70
    number_of_configurations = 120
    sweeps_between_measurements = 3
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
        "Multi-Operator Scalar Correlator Matrix"
    )
    print("=" * 100)
    print(f"Lattice shape:                  {shape}")
    print(f"Beta:                           {beta}")
    print(f"Proposal epsilon:               {epsilon}")
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
        f"Operator names:                 "
        f"{basis.names}"
    )
    print(
        f"Smearing alpha:                 "
        f"{basis.alpha}"
    )
    print()

    (
        ensemble,
        thermal_acceptance,
        measurement_acceptance,
    ) = collect_basis_ensemble(
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

    result = ensemble_correlator_matrix(
        ensemble
    )

    raw_asymmetry = maximum_matrix_asymmetry(
        result.correlation_matrices
    )

    symmetric_matrices = (
        symmetrize_correlator_matrix(
            result.correlation_matrices
        )
    )

    symmetric_asymmetry = (
        maximum_matrix_asymmetry(
            symmetric_matrices
        )
    )

    eigenvalues = (
        correlation_matrix_eigenvalues(
            symmetric_matrices,
            symmetrize=False,
        )
    )

    print()
    print("Ensemble summary")
    print("-" * 100)
    print(
        f"Operator ensemble shape:         "
        f"{ensemble.shape}"
    )
    print(
        f"Mean thermal acceptance:         "
        f"{thermal_acceptance:.8f}"
    )
    print(
        f"Mean measurement acceptance:     "
        f"{measurement_acceptance:.8f}"
    )
    print(
        f"Number of operators:             "
        f"{result.number_of_operators}"
    )
    print(
        f"Temporal extent:                 "
        f"{result.temporal_extent}"
    )
    print(
        f"Raw maximum matrix asymmetry:    "
        f"{raw_asymmetry:.12e}"
    )
    print(
        f"Symmetrized matrix asymmetry:    "
        f"{symmetric_asymmetry:.12e}"
    )
    print()

    print("Operator means")
    print("-" * 100)

    for name, mean_value in zip(
        basis.names,
        result.operator_means,
    ):
        print(
            f"{name:<30}"
            f"{mean_value:>24.12e}"
        )

    print()
    print("Symmetrized C_ij(tau) matrices")
    print("-" * 100)

    for lag in result.lags:
        print()
        print(f"tau = {lag}")
        print(
            np.array2string(
                symmetric_matrices[lag],
                precision=8,
                suppress_small=False,
            )
        )

        print(
            "eigenvalues = "
            + np.array2string(
                eigenvalues[lag],
                precision=8,
                suppress_small=False,
            )
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

    matrix_path = (
        data_dir
        / "operator_correlator_matrix.csv"
    )

    rows = []

    for lag in result.lags:
        for i, name_i in enumerate(
            basis.names
        ):
            for j, name_j in enumerate(
                basis.names
            ):
                rows.append(
                    {
                        "lag": int(lag),
                        "operator_i": name_i,
                        "operator_j": name_j,
                        "connected_correlation": float(
                            result.correlation_matrices[
                                lag,
                                i,
                                j,
                            ]
                        ),
                        "symmetrized_correlation": float(
                            symmetric_matrices[
                                lag,
                                i,
                                j,
                            ]
                        ),
                    }
                )

    with matrix_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    eigenvalue_path = (
        data_dir
        / "operator_correlator_eigenvalues.csv"
    )

    with eigenvalue_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            ["lag"]
            + [
                f"eigenvalue_{index}"
                for index in range(
                    result.number_of_operators
                )
            ]
        )

        for lag in result.lags:
            writer.writerow(
                [int(lag)]
                + list(
                    eigenvalues[lag]
                )
            )

    for lag in [
        0,
        1,
        min(
            2,
            result.temporal_extent - 1,
        ),
    ]:
        plt.figure()
        plt.imshow(
            symmetric_matrices[lag],
            aspect="auto",
        )
        plt.xticks(
            np.arange(
                result.number_of_operators
            ),
            basis.names,
            rotation=30,
            ha="right",
        )
        plt.yticks(
            np.arange(
                result.number_of_operators
            ),
            basis.names,
        )
        plt.colorbar(
            label="Connected correlation"
        )
        plt.title(
            f"Operator Correlator Matrix at tau={lag}"
        )
        plt.tight_layout()
        plt.savefig(
            figures_dir
            / (
                "operator_correlator_matrix_"
                f"tau_{lag}.png"
            ),
            dpi=200,
        )

    plt.figure()

    for eigenvalue_index in range(
        result.number_of_operators
    ):
        plt.plot(
            result.lags,
            eigenvalues[
                :,
                eigenvalue_index,
            ],
            marker="o",
            label=(
                "eigenvalue "
                f"{eigenvalue_index}"
            ),
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Correlation-matrix eigenvalue"
    )
    plt.title(
        "Correlator Matrix Eigenvalues"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "operator_correlator_eigenvalues.png",
        dpi=200,
    )

    print()
    print(
        f"Saved correlator matrix: {matrix_path}"
    )
    print(
        f"Saved matrix eigenvalues: "
        f"{eigenvalue_path}"
    )
    print("Saved figures:")
    print(
        "results/figures/"
        "operator_correlator_matrix_tau_0.png"
    )
    print(
        "results/figures/"
        "operator_correlator_matrix_tau_1.png"
    )
    print(
        "results/figures/"
        "operator_correlator_matrix_tau_2.png"
    )
    print(
        "results/figures/"
        "operator_correlator_eigenvalues.png"
    )


if __name__ == "__main__":
    main()
