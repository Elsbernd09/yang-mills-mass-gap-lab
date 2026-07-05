"""
Exploratory SU(3) finite-lattice ensemble diagnostics.

The experiment runs several independent short generic SU(3) Metropolis chains
and measures:

1. acceptance rate,
2. average plaquette,
3. Wilson action per plaquette,
4. plaquette integrated autocorrelation time,
5. plaquette effective sample size,
6. rectangular Wilson loops,
7. final SU(3) link-membership error.

The study is a finite-lattice numerical prototype.

Equal bare beta values are not used to claim physical equivalence with SU(2).
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from ymlab.diagnostics import (
    diagnose_autocorrelation,
)
from ymlab.gauge_lattice import (
    GaugeLattice,
)
from ymlab.generic_gauge import (
    generic_average_plaquette,
    generic_metropolis_sweep,
    generic_number_of_plaquettes,
    generic_rectangular_wilson_loop,
    generic_wilson_action,
)
from ymlab.generic_gauge_transformations import (
    maximum_generic_link_membership_error,
)
from ymlab.group_interface import (
    su3_group,
)


def run_chain(
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    measurement_sweeps,
    seed,
):
    lattice = GaugeLattice(
        shape=shape,
        group=su3_group(),
        cold_start=True,
        seed=seed,
    )

    thermal_acceptance = []

    for _ in range(
        thermalization_sweeps
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=beta,
            epsilon=epsilon,
        )

        thermal_acceptance.append(
            result.acceptance_rate
        )

    plaquette_series = []
    action_density_series = []
    measurement_acceptance = []

    for sweep in range(
        measurement_sweeps
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=beta,
            epsilon=epsilon,
        )

        measurement_acceptance.append(
            result.acceptance_rate
        )

        plaquette_series.append(
            generic_average_plaquette(
                lattice
            )
        )

        action_density_series.append(
            generic_wilson_action(
                lattice=lattice,
                beta=beta,
            )
            / generic_number_of_plaquettes(
                lattice
            )
        )

        if (
            sweep == 0
            or (
                sweep + 1
            ) % 50 == 0
        ):
            print(
                f"    sweep {sweep + 1:04d}/"
                f"{measurement_sweeps:04d}"
            )

    plaquette_series = np.asarray(
        plaquette_series,
        dtype=float,
    )

    action_density_series = np.asarray(
        action_density_series,
        dtype=float,
    )

    diagnostics = diagnose_autocorrelation(
        plaquette_series
    )

    loops = {
        "W11": generic_rectangular_wilson_loop(
            lattice=lattice,
            site=tuple(
                0
                for _ in shape
            ),
            mu=0,
            nu=1,
            width=1,
            height=1,
        ),
        "W12": generic_rectangular_wilson_loop(
            lattice=lattice,
            site=tuple(
                0
                for _ in shape
            ),
            mu=0,
            nu=1,
            width=1,
            height=2,
        ),
        "W22": generic_rectangular_wilson_loop(
            lattice=lattice,
            site=tuple(
                0
                for _ in shape
            ),
            mu=0,
            nu=1,
            width=2,
            height=2,
        ),
    }

    return {
        "seed": seed,
        "mean_thermal_acceptance": float(
            np.mean(
                thermal_acceptance
            )
        ),
        "mean_measurement_acceptance": float(
            np.mean(
                measurement_acceptance
            )
        ),
        "mean_plaquette": float(
            np.mean(
                plaquette_series
            )
        ),
        "plaquette_standard_deviation": float(
            np.std(
                plaquette_series,
                ddof=1,
            )
        ),
        "mean_action_per_plaquette": float(
            np.mean(
                action_density_series
            )
        ),
        "plaquette_tau_int": float(
            diagnostics.integrated_autocorrelation_time
        ),
        "plaquette_effective_sample_size": float(
            diagnostics.effective_sample_size
        ),
        "W11": float(
            loops[
                "W11"
            ]
        ),
        "W12": float(
            loops[
                "W12"
            ]
        ),
        "W22": float(
            loops[
                "W22"
            ]
        ),
        "maximum_link_membership_error": float(
            maximum_generic_link_membership_error(
                lattice
            )
        ),
        "plaquette_series": plaquette_series,
    }


def main() -> None:
    shape = (
        4,
        4,
        4,
    )

    beta = 5.5
    epsilon = 0.05

    thermalization_sweeps = 80
    measurement_sweeps = 250

    seeds = [
        2026,
        2027,
        2028,
    ]

    print(
        "Exploratory SU(3) Finite-Lattice Ensemble Diagnostics"
    )
    print("=" * 142)

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
        f"Measurement sweeps per chain:   "
        f"{measurement_sweeps}"
    )

    print(
        f"Independent seeds:              "
        f"{seeds}"
    )

    results = []

    for seed in seeds:
        print()
        print(
            f"Running SU(3) chain with seed {seed}..."
        )

        results.append(
            run_chain(
                shape=shape,
                beta=beta,
                epsilon=epsilon,
                thermalization_sweeps=(
                    thermalization_sweeps
                ),
                measurement_sweeps=(
                    measurement_sweeps
                ),
                seed=seed,
            )
        )

    print()
    print(
        "SU(3) chain summary"
    )
    print("-" * 142)

    print(
        f"{'Seed':>8}"
        f"{'Accept':>12}"
        f"{'Plaq mean':>16}"
        f"{'Plaq std':>14}"
        f"{'S/P':>16}"
        f"{'tau_int':>14}"
        f"{'ESS':>14}"
        f"{'W11':>14}"
        f"{'W12':>14}"
        f"{'W22':>14}"
        f"{'Group err':>16}"
    )

    print("-" * 142)

    for result in results:
        print(
            f"{result['seed']:>8d}"
            f"{result['mean_measurement_acceptance']:>12.6f}"
            f"{result['mean_plaquette']:>16.8f}"
            f"{result['plaquette_standard_deviation']:>14.8f}"
            f"{result['mean_action_per_plaquette']:>16.8f}"
            f"{result['plaquette_tau_int']:>14.6f}"
            f"{result['plaquette_effective_sample_size']:>14.4f}"
            f"{result['W11']:>14.8f}"
            f"{result['W12']:>14.8f}"
            f"{result['W22']:>14.8f}"
            f"{result['maximum_link_membership_error']:>16.6e}"
        )

    plaquette_means = np.asarray(
        [
            result[
                "mean_plaquette"
            ]
            for result in results
        ],
        dtype=float,
    )

    acceptance_means = np.asarray(
        [
            result[
                "mean_measurement_acceptance"
            ]
            for result in results
        ],
        dtype=float,
    )

    print()
    print(
        "Cross-chain diagnostics"
    )
    print("-" * 142)

    print(
        f"Mean chain plaquette:           "
        f"{np.mean(plaquette_means):.12f}"
    )

    print(
        f"Std of chain plaquette means:   "
        f"{np.std(plaquette_means, ddof=1):.12f}"
    )

    print(
        f"Mean Metropolis acceptance:     "
        f"{np.mean(acceptance_means):.8f}"
    )

    print(
        f"Maximum final group error:      "
        f"{max(result['maximum_link_membership_error'] for result in results):.12e}"
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

    output_path = (
        data_dir
        / "su3_ensemble_diagnostics.csv"
    )

    rows = []

    for result in results:
        rows.append(
            {
                key: value
                for key, value in result.items()
                if key != "plaquette_series"
            }
        )

    with output_path.open(
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

    plt.figure()

    for result in results:
        plt.plot(
            np.arange(
                len(
                    result[
                        "plaquette_series"
                    ]
                )
            ),
            result[
                "plaquette_series"
            ],
            label=(
                f"seed {result['seed']}"
            ),
        )

    plt.xlabel(
        "Measurement sweep"
    )

    plt.ylabel(
        "Average normalized plaquette"
    )

    plt.title(
        "SU(3) Finite-Lattice Plaquette Chains"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "su3_plaquette_chains.png",
        dpi=200,
    )

    plt.figure()

    plt.errorbar(
        np.arange(
            len(
                results
            )
        ),
        plaquette_means,
        yerr=np.asarray(
            [
                result[
                    "plaquette_standard_deviation"
                ]
                for result in results
            ],
            dtype=float,
        ),
        marker="o",
        linestyle="none",
    )

    plt.xlabel(
        "Independent chain index"
    )

    plt.ylabel(
        "Mean average plaquette"
    )

    plt.title(
        "SU(3) Independent-Chain Plaquette Summaries"
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "su3_chain_plaquette_summary.png",
        dpi=200,
    )

    print()
    print(
        f"Saved ensemble data: {output_path}"
    )

    print(
        "Saved figures:"
    )

    print(
        "results/figures/"
        "su3_plaquette_chains.png"
    )

    print(
        "results/figures/"
        "su3_chain_plaquette_summary.png"
    )


if __name__ == "__main__":
    main()
