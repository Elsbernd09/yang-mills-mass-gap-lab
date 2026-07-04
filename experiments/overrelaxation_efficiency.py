"""
Metropolis versus hybrid Metropolis-overrelaxation efficiency comparison.

The experiment compares update schedules:

    M
    M + 1 OR
    M + 2 OR
    M + 4 OR

where M is one Metropolis sweep and OR is one SU(2) microcanonical
overrelaxation sweep.

The comparison measures:

1. runtime,
2. Metropolis acceptance,
3. average plaquette,
4. scalar composite operator,
5. integrated autocorrelation time,
6. effective sample size,
7. effective sample size per second.

The purpose is not to assume that more overrelaxation is always better.

Additional microcanonical sweeps can reduce autocorrelation while also
increasing computational cost.

The relevant efficiency measure is therefore observable-dependent ESS/sec.
"""

from __future__ import annotations

from pathlib import Path
import csv
from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ymlab.diagnostics import diagnose_autocorrelation
from ymlab.glueball import scalar_glueball_time_series
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.overrelaxation import (
    hybrid_metropolis_overrelaxation_sweep,
)
from ymlab.plaquette import average_plaquette


@dataclass(frozen=True)
class EfficiencyResult:
    schedule: str
    overrelaxation_sweeps: int
    runtime_seconds: float
    mean_acceptance_rate: float
    plaquette_mean: float
    plaquette_tau_int: float
    plaquette_ess: float
    plaquette_ess_per_second: float
    scalar_mean: float
    scalar_tau_int: float
    scalar_ess: float
    scalar_ess_per_second: float
    maximum_microcanonical_error: float


def initialize_thermalized_lattice(
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    for _ in range(
        thermalization_sweeps
    ):
        metropolis_sweep(
            lattice=lattice,
            beta=beta,
            epsilon=epsilon,
        )

    return lattice


def copy_lattice(
    lattice,
):
    copied = Lattice(
        shape=lattice.shape,
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            copied.set_link(
                site,
                mu,
                np.array(
                    lattice.get_link(
                        site,
                        mu,
                    ),
                    dtype=complex,
                    copy=True,
                ),
            )

    return copied


def run_schedule(
    initial_lattice,
    beta,
    epsilon,
    overrelaxation_sweeps,
    measurement_updates,
    time_direction,
):
    lattice = copy_lattice(
        initial_lattice
    )

    plaquette_series = []
    scalar_series = []
    acceptance_rates = []

    maximum_microcanonical_error = 0.0

    start = perf_counter()

    for update_index in range(
        measurement_updates
    ):
        result = (
            hybrid_metropolis_overrelaxation_sweep(
                lattice=lattice,
                beta=beta,
                epsilon=epsilon,
                overrelaxation_sweeps=(
                    overrelaxation_sweeps
                ),
            )
        )

        acceptance_rates.append(
            result.metropolis_acceptance_rate
        )

        maximum_microcanonical_error = max(
            maximum_microcanonical_error,
            result.maximum_local_action_error,
        )

        plaquette_series.append(
            average_plaquette(
                lattice
            )
        )

        scalar_series.append(
            float(
                np.mean(
                    scalar_glueball_time_series(
                        lattice=lattice,
                        time_direction=time_direction,
                    )
                )
            )
        )

        if (
            update_index == 0
            or (
                update_index + 1
            ) % 100 == 0
        ):
            print(
                "    completed update "
                f"{update_index + 1:04d}/"
                f"{measurement_updates:04d}"
            )

    runtime = (
        perf_counter()
        - start
    )

    plaquette_series = np.asarray(
        plaquette_series,
        dtype=float,
    )

    scalar_series = np.asarray(
        scalar_series,
        dtype=float,
    )

    plaquette_diagnostics = (
        diagnose_autocorrelation(
            plaquette_series
        )
    )

    scalar_diagnostics = (
        diagnose_autocorrelation(
            scalar_series
        )
    )

    schedule = (
        "Metropolis"
        if overrelaxation_sweeps == 0
        else (
            "Metropolis + "
            f"{overrelaxation_sweeps} OR"
        )
    )

    return EfficiencyResult(
        schedule=schedule,
        overrelaxation_sweeps=(
            overrelaxation_sweeps
        ),
        runtime_seconds=float(
            runtime
        ),
        mean_acceptance_rate=float(
            np.mean(
                acceptance_rates
            )
        ),
        plaquette_mean=float(
            np.mean(
                plaquette_series
            )
        ),
        plaquette_tau_int=float(
            plaquette_diagnostics
            .integrated_autocorrelation_time
        ),
        plaquette_ess=float(
            plaquette_diagnostics
            .effective_sample_size
        ),
        plaquette_ess_per_second=float(
            plaquette_diagnostics
            .effective_sample_size
            / runtime
        ),
        scalar_mean=float(
            np.mean(
                scalar_series
            )
        ),
        scalar_tau_int=float(
            scalar_diagnostics
            .integrated_autocorrelation_time
        ),
        scalar_ess=float(
            scalar_diagnostics
            .effective_sample_size
        ),
        scalar_ess_per_second=float(
            scalar_diagnostics
            .effective_sample_size
            / runtime
        ),
        maximum_microcanonical_error=float(
            maximum_microcanonical_error
        ),
    )


def main() -> None:
    shape = (
        6,
        5,
        5,
    )

    beta = 2.0
    epsilon = 0.18

    thermalization_sweeps = 100
    measurement_updates = 600

    time_direction = 0
    seed = 2026

    schedules = [
        0,
        1,
        2,
        4,
    ]

    print(
        "Metropolis / Overrelaxation Efficiency Comparison"
    )
    print("=" * 154)

    print(
        f"Lattice shape:                 {shape}"
    )
    print(
        f"Beta:                          {beta}"
    )
    print(
        f"Proposal epsilon:              {epsilon}"
    )
    print(
        f"Thermalization sweeps:         "
        f"{thermalization_sweeps}"
    )
    print(
        f"Measured hybrid updates:       "
        f"{measurement_updates}"
    )
    print(
        f"Schedules:                     "
        f"{schedules}"
    )
    print()

    print(
        "Constructing shared thermalized starting configuration..."
    )

    initial_lattice = (
        initialize_thermalized_lattice(
            shape=shape,
            beta=beta,
            epsilon=epsilon,
            thermalization_sweeps=(
                thermalization_sweeps
            ),
            seed=seed,
        )
    )

    results = []

    for overrelaxation_sweeps in schedules:
        print()
        print("=" * 154)

        print(
            "Running schedule: "
            + (
                "Metropolis only"
                if overrelaxation_sweeps == 0
                else (
                    "Metropolis + "
                    f"{overrelaxation_sweeps} "
                    "overrelaxation sweep(s)"
                )
            )
        )

        print("=" * 154)

        result = run_schedule(
            initial_lattice=(
                initial_lattice
            ),
            beta=beta,
            epsilon=epsilon,
            overrelaxation_sweeps=(
                overrelaxation_sweeps
            ),
            measurement_updates=(
                measurement_updates
            ),
            time_direction=time_direction,
        )

        results.append(
            result
        )

    print()
    print(
        "Efficiency summary"
    )
    print("-" * 154)

    print(
        f"{'Schedule':<24}"
        f"{'Runtime':>12}"
        f"{'Accept':>12}"
        f"{'Plaq tau':>14}"
        f"{'Plaq ESS':>14}"
        f"{'Plaq ESS/s':>16}"
        f"{'Scalar tau':>14}"
        f"{'Scalar ESS':>14}"
        f"{'Scalar ESS/s':>16}"
        f"{'Max dS err':>16}"
    )

    print("-" * 154)

    for result in results:
        print(
            f"{result.schedule:<24}"
            f"{result.runtime_seconds:>12.4f}"
            f"{result.mean_acceptance_rate:>12.6f}"
            f"{result.plaquette_tau_int:>14.6f}"
            f"{result.plaquette_ess:>14.4f}"
            f"{result.plaquette_ess_per_second:>16.6f}"
            f"{result.scalar_tau_int:>14.6f}"
            f"{result.scalar_ess:>14.4f}"
            f"{result.scalar_ess_per_second:>16.6f}"
            f"{result.maximum_microcanonical_error:>16.6e}"
        )

    best_plaquette = max(
        results,
        key=lambda result: (
            result.plaquette_ess_per_second
        ),
    )

    best_scalar = max(
        results,
        key=lambda result: (
            result.scalar_ess_per_second
        ),
    )

    print()
    print(
        "Best plaquette ESS/sec schedule: "
        f"{best_plaquette.schedule}"
    )
    print(
        "Best scalar ESS/sec schedule:    "
        f"{best_scalar.schedule}"
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
        / "overrelaxation_efficiency.csv"
    )

    rows = [
        {
            field: getattr(
                result,
                field,
            )
            for field in (
                EfficiencyResult
                .__dataclass_fields__
                .keys()
            )
        }
        for result in results
    ]

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

    counts = np.asarray(
        [
            result.overrelaxation_sweeps
            for result in results
        ],
        dtype=int,
    )

    plaquette_tau = np.asarray(
        [
            result.plaquette_tau_int
            for result in results
        ],
        dtype=float,
    )

    scalar_tau = np.asarray(
        [
            result.scalar_tau_int
            for result in results
        ],
        dtype=float,
    )

    plt.figure()

    plt.plot(
        counts,
        plaquette_tau,
        marker="o",
        label="Average plaquette",
    )

    plt.plot(
        counts,
        scalar_tau,
        marker="o",
        label="Scalar operator",
    )

    plt.xlabel(
        "Overrelaxation sweeps per Metropolis sweep"
    )
    plt.ylabel(
        "Integrated autocorrelation time"
    )
    plt.title(
        "Autocorrelation vs Overrelaxation Frequency"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "overrelaxation_autocorrelation.png",
        dpi=200,
    )

    plaquette_efficiency = np.asarray(
        [
            result.plaquette_ess_per_second
            for result in results
        ],
        dtype=float,
    )

    scalar_efficiency = np.asarray(
        [
            result.scalar_ess_per_second
            for result in results
        ],
        dtype=float,
    )

    plt.figure()

    plt.plot(
        counts,
        plaquette_efficiency,
        marker="o",
        label="Average plaquette",
    )

    plt.plot(
        counts,
        scalar_efficiency,
        marker="o",
        label="Scalar operator",
    )

    plt.xlabel(
        "Overrelaxation sweeps per Metropolis sweep"
    )
    plt.ylabel(
        "Effective sample size per second"
    )
    plt.title(
        "Sampling Efficiency vs Overrelaxation Frequency"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "overrelaxation_ess_per_second.png",
        dpi=200,
    )

    print()
    print(
        f"Saved efficiency data: {output_path}"
    )
    print(
        "Saved figures:"
    )
    print(
        "results/figures/"
        "overrelaxation_autocorrelation.png"
    )
    print(
        "results/figures/"
        "overrelaxation_ess_per_second.png"
    )


if __name__ == "__main__":
    main()
