"""
Spatial-smearing comparison for scalar glueball-style spectroscopy.

For each measured SU(2) gauge configuration, the experiment constructs scalar
operator time series at several spatial smearing levels.

The same underlying configurations are used at every level.

The experiment compares:

1. Connected correlators.
2. Normalized correlators.
3. Arccosh effective-mass validity.
4. Plateau diagnostics.
5. Periodic cosh fit availability.
6. Fit residuals.
7. SU(2) projection errors.

This is an exploratory operator-improvement study on finite lattices.
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
from ymlab.smearing import (
    maximum_su2_link_error,
    smear_spatial_links,
)
from ymlab.spectroscopy import (
    arccosh_effective_mass,
    fit_periodic_cosh,
    scan_effective_mass_plateaus,
)


def collect_multilevel_ensemble(
    shape,
    time_direction,
    beta,
    epsilon,
    thermalization_sweeps,
    number_of_configurations,
    sweeps_between_measurements,
    smearing_levels,
    alpha,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    print("Thermalizing lattice...")

    thermal_acceptance = []

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

    ensembles = {
        level: []
        for level in smearing_levels
    }

    maximum_projection_errors = {
        level: 0.0
        for level in smearing_levels
    }

    measurement_acceptance = []

    print(
        "Collecting multi-level operator ensemble..."
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

        for level in smearing_levels:
            smeared = smear_spatial_links(
                lattice=lattice,
                time_direction=time_direction,
                alpha=alpha,
                steps=level,
            )

            operator = (
                scalar_glueball_time_series(
                    lattice=smeared,
                    time_direction=time_direction,
                )
            )

            ensembles[level].append(
                operator
            )

            maximum_projection_errors[level] = max(
                maximum_projection_errors[level],
                maximum_su2_link_error(
                    smeared
                ),
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

    converted = {
        level: np.asarray(
            values,
            dtype=float,
        )
        for level, values
        in ensembles.items()
    }

    return (
        converted,
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
        maximum_projection_errors,
    )


def best_positive_cosh_fit(
    correlation,
):
    temporal_extent = len(
        correlation
    )

    candidates = []

    maximum_stop = (
        temporal_extent // 2
        + 1
    )

    for fit_start in [1, 2]:
        for fit_stop in range(
            fit_start + 2,
            maximum_stop + 1,
        ):
            try:
                fit = fit_periodic_cosh(
                    correlation=correlation,
                    fit_start=fit_start,
                    fit_stop=fit_stop,
                )
            except ValueError:
                continue

            if not fit.success:
                continue

            residuals = (
                fit.fit_values
                - fit.model_values
            )

            mse = float(
                np.mean(
                    residuals ** 2
                )
            )

            candidates.append(
                (
                    mse,
                    -len(
                        fit.fit_values
                    ),
                    fit,
                )
            )

    if len(candidates) == 0:
        return None, np.nan

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return (
        candidates[0][2],
        candidates[0][0],
    )


def main() -> None:
    shape = (8, 5, 5)
    time_direction = 0
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 70
    number_of_configurations = 100
    sweeps_between_measurements = 3

    smearing_levels = [
        0,
        2,
        5,
        10,
    ]

    alpha = 0.5
    seed = 2026

    print(
        "Spatial Smearing Spectroscopy Comparison"
    )
    print("=" * 100)
    print(
        f"Lattice shape:                 {shape}"
    )
    print(
        f"Euclidean time direction:      {time_direction}"
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
        f"Measured configurations:       "
        f"{number_of_configurations}"
    )
    print(
        "Sweeps between measurements:   "
        f"{sweeps_between_measurements}"
    )
    print(
        f"Smearing alpha:                {alpha}"
    )
    print(
        f"Smearing levels:               "
        f"{smearing_levels}"
    )
    print()

    (
        ensembles,
        thermal_acceptance,
        measurement_acceptance,
        projection_errors,
    ) = collect_multilevel_ensemble(
        shape=shape,
        time_direction=time_direction,
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
        smearing_levels=smearing_levels,
        alpha=alpha,
        seed=seed,
    )

    print()
    print(
        "Mean thermal acceptance:      "
        f"{thermal_acceptance:.8f}"
    )
    print(
        "Mean measurement acceptance:  "
        f"{measurement_acceptance:.8f}"
    )
    print()

    summaries = []
    analysis = {}

    for level in smearing_levels:
        ensemble = ensembles[level]

        correlator = (
            ensemble_connected_correlator(
                ensemble
            )
        )

        normalized = (
            normalized_connected_correlator(
                correlator.correlation
            )
        )

        effective_mass = (
            arccosh_effective_mass(
                correlator.correlation
            )
        )

        plateaus = (
            scan_effective_mass_plateaus(
                effective_mass=effective_mass,
                minimum_window=2,
                maximum_window=4,
            )
        )

        fit, fit_mse = (
            best_positive_cosh_fit(
                correlator.correlation
            )
        )

        finite_effective_masses = int(
            np.sum(
                np.isfinite(
                    effective_mass
                )
            )
        )

        best_plateau_score = (
            plateaus[0].score
            if len(plateaus) > 0
            else np.nan
        )

        fitted_mass = (
            fit.mass
            if fit is not None
            else np.nan
        )

        analysis[level] = {
            "correlator": correlator,
            "normalized": normalized,
            "effective_mass": effective_mass,
            "plateaus": plateaus,
            "fit": fit,
        }

        summaries.append(
            {
                "smearing_level": level,
                "alpha": alpha,
                "mean_operator": (
                    correlator.mean_operator
                ),
                "c0": float(
                    correlator.correlation[0]
                ),
                "finite_effective_mass_points": (
                    finite_effective_masses
                ),
                "best_plateau_score": float(
                    best_plateau_score
                ),
                "cosh_fit_available": (
                    fit is not None
                ),
                "fitted_mass": float(
                    fitted_mass
                ),
                "fit_mse": float(
                    fit_mse
                ),
                "maximum_su2_error": float(
                    projection_errors[level]
                ),
            }
        )

    print(
        "Smearing-level spectroscopy summary"
    )
    print("-" * 154)
    print(
        f"{'Level':>8}"
        f"{'<O>':>20}"
        f"{'C(0)':>20}"
        f"{'Finite m_eff':>16}"
        f"{'Plateau Score':>20}"
        f"{'Cosh Fit':>12}"
        f"{'Fit Mass':>20}"
        f"{'Fit MSE':>20}"
        f"{'SU2 Error':>18}"
    )
    print("-" * 154)

    for row in summaries:
        plateau_text = (
            f"{row['best_plateau_score']:.8e}"
            if np.isfinite(
                row["best_plateau_score"]
            )
            else "NaN"
        )

        mass_text = (
            f"{row['fitted_mass']:.10f}"
            if np.isfinite(
                row["fitted_mass"]
            )
            else "NaN"
        )

        mse_text = (
            f"{row['fit_mse']:.8e}"
            if np.isfinite(
                row["fit_mse"]
            )
            else "NaN"
        )

        print(
            f"{row['smearing_level']:>8d}"
            f"{row['mean_operator']:>20.10f}"
            f"{row['c0']:>20.10e}"
            f"{row['finite_effective_mass_points']:>16d}"
            f"{plateau_text:>20}"
            f"{str(row['cosh_fit_available']):>12}"
            f"{mass_text:>20}"
            f"{mse_text:>20}"
            f"{row['maximum_su2_error']:>18.8e}"
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

    summary_path = (
        data_dir
        / "smearing_spectroscopy_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    plt.figure()

    for level in smearing_levels:
        result = analysis[level]

        correlator = result[
            "correlator"
        ]

        normalized = result[
            "normalized"
        ]

        plt.plot(
            correlator.lags,
            normalized,
            marker="o",
            label=(
                f"smearing level {level}"
            ),
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "C(t) / C(0)"
    )
    plt.title(
        "Normalized Correlators Across Smearing Levels"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "smearing_normalized_correlators.png",
        dpi=200,
    )

    plt.figure()

    for level in smearing_levels:
        result = analysis[level]

        correlator = result[
            "correlator"
        ]

        effective_mass = result[
            "effective_mass"
        ]

        finite = np.isfinite(
            effective_mass
        )

        plt.plot(
            correlator.lags[finite],
            effective_mass[finite],
            marker="o",
            label=(
                f"smearing level {level}"
            ),
        )

    plt.xlabel(
        "Euclidean time lag"
    )
    plt.ylabel(
        "Arccosh effective mass"
    )
    plt.title(
        "Effective Mass Across Smearing Levels"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "smearing_effective_mass_comparison.png",
        dpi=200,
    )

    levels = np.asarray(
        [
            row["smearing_level"]
            for row in summaries
        ],
        dtype=int,
    )

    finite_counts = np.asarray(
        [
            row[
                "finite_effective_mass_points"
            ]
            for row in summaries
        ],
        dtype=int,
    )

    plt.figure()
    plt.plot(
        levels,
        finite_counts,
        marker="o",
    )
    plt.xlabel(
        "Smearing steps"
    )
    plt.ylabel(
        "Finite effective-mass points"
    )
    plt.title(
        "Effective-Mass Validity vs Smearing Level"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "smearing_effective_mass_validity.png",
        dpi=200,
    )

    print()
    print(
        f"Saved summary data: {summary_path}"
    )
    print(
        "Saved figures:"
    )
    print(
        "results/figures/"
        "smearing_normalized_correlators.png"
    )
    print(
        "results/figures/"
        "smearing_effective_mass_comparison.png"
    )
    print(
        "results/figures/"
        "smearing_effective_mass_validity.png"
    )


if __name__ == "__main__":
    main()
