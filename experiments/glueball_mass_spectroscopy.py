"""
Finite-lattice scalar glueball-style spectroscopy experiment.

Pipeline:

1. Thermalize a finite 3D SU(2) lattice.
2. Collect gauge-invariant scalar operator O(t) measurements.
3. Construct the connected periodic Euclidean correlator.
4. Compute the arccosh effective mass.
5. Scan candidate plateau windows.
6. Search positive periodic-cosh fit windows.
7. Bootstrap the selected cosh mass at the configuration level.

The resulting quantities are finite-lattice spectroscopy diagnostics. They are
not a proof of the continuum Yang-Mills mass gap or a precision glueball mass.
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
from ymlab.spectroscopy import (
    arccosh_effective_mass,
    bootstrap_cosh_mass,
    fit_periodic_cosh,
    periodic_cosh_correlator,
    scan_effective_mass_plateaus,
)


def collect_operator_ensemble(
    shape,
    time_direction,
    beta,
    epsilon,
    thermalization_sweeps,
    number_of_configurations,
    sweeps_between_measurements,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    thermal_acceptance = []

    for _ in range(thermalization_sweeps):
        thermal_acceptance.append(
            metropolis_sweep(
                lattice=lattice,
                beta=beta,
                epsilon=epsilon,
            )
        )

    operator_ensemble = []
    measurement_acceptance = []

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

        operator_ensemble.append(
            scalar_glueball_time_series(
                lattice=lattice,
                time_direction=time_direction,
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
            operator_ensemble,
            dtype=float,
        ),
        float(
            np.mean(thermal_acceptance)
        ),
        float(
            np.mean(measurement_acceptance)
        ),
    )


def find_positive_fit(
    correlation,
):
    temporal_extent = len(correlation)

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

            residual = (
                fit.fit_values
                - fit.model_values
            )

            mean_squared_residual = float(
                np.mean(
                    residual ** 2
                )
            )

            candidates.append(
                (
                    mean_squared_residual,
                    -(fit.fit_stop - fit.fit_start),
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
    shape = (10, 6, 6)
    time_direction = 0
    beta = 2.0
    epsilon = 0.18
    thermalization_sweeps = 100
    number_of_configurations = 180
    sweeps_between_measurements = 4
    seed = 2026
    n_bootstrap = 500

    print("Scalar Glueball-Style Mass Spectroscopy")
    print("=" * 96)
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
    print(f"Bootstrap mass attempts:        {n_bootstrap}")
    print()

    print(
        "Generating scalar operator ensemble..."
    )

    (
        ensemble,
        thermal_acceptance,
        measurement_acceptance,
    ) = collect_operator_ensemble(
        shape=shape,
        time_direction=time_direction,
        beta=beta,
        epsilon=epsilon,
        thermalization_sweeps=thermalization_sweeps,
        number_of_configurations=number_of_configurations,
        sweeps_between_measurements=sweeps_between_measurements,
        seed=seed,
    )

    correlator = ensemble_connected_correlator(
        ensemble
    )

    normalized = normalized_connected_correlator(
        correlator.correlation
    )

    effective_mass = arccosh_effective_mass(
        correlator.correlation
    )

    plateaus = scan_effective_mass_plateaus(
        effective_mass=effective_mass,
        minimum_window=2,
        maximum_window=4,
    )

    fit = find_positive_fit(
        correlator.correlation
    )

    print()
    print("Simulation summary")
    print("-" * 96)
    print(
        f"Operator ensemble shape:        {ensemble.shape}"
    )
    print(
        "Mean thermal acceptance:        "
        f"{thermal_acceptance:.8f}"
    )
    print(
        "Mean measurement acceptance:    "
        f"{measurement_acceptance:.8f}"
    )
    print(
        "Mean scalar operator:            "
        f"{correlator.mean_operator:.12f}"
    )
    print()

    print("Correlator and effective mass")
    print("-" * 112)
    print(
        f"{'Lag':>8}"
        f"{'Connected C(t)':>26}"
        f"{'Normalized C(t)':>26}"
        f"{'Arccosh m_eff':>26}"
    )
    print("-" * 112)

    for lag, connected, norm, mass in zip(
        correlator.lags,
        correlator.correlation,
        normalized,
        effective_mass,
    ):
        mass_text = (
            f"{mass:.12e}"
            if np.isfinite(mass)
            else "NaN"
        )

        print(
            f"{lag:>8d}"
            f"{connected:>26.12e}"
            f"{norm:>26.12e}"
            f"{mass_text:>26}"
        )

    print()
    print("Best effective-mass plateau candidates")
    print("-" * 96)

    if len(plateaus) == 0:
        print(
            "No fully finite positive plateau windows found."
        )
    else:
        print(
            f"{'Start':>10}"
            f"{'Stop':>10}"
            f"{'Points':>10}"
            f"{'Mean Mass':>20}"
            f"{'Std Dev':>20}"
            f"{'Score':>20}"
        )

        for plateau in plateaus[:5]:
            print(
                f"{plateau.start:>10d}"
                f"{plateau.stop:>10d}"
                f"{plateau.number_of_points:>10d}"
                f"{plateau.mean_mass:>20.12f}"
                f"{plateau.standard_deviation:>20.12f}"
                f"{plateau.score:>20.12e}"
            )

    bootstrap = None

    print()
    print("Periodic cosh fit")
    print("-" * 96)

    if fit is None:
        print(
            "No positive fit window supported a successful "
            "periodic single-state cosh fit."
        )
        print(
            "No mass claim will be fabricated from this ensemble."
        )
    else:
        print(
            f"Fit window:                     "
            f"[{fit.fit_start}, {fit.fit_stop})"
        )
        print(
            f"Fitted amplitude:               "
            f"{fit.amplitude:.12e}"
        )
        print(
            f"Covariance amplitude error:      "
            f"{fit.amplitude_error:.12e}"
        )
        print(
            f"Fitted lattice mass:            "
            f"{fit.mass:.12f}"
        )
        print(
            f"Covariance mass error:           "
            f"{fit.mass_error:.12f}"
        )

        print()
        print(
            "Running configuration-level bootstrap "
            "mass analysis..."
        )

        try:
            bootstrap = bootstrap_cosh_mass(
                operator_series_ensemble=ensemble,
                fit_start=fit.fit_start,
                fit_stop=fit.fit_stop,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                seed=seed + 5000,
            )

            print(
                f"Successful bootstrap fits:       "
                f"{bootstrap.successful_fits}/"
                f"{bootstrap.attempted_fits}"
            )
            print(
                f"Bootstrap mean mass:             "
                f"{bootstrap.estimate:.12f}"
            )
            print(
                f"Bootstrap mass standard error:   "
                f"{bootstrap.standard_error:.12f}"
            )
            print(
                f"Bootstrap 95% interval:          "
                f"[{bootstrap.lower_ci:.12f}, "
                f"{bootstrap.upper_ci:.12f}]"
            )

        except ValueError as error:
            print(
                "Bootstrap mass analysis did not produce "
                "enough valid fits."
            )
            print(f"Reason: {error}")

    data_dir = Path("results/data")
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = Path("results/figures")
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlator_path = (
        data_dir
        / "glueball_mass_spectroscopy.csv"
    )

    rows = []

    for lag, connected, norm, mass in zip(
        correlator.lags,
        correlator.correlation,
        normalized,
        effective_mass,
    ):
        rows.append(
            {
                "lag": int(lag),
                "connected_correlation": float(
                    connected
                ),
                "normalized_correlation": float(
                    norm
                ),
                "effective_mass": (
                    float(mass)
                    if np.isfinite(mass)
                    else ""
                ),
            }
        )

    with correlator_path.open(
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
        writer.writerows(rows)

    plt.figure()
    plt.plot(
        correlator.lags,
        correlator.correlation,
        marker="o",
        label="Connected correlator",
    )

    if fit is not None:
        dense_lags = np.linspace(
            fit.fit_start,
            fit.fit_stop - 1,
            200,
        )

        dense_model = periodic_cosh_correlator(
            lag=dense_lags,
            amplitude=fit.amplitude,
            mass=fit.mass,
            temporal_extent=correlator.temporal_extent,
        )

        plt.plot(
            dense_lags,
            dense_model,
            label="Periodic cosh fit",
        )

    plt.xlabel("Euclidean time lag")
    plt.ylabel("Connected correlation")
    plt.title(
        "Scalar Glueball-Style Periodic Cosh Analysis"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "glueball_periodic_cosh_fit.png",
        dpi=200,
    )

    plt.figure()
    finite_mask = np.isfinite(
        effective_mass
    )

    plt.plot(
        correlator.lags[finite_mask],
        effective_mass[finite_mask],
        marker="o",
    )

    if len(plateaus) > 0:
        best = plateaus[0]

        plt.axhline(
            best.mean_mass,
            linestyle="--",
            label=(
                "Best heuristic plateau mean"
            ),
        )

        plt.axvspan(
            best.start,
            best.stop - 1,
            alpha=0.2,
        )

        plt.legend()

    plt.xlabel("Euclidean time lag")
    plt.ylabel("Arccosh effective mass")
    plt.title(
        "Scalar Glueball-Style Effective Mass"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir
        / "glueball_arccosh_effective_mass.png",
        dpi=200,
    )

    if bootstrap is not None:
        bootstrap_path = (
            data_dir
            / "glueball_bootstrap_masses.csv"
        )

        with bootstrap_path.open(
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["bootstrap_mass"]
            )

            for mass in bootstrap.bootstrap_masses:
                writer.writerow(
                    [float(mass)]
                )

        plt.figure()
        plt.hist(
            bootstrap.bootstrap_masses,
            bins=30,
        )
        plt.xlabel("Fitted lattice mass")
        plt.ylabel("Bootstrap count")
        plt.title(
            "Configuration-Bootstrap Mass Distribution"
        )
        plt.tight_layout()
        plt.savefig(
            figures_dir
            / "glueball_bootstrap_mass_distribution.png",
            dpi=200,
        )

        print()
        print(
            f"Saved bootstrap masses: {bootstrap_path}"
        )

    print()
    print(f"Saved spectroscopy data: {correlator_path}")
    print("Saved figures:")
    print(
        "results/figures/"
        "glueball_periodic_cosh_fit.png"
    )
    print(
        "results/figures/"
        "glueball_arccosh_effective_mass.png"
    )

    if bootstrap is not None:
        print(
            "results/figures/"
            "glueball_bootstrap_mass_distribution.png"
        )


if __name__ == "__main__":
    main()
