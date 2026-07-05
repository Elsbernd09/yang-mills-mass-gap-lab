"""
Final controlled SU(2) overrelaxation-efficiency research campaign.

Research question:

How does microcanonical overrelaxation frequency affect effective sample size
per wall-clock second for the average plaquette and a scalar glueball-style
composite observable across finite bare-coupling regimes?

The campaign uses paired thermalized starting configurations for each
beta-seed pair.

No winner is selected before the experiment.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
)
from pathlib import Path
import csv
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ymlab.reproducibility import (
    create_run_manifest,
    default_run_directory,
    finalize_manifest,
    register_output_file,
    write_json,
    write_manifest,
)
from ymlab.research_campaign import (
    OverrelaxationCampaignConfig,
    records_as_dicts,
    run_overrelaxation_campaign,
    summarize_campaign,
    summaries_as_dicts,
)


def write_csv(
    path,
    rows,
):
    if len(
        rows
    ) == 0:
        raise ValueError(
            "Cannot write empty CSV rows."
        )

    with Path(
        path
    ).open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main() -> None:
    configuration = (
        OverrelaxationCampaignConfig(
            shape=(
                5,
                4,
                4,
            ),
            betas=(
                1.5,
                2.0,
                2.5,
            ),
            seeds=(
                2026,
                2027,
                2028,
            ),
            schedules=(
                0,
                1,
                2,
                4,
            ),
            epsilon=0.18,
            thermalization_sweeps=60,
            measurement_updates=180,
            time_direction=0,
        )
    )

    configuration_dictionary = asdict(
        configuration
    )

    configuration_dictionary[
        "shape"
    ] = list(
        configuration.shape
    )

    configuration_dictionary[
        "betas"
    ] = list(
        configuration.betas
    )

    configuration_dictionary[
        "seeds"
    ] = list(
        configuration.seeds
    )

    configuration_dictionary[
        "schedules"
    ] = list(
        configuration.schedules
    )

    manifest = create_run_manifest(
        experiment_name=(
            "final-overrelaxation-efficiency-campaign"
        ),
        configuration=configuration_dictionary,
        repository_root=".",
    )

    run_directory = default_run_directory(
        manifest=manifest,
        root="results/runs",
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_path = (
        run_directory
        / "manifest.json"
    )

    write_manifest(
        manifest=manifest,
        path=manifest_path,
    )

    print(
        "Final Controlled Overrelaxation Efficiency Campaign"
    )

    print(
        "=" * 150
    )

    print(
        "Research question:"
    )

    print(
        "How does SU(2) microcanonical overrelaxation frequency affect "
        "ESS/sec for plaquette and scalar composite observables across "
        "finite bare-coupling regimes?"
    )

    print()

    print(
        f"Shape:                         "
        f"{configuration.shape}"
    )

    print(
        f"Betas:                         "
        f"{configuration.betas}"
    )

    print(
        f"Seeds:                         "
        f"{configuration.seeds}"
    )

    print(
        f"Schedules:                     "
        f"{configuration.schedules}"
    )

    print(
        f"Thermalization sweeps:         "
        f"{configuration.thermalization_sweeps}"
    )

    print(
        f"Measurement updates:           "
        f"{configuration.measurement_updates}"
    )

    print(
        f"Run directory:                 "
        f"{run_directory}"
    )

    print()

    start = perf_counter()

    try:
        records = run_overrelaxation_campaign(
            configuration=configuration,
            progress=True,
        )

        summaries = summarize_campaign(
            records
        )

        runtime = (
            perf_counter()
            - start
        )

        record_rows = records_as_dicts(
            records
        )

        summary_rows = summaries_as_dicts(
            summaries
        )

        records_path = (
            run_directory
            / "campaign_records.csv"
        )

        summary_path = (
            run_directory
            / "campaign_summary.csv"
        )

        write_csv(
            records_path,
            record_rows,
        )

        write_csv(
            summary_path,
            summary_rows,
        )

        register_output_file(
            manifest=manifest,
            path=records_path,
            run_directory=run_directory,
        )

        register_output_file(
            manifest=manifest,
            path=summary_path,
            run_directory=run_directory,
        )

        print()

        print(
            "Aggregate campaign summary"
        )

        print(
            "-" * 150
        )

        print(
            f"{'Beta':>8}"
            f"{'Schedule':>12}"
            f"{'Plaq tau':>14}"
            f"{'Plaq ESS/s':>16}"
            f"{'Plaq ratio':>16}"
            f"{'Scalar tau':>14}"
            f"{'Scalar ESS/s':>16}"
            f"{'Scalar ratio':>16}"
            f"{'Runtime':>14}"
        )

        print(
            "-" * 150
        )

        for summary in summaries:
            print(
                f"{summary.beta:>8.3f}"
                f"{summary.schedule_label:>12}"
                f"{summary.mean_plaquette_tau_int:>14.6f}"
                f"{summary.mean_plaquette_ess_per_second:>16.6f}"
                f"{summary.mean_plaquette_efficiency_ratio_vs_metropolis:>16.6f}"
                f"{summary.mean_scalar_tau_int:>14.6f}"
                f"{summary.mean_scalar_ess_per_second:>16.6f}"
                f"{summary.mean_scalar_efficiency_ratio_vs_metropolis:>16.6f}"
                f"{summary.mean_runtime_seconds:>14.6f}"
            )

        best_by_beta = {}

        for beta in configuration.betas:
            selected = [
                summary
                for summary in summaries
                if summary.beta == beta
            ]

            best_plaquette = max(
                selected,
                key=lambda summary: (
                    summary.mean_plaquette_ess_per_second
                ),
            )

            best_scalar = max(
                selected,
                key=lambda summary: (
                    summary.mean_scalar_ess_per_second
                ),
            )

            best_by_beta[
                str(
                    beta
                )
            ] = {
                "best_plaquette_schedule": (
                    best_plaquette.schedule_label
                ),
                "best_plaquette_ess_per_second": (
                    best_plaquette.mean_plaquette_ess_per_second
                ),
                "best_plaquette_efficiency_ratio_vs_metropolis": (
                    best_plaquette
                    .mean_plaquette_efficiency_ratio_vs_metropolis
                ),
                "best_scalar_schedule": (
                    best_scalar.schedule_label
                ),
                "best_scalar_ess_per_second": (
                    best_scalar.mean_scalar_ess_per_second
                ),
                "best_scalar_efficiency_ratio_vs_metropolis": (
                    best_scalar
                    .mean_scalar_efficiency_ratio_vs_metropolis
                ),
            }

        plaquette_wins = {}

        scalar_wins = {}

        for result in best_by_beta.values():
            plaquette_label = result[
                "best_plaquette_schedule"
            ]

            scalar_label = result[
                "best_scalar_schedule"
            ]

            plaquette_wins[
                plaquette_label
            ] = (
                plaquette_wins.get(
                    plaquette_label,
                    0,
                )
                + 1
            )

            scalar_wins[
                scalar_label
            ] = (
                scalar_wins.get(
                    scalar_label,
                    0,
                )
                + 1
            )

        conclusion = {
            "research_question": (
                "How does SU(2) microcanonical overrelaxation frequency "
                "affect ESS/sec for the average plaquette and a scalar "
                "glueball-style composite observable across finite bare "
                "Wilson couplings?"
            ),
            "scope": (
                "Finite 3D SU(2) Wilson lattice computational-methodology "
                "study; no continuum or infinite-volume claim."
            ),
            "best_schedule_by_beta": best_by_beta,
            "plaquette_schedule_win_counts": (
                plaquette_wins
            ),
            "scalar_schedule_win_counts": (
                scalar_wins
            ),
            "interpretation_rule": (
                "A schedule is described as more efficient only where its "
                "measured mean ESS/sec exceeds the paired Metropolis baseline. "
                "The campaign does not infer formal significance from three "
                "independent seeds."
            ),
        }

        conclusion_path = (
            run_directory
            / "campaign_conclusion.json"
        )

        write_json(
            path=conclusion_path,
            value=conclusion,
        )

        register_output_file(
            manifest=manifest,
            path=conclusion_path,
            run_directory=run_directory,
        )

        figures_directory = (
            run_directory
            / "figures"
        )

        figures_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for observable in (
            "plaquette",
            "scalar",
        ):
            plt.figure()

            for schedule in configuration.schedules:
                selected = [
                    summary
                    for summary in summaries
                    if (
                        summary.overrelaxation_sweeps
                        == schedule
                    )
                ]

                selected = sorted(
                    selected,
                    key=lambda summary: (
                        summary.beta
                    ),
                )

                betas = np.asarray(
                    [
                        summary.beta
                        for summary in selected
                    ],
                    dtype=float,
                )

                if observable == "plaquette":
                    values = np.asarray(
                        [
                            summary.mean_plaquette_ess_per_second
                            for summary in selected
                        ],
                        dtype=float,
                    )

                    errors = np.asarray(
                        [
                            summary.std_plaquette_ess_per_second
                            for summary in selected
                        ],
                        dtype=float,
                    )

                    ylabel = (
                        "Plaquette effective sample size per second"
                    )
                else:
                    values = np.asarray(
                        [
                            summary.mean_scalar_ess_per_second
                            for summary in selected
                        ],
                        dtype=float,
                    )

                    errors = np.asarray(
                        [
                            summary.std_scalar_ess_per_second
                            for summary in selected
                        ],
                        dtype=float,
                    )

                    ylabel = (
                        "Scalar effective sample size per second"
                    )

                plt.errorbar(
                    betas,
                    values,
                    yerr=errors,
                    marker="o",
                    label=selected[
                        0
                    ].schedule_label,
                )

            plt.xlabel(
                "Bare Wilson beta"
            )

            plt.ylabel(
                ylabel
            )

            plt.title(
                f"{observable.capitalize()} Sampling Efficiency Across Beta"
            )

            plt.legend()

            plt.tight_layout()

            figure_path = (
                figures_directory
                / (
                    f"{observable}_ess_per_second_vs_beta.png"
                )
            )

            plt.savefig(
                figure_path,
                dpi=220,
            )

            plt.close()

            register_output_file(
                manifest=manifest,
                path=figure_path,
                run_directory=run_directory,
            )

        plt.figure()

        for schedule in configuration.schedules:
            selected = [
                summary
                for summary in summaries
                if (
                    summary.overrelaxation_sweeps
                    == schedule
                )
            ]

            selected = sorted(
                selected,
                key=lambda summary: (
                    summary.beta
                ),
            )

            plt.plot(
                [
                    summary.beta
                    for summary in selected
                ],
                [
                    summary.mean_plaquette_efficiency_ratio_vs_metropolis
                    for summary in selected
                ],
                marker="o",
                label=selected[
                    0
                ].schedule_label,
            )

        plt.axhline(
            1.0,
            linestyle="--",
        )

        plt.xlabel(
            "Bare Wilson beta"
        )

        plt.ylabel(
            "Plaquette ESS/sec ratio to paired Metropolis"
        )

        plt.title(
            "Paired Plaquette Efficiency Ratio"
        )

        plt.legend()

        plt.tight_layout()

        ratio_path = (
            figures_directory
            / "plaquette_efficiency_ratio_vs_beta.png"
        )

        plt.savefig(
            ratio_path,
            dpi=220,
        )

        plt.close()

        register_output_file(
            manifest=manifest,
            path=ratio_path,
            run_directory=run_directory,
        )

        finalize_manifest(
            manifest=manifest,
            runtime_seconds=runtime,
            status="completed",
            notes=[
                (
                    "Controlled paired finite-lattice overrelaxation "
                    "efficiency campaign."
                ),
                (
                    "Three independent seeds are reported descriptively; "
                    "no formal significance claim is made."
                ),
            ],
        )

        write_manifest(
            manifest=manifest,
            path=manifest_path,
        )

        print()

        print(
            "Campaign result by beta"
        )

        print(
            "-" * 150
        )

        for beta, result in best_by_beta.items():
            print(
                f"beta={beta}: "
                f"best plaquette={result['best_plaquette_schedule']}, "
                f"best scalar={result['best_scalar_schedule']}"
            )

        print()

        print(
            "Plaquette schedule win counts:"
        )

        print(
            plaquette_wins
        )

        print(
            "Scalar schedule win counts:"
        )

        print(
            scalar_wins
        )

        print()

        print(
            f"Campaign runtime seconds: {runtime:.6f}"
        )

        print(
            f"Run directory: {run_directory}"
        )

        print(
            "FINAL CAMPAIGN STATUS: COMPLETED"
        )

    except Exception as error:
        runtime = (
            perf_counter()
            - start
        )

        finalize_manifest(
            manifest=manifest,
            runtime_seconds=runtime,
            status="failed",
            notes=[
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                )
            ],
        )

        write_manifest(
            manifest=manifest,
            path=manifest_path,
        )

        raise


if __name__ == "__main__":
    main()
