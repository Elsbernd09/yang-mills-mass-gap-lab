"""
Controlled SU(2) overrelaxation-efficiency research campaign.

Research question
-----------------

How does microcanonical overrelaxation frequency alter effective sample size
per wall-clock second for two finite-lattice observables across multiple bare
Wilson couplings?

Schedules
---------

    M
    M + 1 OR
    M + 2 OR
    M + 4 OR

where M is one Metropolis sweep and OR is one SU(2) microcanonical
overrelaxation sweep.

For every beta and random seed:

1. one lattice is thermalized using Metropolis,
2. the thermalized configuration is copied,
3. every update schedule begins from the same configuration,
4. every schedule uses the same number of measured hybrid updates,
5. plaquette and scalar composite time series are measured,
6. integrated autocorrelation time and effective sample size are estimated,
7. wall-clock runtime is measured,
8. ESS per second is calculated.

The paired starting-configuration design reduces one source of baseline
configuration variation when schedules are compared within a beta-seed pair.

The campaign remains a finite-lattice computational methodology study.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from time import perf_counter
from typing import Iterable

import numpy as np

from ymlab.diagnostics import (
    diagnose_autocorrelation,
)
from ymlab.glueball import (
    scalar_glueball_time_series,
)
from ymlab.lattice import (
    Lattice,
)
from ymlab.monte_carlo import (
    metropolis_sweep,
)
from ymlab.overrelaxation import (
    hybrid_metropolis_overrelaxation_sweep,
)
from ymlab.plaquette import (
    average_plaquette,
)


@dataclass(frozen=True)
class OverrelaxationCampaignConfig:
    """Validated controlled-campaign configuration."""

    shape: tuple[int, ...]
    betas: tuple[float, ...]
    seeds: tuple[int, ...]
    schedules: tuple[int, ...]
    epsilon: float
    thermalization_sweeps: int
    measurement_updates: int
    time_direction: int = 0

    def __post_init__(
        self,
    ) -> None:
        shape = tuple(
            int(
                extent
            )
            for extent in self.shape
        )

        betas = tuple(
            float(
                beta
            )
            for beta in self.betas
        )

        seeds = tuple(
            int(
                seed
            )
            for seed in self.seeds
        )

        schedules = tuple(
            int(
                schedule
            )
            for schedule in self.schedules
        )

        if len(
            shape
        ) < 3:
            raise ValueError(
                "Scalar composite campaign requires at least three dimensions."
            )

        if any(
            extent <= 0
            for extent in shape
        ):
            raise ValueError(
                "All lattice extents must be positive."
            )

        if len(
            betas
        ) == 0:
            raise ValueError(
                "At least one beta value is required."
            )

        if any(
            beta < 0.0
            for beta in betas
        ):
            raise ValueError(
                "Beta values must be nonnegative."
            )

        if len(
            seeds
        ) == 0:
            raise ValueError(
                "At least one random seed is required."
            )

        if len(
            schedules
        ) == 0:
            raise ValueError(
                "At least one update schedule is required."
            )

        if any(
            schedule < 0
            for schedule in schedules
        ):
            raise ValueError(
                "Overrelaxation counts must be nonnegative."
            )

        if len(
            set(
                schedules
            )
        ) != len(
            schedules
        ):
            raise ValueError(
                "Update schedules must be unique."
            )

        if 0 not in schedules:
            raise ValueError(
                "Campaign must include Metropolis-only schedule zero."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive."
            )

        if self.thermalization_sweeps < 0:
            raise ValueError(
                "thermalization_sweeps must be nonnegative."
            )

        if self.measurement_updates < 4:
            raise ValueError(
                "measurement_updates must be at least four."
            )

        if (
            self.time_direction < 0
            or self.time_direction >= len(
                shape
            )
        ):
            raise ValueError(
                "time_direction is invalid."
            )

        object.__setattr__(
            self,
            "shape",
            shape,
        )

        object.__setattr__(
            self,
            "betas",
            betas,
        )

        object.__setattr__(
            self,
            "seeds",
            seeds,
        )

        object.__setattr__(
            self,
            "schedules",
            schedules,
        )


@dataclass(frozen=True)
class CampaignRecord:
    """One beta-seed-schedule chain result."""

    beta: float
    seed: int
    overrelaxation_sweeps: int
    schedule_label: str
    runtime_seconds: float
    mean_acceptance_rate: float
    mean_plaquette: float
    plaquette_tau_int: float
    plaquette_ess: float
    plaquette_ess_per_second: float
    scalar_mean: float
    scalar_tau_int: float
    scalar_ess: float
    scalar_ess_per_second: float
    maximum_microcanonical_error: float


@dataclass(frozen=True)
class CampaignSummary:
    """Aggregated schedule result at one beta."""

    beta: float
    overrelaxation_sweeps: int
    schedule_label: str
    chain_count: int
    mean_runtime_seconds: float
    mean_acceptance_rate: float
    mean_plaquette: float
    mean_plaquette_tau_int: float
    mean_plaquette_ess_per_second: float
    std_plaquette_ess_per_second: float
    mean_scalar_tau_int: float
    mean_scalar_ess_per_second: float
    std_scalar_ess_per_second: float
    mean_plaquette_efficiency_ratio_vs_metropolis: float
    mean_scalar_efficiency_ratio_vs_metropolis: float


def schedule_label(
    overrelaxation_sweeps: int,
) -> str:
    """Return a compact schedule label."""
    if overrelaxation_sweeps == 0:
        return "M"

    return (
        "M+"
        f"{overrelaxation_sweeps}"
        "OR"
    )


def copy_su2_lattice(
    lattice: Lattice,
) -> Lattice:
    """Return an independent SU(2) lattice copy."""
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
                np.asarray(
                    lattice.get_link(
                        site,
                        mu,
                    ),
                    dtype=complex,
                ).copy(),
            )

    return copied


def thermalize_starting_configuration(
    shape: tuple[int, ...],
    beta: float,
    epsilon: float,
    thermalization_sweeps: int,
    seed: int,
) -> Lattice:
    """Construct one shared thermalized starting configuration."""
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


def run_schedule_chain(
    starting_lattice: Lattice,
    beta: float,
    epsilon: float,
    overrelaxation_sweeps: int,
    measurement_updates: int,
    time_direction: int,
    seed: int,
) -> CampaignRecord:
    """Run one measured hybrid schedule from a copied starting field."""
    lattice = copy_su2_lattice(
        starting_lattice
    )

    plaquette_series = []
    scalar_series = []
    acceptance_rates = []

    maximum_microcanonical_error = 0.0

    start = perf_counter()

    for _ in range(
        measurement_updates
    ):
        update = (
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
            update.metropolis_acceptance_rate
        )

        maximum_microcanonical_error = max(
            maximum_microcanonical_error,
            update.maximum_local_action_error,
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

    runtime_seconds = (
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

    return CampaignRecord(
        beta=float(
            beta
        ),
        seed=int(
            seed
        ),
        overrelaxation_sweeps=int(
            overrelaxation_sweeps
        ),
        schedule_label=schedule_label(
            overrelaxation_sweeps
        ),
        runtime_seconds=float(
            runtime_seconds
        ),
        mean_acceptance_rate=float(
            np.mean(
                acceptance_rates
            )
        ),
        mean_plaquette=float(
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
            / runtime_seconds
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
            / runtime_seconds
        ),
        maximum_microcanonical_error=float(
            maximum_microcanonical_error
        ),
    )


def run_overrelaxation_campaign(
    configuration: OverrelaxationCampaignConfig,
    progress: bool = True,
) -> tuple[
    CampaignRecord,
    ...,
]:
    """Run the complete paired controlled campaign."""
    records = []

    total_conditions = (
        len(
            configuration.betas
        )
        * len(
            configuration.seeds
        )
        * len(
            configuration.schedules
        )
    )

    completed = 0

    for beta in configuration.betas:
        for seed in configuration.seeds:
            if progress:
                print()
                print(
                    "Thermalizing shared start for "
                    f"beta={beta:.4f}, seed={seed}..."
                )

            starting_lattice = (
                thermalize_starting_configuration(
                    shape=configuration.shape,
                    beta=beta,
                    epsilon=configuration.epsilon,
                    thermalization_sweeps=(
                        configuration.thermalization_sweeps
                    ),
                    seed=seed,
                )
            )

            for schedule in configuration.schedules:
                if progress:
                    print(
                        "  running "
                        f"{schedule_label(schedule):<6} "
                        f"condition {completed + 1}/"
                        f"{total_conditions}"
                    )

                record = run_schedule_chain(
                    starting_lattice=starting_lattice,
                    beta=beta,
                    epsilon=configuration.epsilon,
                    overrelaxation_sweeps=schedule,
                    measurement_updates=(
                        configuration.measurement_updates
                    ),
                    time_direction=(
                        configuration.time_direction
                    ),
                    seed=seed,
                )

                records.append(
                    record
                )

                completed += 1

    return tuple(
        records
    )


def summarize_campaign(
    records: Iterable[
        CampaignRecord
    ],
) -> tuple[
    CampaignSummary,
    ...,
]:
    """Aggregate schedule efficiencies and paired ratios by beta."""
    records = tuple(
        records
    )

    if len(
        records
    ) == 0:
        raise ValueError(
            "Campaign records cannot be empty."
        )

    baseline = {
        (
            record.beta,
            record.seed,
        ): record
        for record in records
        if record.overrelaxation_sweeps == 0
    }

    summaries = []

    conditions = sorted(
        {
            (
                record.beta,
                record.overrelaxation_sweeps,
                record.schedule_label,
            )
            for record in records
        }
    )

    for (
        beta,
        overrelaxation_sweeps,
        label,
    ) in conditions:
        selected = [
            record
            for record in records
            if (
                record.beta == beta
                and record.overrelaxation_sweeps
                == overrelaxation_sweeps
            )
        ]

        plaquette_ratios = []
        scalar_ratios = []

        for record in selected:
            key = (
                record.beta,
                record.seed,
            )

            if key not in baseline:
                raise ValueError(
                    "Missing Metropolis baseline for paired comparison."
                )

            baseline_record = baseline[
                key
            ]

            plaquette_ratios.append(
                record.plaquette_ess_per_second
                / baseline_record.plaquette_ess_per_second
            )

            scalar_ratios.append(
                record.scalar_ess_per_second
                / baseline_record.scalar_ess_per_second
            )

        plaquette_efficiencies = np.asarray(
            [
                record.plaquette_ess_per_second
                for record in selected
            ],
            dtype=float,
        )

        scalar_efficiencies = np.asarray(
            [
                record.scalar_ess_per_second
                for record in selected
            ],
            dtype=float,
        )

        summaries.append(
            CampaignSummary(
                beta=float(
                    beta
                ),
                overrelaxation_sweeps=int(
                    overrelaxation_sweeps
                ),
                schedule_label=label,
                chain_count=len(
                    selected
                ),
                mean_runtime_seconds=float(
                    np.mean(
                        [
                            record.runtime_seconds
                            for record in selected
                        ]
                    )
                ),
                mean_acceptance_rate=float(
                    np.mean(
                        [
                            record.mean_acceptance_rate
                            for record in selected
                        ]
                    )
                ),
                mean_plaquette=float(
                    np.mean(
                        [
                            record.mean_plaquette
                            for record in selected
                        ]
                    )
                ),
                mean_plaquette_tau_int=float(
                    np.mean(
                        [
                            record.plaquette_tau_int
                            for record in selected
                        ]
                    )
                ),
                mean_plaquette_ess_per_second=float(
                    np.mean(
                        plaquette_efficiencies
                    )
                ),
                std_plaquette_ess_per_second=(
                    float(
                        np.std(
                            plaquette_efficiencies,
                            ddof=1,
                        )
                    )
                    if len(
                        plaquette_efficiencies
                    ) > 1
                    else 0.0
                ),
                mean_scalar_tau_int=float(
                    np.mean(
                        [
                            record.scalar_tau_int
                            for record in selected
                        ]
                    )
                ),
                mean_scalar_ess_per_second=float(
                    np.mean(
                        scalar_efficiencies
                    )
                ),
                std_scalar_ess_per_second=(
                    float(
                        np.std(
                            scalar_efficiencies,
                            ddof=1,
                        )
                    )
                    if len(
                        scalar_efficiencies
                    ) > 1
                    else 0.0
                ),
                mean_plaquette_efficiency_ratio_vs_metropolis=float(
                    np.mean(
                        plaquette_ratios
                    )
                ),
                mean_scalar_efficiency_ratio_vs_metropolis=float(
                    np.mean(
                        scalar_ratios
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def records_as_dicts(
    records: Iterable[
        CampaignRecord
    ],
) -> list[
    dict
]:
    """Convert detailed records to serializable dictionaries."""
    return [
        asdict(
            record
        )
        for record in records
    ]


def summaries_as_dicts(
    summaries: Iterable[
        CampaignSummary
    ],
) -> list[
    dict
]:
    """Convert aggregate summaries to serializable dictionaries."""
    return [
        asdict(
            summary
        )
        for summary in summaries
    ]
