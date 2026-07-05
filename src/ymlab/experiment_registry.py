"""
Small validated experiment registry for the YMLab command-line interface.

The registry intentionally begins with compact experiments that execute through
the generic configuration-and-manifest infrastructure.

The final research campaign can build on the same interface without requiring
every historical exploratory script to be rewritten at once.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    Callable,
)

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
    generic_wilson_action,
)
from ymlab.group_interface import (
    su2_group,
    su3_group,
)
from ymlab.lattice import (
    Lattice,
)
from ymlab.monte_carlo import (
    metropolis_sweep,
)
from ymlab.plaquette import (
    average_plaquette,
)
from ymlab.reproducibility import (
    RunManifest,
    create_run_manifest,
    default_run_directory,
    finalize_manifest,
    register_output_file,
    write_json,
    write_manifest,
)
from ymlab.wilson_action import (
    number_of_plaquettes,
    wilson_action,
)


@dataclass(frozen=True)
class PlaquetteExperimentConfig:
    """Validated configuration for compact plaquette-chain experiments."""

    shape: tuple[int, ...]
    beta: float
    epsilon: float
    thermalization_sweeps: int
    measurement_sweeps: int
    seed: int

    def __post_init__(
        self,
    ) -> None:
        shape = tuple(
            int(
                extent
            )
            for extent in self.shape
        )

        if len(
            shape
        ) < 2:
            raise ValueError(
                "shape must contain at least two lattice dimensions."
            )

        if any(
            extent <= 0
            for extent in shape
        ):
            raise ValueError(
                "all lattice extents must be positive."
            )

        if self.beta < 0.0:
            raise ValueError(
                "beta must be nonnegative."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive."
            )

        if self.thermalization_sweeps < 0:
            raise ValueError(
                "thermalization_sweeps must be nonnegative."
            )

        if self.measurement_sweeps < 2:
            raise ValueError(
                "measurement_sweeps must be at least two."
            )

        object.__setattr__(
            self,
            "shape",
            shape,
        )


@dataclass(frozen=True)
class ExperimentDefinition:
    """Registered experiment metadata."""

    name: str
    description: str
    runner: Callable[
        [
            PlaquetteExperimentConfig,
            Path,
            RunManifest,
        ],
        dict[str, Any],
    ]


def _write_measurement_csv(
    run_directory: Path,
    plaquette_values: np.ndarray,
    action_density_values: np.ndarray,
    acceptance_rates: np.ndarray,
) -> Path:
    """Write aligned compact measurement columns."""
    output_path = (
        run_directory
        / "measurements.csv"
    )

    rows = [
        "sweep,average_plaquette,action_per_plaquette,acceptance_rate"
    ]

    for index in range(
        len(
            plaquette_values
        )
    ):
        rows.append(
            f"{index},"
            f"{plaquette_values[index]:.17g},"
            f"{action_density_values[index]:.17g},"
            f"{acceptance_rates[index]:.17g}"
        )

    output_path.write_text(
        "\n".join(
            rows
        )
        + "\n"
    )

    return output_path


def run_su2_plaquette_experiment(
    configuration: PlaquetteExperimentConfig,
    run_directory: Path,
    manifest: RunManifest,
) -> dict[str, Any]:
    """Execute one compact SU(2) plaquette-chain experiment."""
    lattice = Lattice(
        shape=configuration.shape,
        cold_start=True,
        seed=configuration.seed,
    )

    thermal_acceptance = []

    for _ in range(
        configuration.thermalization_sweeps
    ):
        thermal_acceptance.append(
            metropolis_sweep(
                lattice=lattice,
                beta=configuration.beta,
                epsilon=configuration.epsilon,
            )
        )

    plaquette_values = []
    action_density_values = []
    acceptance_rates = []

    plaquette_count = number_of_plaquettes(
        lattice
    )

    for _ in range(
        configuration.measurement_sweeps
    ):
        acceptance = metropolis_sweep(
            lattice=lattice,
            beta=configuration.beta,
            epsilon=configuration.epsilon,
        )

        acceptance_rates.append(
            acceptance
        )

        plaquette_values.append(
            average_plaquette(
                lattice
            )
        )

        action_density_values.append(
            wilson_action(
                lattice,
                beta=configuration.beta,
            )
            / plaquette_count
        )

    plaquette_values = np.asarray(
        plaquette_values,
        dtype=float,
    )

    action_density_values = np.asarray(
        action_density_values,
        dtype=float,
    )

    acceptance_rates = np.asarray(
        acceptance_rates,
        dtype=float,
    )

    diagnostics = diagnose_autocorrelation(
        plaquette_values
    )

    measurement_path = _write_measurement_csv(
        run_directory=run_directory,
        plaquette_values=plaquette_values,
        action_density_values=action_density_values,
        acceptance_rates=acceptance_rates,
    )

    register_output_file(
        manifest=manifest,
        path=measurement_path,
        run_directory=run_directory,
    )

    return {
        "gauge_group": "SU(2)",
        "shape": list(
            configuration.shape
        ),
        "beta": configuration.beta,
        "epsilon": configuration.epsilon,
        "mean_thermal_acceptance": (
            float(
                np.mean(
                    thermal_acceptance
                )
            )
            if len(
                thermal_acceptance
            ) > 0
            else None
        ),
        "mean_measurement_acceptance": float(
            np.mean(
                acceptance_rates
            )
        ),
        "mean_average_plaquette": float(
            np.mean(
                plaquette_values
            )
        ),
        "mean_action_per_plaquette": float(
            np.mean(
                action_density_values
            )
        ),
        "plaquette_tau_int": float(
            diagnostics.integrated_autocorrelation_time
        ),
        "plaquette_effective_sample_size": float(
            diagnostics.effective_sample_size
        ),
    }


def run_su3_plaquette_experiment(
    configuration: PlaquetteExperimentConfig,
    run_directory: Path,
    manifest: RunManifest,
) -> dict[str, Any]:
    """Execute one compact generic SU(3) plaquette-chain experiment."""
    lattice = GaugeLattice(
        shape=configuration.shape,
        group=su3_group(),
        cold_start=True,
        seed=configuration.seed,
    )

    thermal_acceptance = []

    for _ in range(
        configuration.thermalization_sweeps
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=configuration.beta,
            epsilon=configuration.epsilon,
        )

        thermal_acceptance.append(
            result.acceptance_rate
        )

    plaquette_values = []
    action_density_values = []
    acceptance_rates = []

    plaquette_count = (
        generic_number_of_plaquettes(
            lattice
        )
    )

    for _ in range(
        configuration.measurement_sweeps
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=configuration.beta,
            epsilon=configuration.epsilon,
        )

        acceptance_rates.append(
            result.acceptance_rate
        )

        plaquette_values.append(
            generic_average_plaquette(
                lattice
            )
        )

        action_density_values.append(
            generic_wilson_action(
                lattice=lattice,
                beta=configuration.beta,
            )
            / plaquette_count
        )

    plaquette_values = np.asarray(
        plaquette_values,
        dtype=float,
    )

    action_density_values = np.asarray(
        action_density_values,
        dtype=float,
    )

    acceptance_rates = np.asarray(
        acceptance_rates,
        dtype=float,
    )

    diagnostics = diagnose_autocorrelation(
        plaquette_values
    )

    measurement_path = _write_measurement_csv(
        run_directory=run_directory,
        plaquette_values=plaquette_values,
        action_density_values=action_density_values,
        acceptance_rates=acceptance_rates,
    )

    register_output_file(
        manifest=manifest,
        path=measurement_path,
        run_directory=run_directory,
    )

    return {
        "gauge_group": "SU(3)",
        "shape": list(
            configuration.shape
        ),
        "beta": configuration.beta,
        "epsilon": configuration.epsilon,
        "mean_thermal_acceptance": (
            float(
                np.mean(
                    thermal_acceptance
                )
            )
            if len(
                thermal_acceptance
            ) > 0
            else None
        ),
        "mean_measurement_acceptance": float(
            np.mean(
                acceptance_rates
            )
        ),
        "mean_average_plaquette": float(
            np.mean(
                plaquette_values
            )
        ),
        "mean_action_per_plaquette": float(
            np.mean(
                action_density_values
            )
        ),
        "plaquette_tau_int": float(
            diagnostics.integrated_autocorrelation_time
        ),
        "plaquette_effective_sample_size": float(
            diagnostics.effective_sample_size
        ),
    }


EXPERIMENTS: dict[
    str,
    ExperimentDefinition,
] = {
    "su2-plaquette": ExperimentDefinition(
        name="su2-plaquette",
        description=(
            "Compact SU(2) average-plaquette and Wilson-action chain."
        ),
        runner=run_su2_plaquette_experiment,
    ),
    "su3-plaquette": ExperimentDefinition(
        name="su3-plaquette",
        description=(
            "Compact generic SU(3) average-plaquette and Wilson-action chain."
        ),
        runner=run_su3_plaquette_experiment,
    ),
}


def list_experiments(
) -> tuple[
    ExperimentDefinition,
    ...,
]:
    """Return registered experiments in deterministic name order."""
    return tuple(
        EXPERIMENTS[
            name
        ]
        for name in sorted(
            EXPERIMENTS
        )
    )


def get_experiment(
    name: str,
) -> ExperimentDefinition:
    """Return one registered experiment."""
    try:
        return EXPERIMENTS[
            name
        ]
    except KeyError as error:
        raise ValueError(
            f"Unknown experiment: {name}"
        ) from error


def execute_registered_experiment(
    name: str,
    configuration: PlaquetteExperimentConfig,
    output_root: str | Path = "results/runs",
    repository_root: str | Path = ".",
) -> tuple[
    Path,
    RunManifest,
    dict[str, Any],
]:
    """
    Execute one registered experiment and write manifest and summary outputs.
    """
    definition = get_experiment(
        name
    )

    configuration_dictionary = asdict(
        configuration
    )

    configuration_dictionary[
        "shape"
    ] = list(
        configuration.shape
    )

    manifest = create_run_manifest(
        experiment_name=definition.name,
        configuration=configuration_dictionary,
        repository_root=repository_root,
    )

    run_directory = default_run_directory(
        manifest=manifest,
        root=output_root,
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

    start = perf_counter()

    try:
        summary = definition.runner(
            configuration,
            run_directory,
            manifest,
        )

        runtime = (
            perf_counter()
            - start
        )

        summary_path = (
            run_directory
            / "summary.json"
        )

        write_json(
            path=summary_path,
            value=summary,
        )

        register_output_file(
            manifest=manifest,
            path=summary_path,
            run_directory=run_directory,
        )

        finalize_manifest(
            manifest=manifest,
            runtime_seconds=runtime,
            status="completed",
        )

        write_manifest(
            manifest=manifest,
            path=manifest_path,
        )

        return (
            run_directory,
            manifest,
            summary,
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
