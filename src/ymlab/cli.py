"""
Command-line interface for reproducible YMLab experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ymlab.experiment_registry import (
    PlaquetteExperimentConfig,
    execute_registered_experiment,
    list_experiments,
)


def build_parser(
) -> argparse.ArgumentParser:
    """Construct the YMLab CLI parser."""
    parser = argparse.ArgumentParser(
        prog="ymlab",
        description=(
            "Reproducible finite-lattice Yang-Mills experiment runner."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "list",
        help="List registered experiments.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one registered experiment.",
    )

    run_parser.add_argument(
        "experiment",
        type=str,
        help="Registered experiment name.",
    )

    run_parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        required=True,
        help="Periodic lattice extents.",
    )

    run_parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Wilson coupling parameter.",
    )

    run_parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Near-identity Metropolis proposal scale.",
    )

    run_parser.add_argument(
        "--thermalization",
        type=int,
        default=20,
        help="Thermalization sweeps.",
    )

    run_parser.add_argument(
        "--measurements",
        type=int,
        default=50,
        help="Measurement sweeps.",
    )

    run_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Deterministic random seed.",
    )

    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/runs"
        ),
        help="Root directory for run outputs.",
    )

    return parser


def main(
    arguments: list[str] | None = None,
) -> int:
    """Run the CLI."""
    parser = build_parser()

    args = parser.parse_args(
        arguments
    )

    if args.command == "list":
        for experiment in list_experiments():
            print(
                f"{experiment.name:<20} "
                f"{experiment.description}"
            )

        return 0

    if args.command == "run":
        configuration = PlaquetteExperimentConfig(
            shape=tuple(
                args.shape
            ),
            beta=args.beta,
            epsilon=args.epsilon,
            thermalization_sweeps=(
                args.thermalization
            ),
            measurement_sweeps=(
                args.measurements
            ),
            seed=args.seed,
        )

        (
            run_directory,
            manifest,
            summary,
        ) = execute_registered_experiment(
            name=args.experiment,
            configuration=configuration,
            output_root=args.output_root,
            repository_root=".",
        )

        print(
            "Experiment completed."
        )

        print(
            f"Run directory: {run_directory}"
        )

        print(
            f"Configuration hash: "
            f"{manifest.configuration_hash}"
        )

        print(
            f"Status: {manifest.status}"
        )

        print(
            f"Runtime seconds: "
            f"{manifest.runtime_seconds:.6f}"
        )

        print(
            "Summary:"
        )

        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    parser.error(
        "Unsupported command."
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
