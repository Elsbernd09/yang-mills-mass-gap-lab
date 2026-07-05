import json

import pytest

from ymlab.cli import main
from ymlab.experiment_registry import (
    PlaquetteExperimentConfig,
    execute_registered_experiment,
    get_experiment,
    list_experiments,
)


def small_su2_configuration():
    return PlaquetteExperimentConfig(
        shape=(
            3,
            3,
        ),
        beta=2.0,
        epsilon=0.18,
        thermalization_sweeps=2,
        measurement_sweeps=4,
        seed=2026,
    )


def test_registry_contains_su2_and_su3():
    names = [
        experiment.name
        for experiment in list_experiments()
    ]

    assert "su2-plaquette" in names

    assert "su3-plaquette" in names


def test_unknown_experiment_rejected():
    with pytest.raises(ValueError):
        get_experiment(
            "not-a-real-experiment"
        )


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        PlaquetteExperimentConfig(
            shape=(
                4,
            ),
            beta=2.0,
            epsilon=0.18,
            thermalization_sweeps=2,
            measurement_sweeps=4,
            seed=2026,
        )


def test_registered_su2_experiment_writes_manifest_and_summary(
    tmp_path,
):
    configuration = small_su2_configuration()

    (
        run_directory,
        manifest,
        summary,
    ) = execute_registered_experiment(
        name="su2-plaquette",
        configuration=configuration,
        output_root=tmp_path,
        repository_root=".",
    )

    assert manifest.status == "completed"

    assert (
        run_directory
        / "manifest.json"
    ).exists()

    assert (
        run_directory
        / "summary.json"
    ).exists()

    assert (
        run_directory
        / "measurements.csv"
    ).exists()

    loaded_summary = json.loads(
        (
            run_directory
            / "summary.json"
        ).read_text()
    )

    assert loaded_summary[
        "gauge_group"
    ] == "SU(2)"

    assert summary[
        "mean_average_plaquette"
    ] == pytest.approx(
        loaded_summary[
            "mean_average_plaquette"
        ]
    )


def test_same_configuration_has_same_hash(
    tmp_path,
):
    configuration = small_su2_configuration()

    _, first_manifest, _ = (
        execute_registered_experiment(
            name="su2-plaquette",
            configuration=configuration,
            output_root=(
                tmp_path
                / "first"
            ),
            repository_root=".",
        )
    )

    _, second_manifest, _ = (
        execute_registered_experiment(
            name="su2-plaquette",
            configuration=configuration,
            output_root=(
                tmp_path
                / "second"
            ),
            repository_root=".",
        )
    )

    assert (
        first_manifest.configuration_hash
        == second_manifest.configuration_hash
    )


def test_cli_list(capsys):
    status = main(
        [
            "list",
        ]
    )

    captured = capsys.readouterr()

    assert status == 0

    assert "su2-plaquette" in captured.out

    assert "su3-plaquette" in captured.out


def test_cli_runs_small_su2_experiment(
    tmp_path,
    capsys,
):
    status = main(
        [
            "run",
            "su2-plaquette",
            "--shape",
            "3",
            "3",
            "--beta",
            "2.0",
            "--epsilon",
            "0.18",
            "--thermalization",
            "2",
            "--measurements",
            "4",
            "--seed",
            "2026",
            "--output-root",
            str(
                tmp_path
            ),
        ]
    )

    captured = capsys.readouterr()

    assert status == 0

    assert (
        "Experiment completed."
        in captured.out
    )

    assert (
        "Configuration hash:"
        in captured.out
    )

    run_directories = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir()
    ]

    assert len(
        run_directories
    ) == 1
