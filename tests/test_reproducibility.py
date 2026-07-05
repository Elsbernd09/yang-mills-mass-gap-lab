import json

import pytest

from ymlab.reproducibility import (
    configuration_hash,
    create_run_manifest,
    finalize_manifest,
    load_manifest,
    register_output_file,
    write_manifest,
)


def test_configuration_hash_is_deterministic():
    first = {
        "beta": 2.0,
        "shape": [
            4,
            4,
        ],
        "seed": 2026,
    }

    second = {
        "seed": 2026,
        "shape": [
            4,
            4,
        ],
        "beta": 2.0,
    }

    assert configuration_hash(
        first
    ) == configuration_hash(
        second
    )


def test_configuration_hash_changes_with_configuration():
    first = {
        "beta": 2.0,
    }

    second = {
        "beta": 2.1,
    }

    assert configuration_hash(
        first
    ) != configuration_hash(
        second
    )


def test_manifest_contains_environment_metadata():
    manifest = create_run_manifest(
        experiment_name="test-experiment",
        configuration={
            "seed": 2026,
        },
    )

    assert manifest.experiment_name == (
        "test-experiment"
    )

    assert manifest.environment.python_version

    assert manifest.environment.numpy_version

    assert manifest.configuration_hash


def test_manifest_round_trip(tmp_path):
    manifest = create_run_manifest(
        experiment_name="round-trip",
        configuration={
            "beta": 2.0,
            "seed": 2026,
        },
    )

    finalize_manifest(
        manifest=manifest,
        runtime_seconds=1.25,
        status="completed",
    )

    output = (
        tmp_path
        / "manifest.json"
    )

    write_manifest(
        manifest=manifest,
        path=output,
    )

    loaded = load_manifest(
        output
    )

    assert loaded[
        "experiment_name"
    ] == "round-trip"

    assert loaded[
        "runtime_seconds"
    ] == pytest.approx(
        1.25
    )

    assert loaded[
        "status"
    ] == "completed"


def test_register_output_file_is_unique(tmp_path):
    manifest = create_run_manifest(
        experiment_name="outputs",
        configuration={
            "seed": 1,
        },
    )

    run_directory = (
        tmp_path
        / "run"
    )

    run_directory.mkdir()

    output = (
        run_directory
        / "summary.json"
    )

    output.write_text(
        json.dumps(
            {
                "value": 1
            }
        )
    )

    register_output_file(
        manifest=manifest,
        path=output,
        run_directory=run_directory,
    )

    register_output_file(
        manifest=manifest,
        path=output,
        run_directory=run_directory,
    )

    assert manifest.output_files == [
        "summary.json"
    ]


def test_negative_runtime_rejected():
    manifest = create_run_manifest(
        experiment_name="runtime",
        configuration={
            "seed": 1,
        },
    )

    with pytest.raises(ValueError):
        finalize_manifest(
            manifest=manifest,
            runtime_seconds=-1.0,
        )


def test_short_configuration_hash_rejected():
    with pytest.raises(ValueError):
        configuration_hash(
            {
                "beta": 2.0,
            },
            length=4,
        )
