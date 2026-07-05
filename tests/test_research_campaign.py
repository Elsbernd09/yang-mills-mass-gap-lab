import numpy as np
import pytest

from ymlab.research_campaign import (
    OverrelaxationCampaignConfig,
    run_overrelaxation_campaign,
    schedule_label,
    summarize_campaign,
)


def tiny_configuration():
    return OverrelaxationCampaignConfig(
        shape=(
            3,
            3,
            3,
        ),
        betas=(
            2.0,
        ),
        seeds=(
            2026,
        ),
        schedules=(
            0,
            1,
        ),
        epsilon=0.18,
        thermalization_sweeps=2,
        measurement_updates=4,
        time_direction=0,
    )


def test_schedule_labels():
    assert schedule_label(
        0
    ) == "M"

    assert schedule_label(
        1
    ) == "M+1OR"

    assert schedule_label(
        4
    ) == "M+4OR"


def test_campaign_requires_metropolis_baseline():
    with pytest.raises(ValueError):
        OverrelaxationCampaignConfig(
            shape=(
                3,
                3,
                3,
            ),
            betas=(
                2.0,
            ),
            seeds=(
                2026,
            ),
            schedules=(
                1,
                2,
            ),
            epsilon=0.18,
            thermalization_sweeps=2,
            measurement_updates=4,
        )


def test_tiny_campaign_runs():
    records = run_overrelaxation_campaign(
        tiny_configuration(),
        progress=False,
    )

    assert len(
        records
    ) == 2

    for record in records:
        assert np.isfinite(
            record.runtime_seconds
        )

        assert record.runtime_seconds > 0.0

        assert (
            0.0
            <= record.mean_acceptance_rate
            <= 1.0
        )

        assert np.isfinite(
            record.plaquette_ess_per_second
        )

        assert np.isfinite(
            record.scalar_ess_per_second
        )

        assert (
            record.maximum_microcanonical_error
            < 1e-8
        )


def test_campaign_summary_contains_paired_baseline_ratios():
    records = run_overrelaxation_campaign(
        tiny_configuration(),
        progress=False,
    )

    summaries = summarize_campaign(
        records
    )

    assert len(
        summaries
    ) == 2

    baseline = [
        summary
        for summary in summaries
        if summary.overrelaxation_sweeps == 0
    ][
        0
    ]

    assert np.isclose(
        baseline.mean_plaquette_efficiency_ratio_vs_metropolis,
        1.0,
    )

    assert np.isclose(
        baseline.mean_scalar_efficiency_ratio_vs_metropolis,
        1.0,
    )


def test_duplicate_schedules_rejected():
    with pytest.raises(ValueError):
        OverrelaxationCampaignConfig(
            shape=(
                3,
                3,
                3,
            ),
            betas=(
                2.0,
            ),
            seeds=(
                2026,
            ),
            schedules=(
                0,
                1,
                1,
            ),
            epsilon=0.18,
            thermalization_sweeps=2,
            measurement_updates=4,
        )
