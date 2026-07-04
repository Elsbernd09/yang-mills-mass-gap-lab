import numpy as np
import pytest

from ymlab.dimensional_analysis import (
    DimensionSummary,
    theoretical_plaquettes_per_site,
    run_dimension_summary,
    run_dimension_comparison,
    dimension_summaries_as_rows,
)


def test_theoretical_plaquettes_per_site():
    assert theoretical_plaquettes_per_site(2) == 1
    assert theoretical_plaquettes_per_site(3) == 3
    assert theoretical_plaquettes_per_site(4) == 6


def test_theoretical_plaquettes_rejects_low_dimension():
    with pytest.raises(ValueError):
        theoretical_plaquettes_per_site(1)


def test_run_dimension_summary_returns_summary():
    summary = run_dimension_summary(
        shape=(3, 3),
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seed=123,
    )

    assert isinstance(summary, DimensionSummary)
    assert summary.dimension == 2
    assert summary.sites == 9
    assert summary.links == 18
    assert summary.plaquettes == 9
    assert np.isfinite(summary.final_action)
    assert 0.0 <= summary.mean_acceptance_rate <= 1.0


def test_run_dimension_comparison_count():
    summaries = run_dimension_comparison(
        shapes=[(3, 3), (2, 2, 2)],
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seed=123,
    )

    assert len(summaries) == 2
    assert summaries[0].dimension == 2
    assert summaries[1].dimension == 3


def test_dimension_summaries_as_rows():
    summaries = run_dimension_comparison(
        shapes=[(3, 3)],
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seed=123,
    )

    rows = dimension_summaries_as_rows(summaries)
    row = rows[0]

    assert row["shape"] == "(3, 3)"
    assert row["dimension"] == 2
    assert "action_density_per_site" in row
    assert "runtime_seconds" in row


def test_run_dimension_comparison_rejects_empty_shapes():
    with pytest.raises(ValueError):
        run_dimension_comparison(
            shapes=[],
            beta=2.0,
            sweeps=2,
            epsilon=0.1,
            seed=123,
        )
