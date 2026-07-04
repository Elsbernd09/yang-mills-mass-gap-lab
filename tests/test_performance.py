import numpy as np
import pytest

from ymlab.performance import (
    PerformanceResult,
    benchmark_lattice,
    benchmark_many,
    performance_results_as_rows,
)


def test_benchmark_lattice_returns_result():
    result = benchmark_lattice(
        shape=(3, 3),
        beta=2.0,
        epsilon=0.1,
        sweeps=1,
        seed=123,
    )

    assert isinstance(result, PerformanceResult)
    assert result.dimension == 2
    assert result.sites == 9
    assert result.links == 18
    assert result.plaquettes == 9
    assert result.total_runtime_seconds > 0
    assert result.average_sweep_seconds > 0
    assert result.link_updates_per_second > 0
    assert result.plaquettes_per_second > 0
    assert 0.0 <= result.mean_acceptance_rate <= 1.0


def test_benchmark_many_returns_expected_count():
    results = benchmark_many(
        shapes=[(3, 3), (2, 2, 2)],
        beta=2.0,
        epsilon=0.1,
        sweeps=1,
        seed=123,
    )

    assert len(results) == 2
    assert results[0].dimension == 2
    assert results[1].dimension == 3


def test_performance_rows_have_expected_keys():
    results = benchmark_many(
        shapes=[(3, 3)],
        beta=2.0,
        epsilon=0.1,
        sweeps=1,
        seed=123,
    )

    rows = performance_results_as_rows(results)
    row = rows[0]

    assert row["shape"] == "(3, 3)"
    assert "average_sweep_seconds" in row
    assert "link_updates_per_second" in row
    assert "plaquettes_per_second" in row


def test_benchmark_rejects_bad_sweeps():
    with pytest.raises(ValueError):
        benchmark_lattice(
            shape=(3, 3),
            beta=2.0,
            epsilon=0.1,
            sweeps=0,
        )


def test_benchmark_many_rejects_empty_shapes():
    with pytest.raises(ValueError):
        benchmark_many(
            shapes=[],
            beta=2.0,
            epsilon=0.1,
            sweeps=1,
        )
