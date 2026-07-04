import numpy as np
import pytest

from ymlab.resampling import (
    BlockSensitivityPoint,
    ResamplingEstimate,
    block_size_sensitivity,
    circular_block_sample,
    delete_one_block_jackknife_mean,
    delete_one_jackknife_mean,
    moving_block_bootstrap_mean,
    naive_bootstrap_mean,
)


def test_naive_bootstrap_returns_estimate():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = naive_bootstrap_mean(
        values,
        n_bootstrap=100,
        seed=123,
    )

    assert isinstance(result, ResamplingEstimate)
    assert np.isclose(result.estimate, 2.5)
    assert result.standard_error >= 0.0
    assert len(result.resampled_statistics) == 100


def test_circular_block_sample_preserves_length():
    values = np.arange(10, dtype=float)
    rng = np.random.default_rng(123)

    sample = circular_block_sample(
        values=values,
        block_size=4,
        rng=rng,
    )

    assert len(sample) == len(values)


def test_circular_block_sample_values_come_from_source():
    values = np.arange(8, dtype=float)
    rng = np.random.default_rng(123)

    sample = circular_block_sample(
        values=values,
        block_size=3,
        rng=rng,
    )

    assert set(sample).issubset(set(values))


def test_moving_block_bootstrap_returns_estimate():
    values = np.arange(20, dtype=float)

    result = moving_block_bootstrap_mean(
        values=values,
        block_size=4,
        n_bootstrap=100,
        seed=123,
    )

    assert isinstance(result, ResamplingEstimate)
    assert np.isclose(result.estimate, np.mean(values))
    assert result.standard_error >= 0.0
    assert len(result.resampled_statistics) == 100


def test_block_size_one_is_valid():
    values = np.arange(10, dtype=float)

    result = moving_block_bootstrap_mean(
        values=values,
        block_size=1,
        n_bootstrap=50,
        seed=123,
    )

    assert np.isfinite(result.standard_error)


def test_delete_one_jackknife_known_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = delete_one_jackknife_mean(values)

    assert isinstance(result, ResamplingEstimate)
    assert np.isclose(result.estimate, 2.5)
    assert result.standard_error > 0.0
    assert len(result.resampled_statistics) == 4


def test_delete_one_block_jackknife():
    values = np.arange(12, dtype=float)

    result = delete_one_block_jackknife_mean(
        values=values,
        block_size=3,
    )

    assert isinstance(result, ResamplingEstimate)
    assert len(result.resampled_statistics) == 4
    assert result.standard_error >= 0.0


def test_delete_one_block_jackknife_rejects_too_few_blocks():
    values = np.arange(5, dtype=float)

    with pytest.raises(ValueError):
        delete_one_block_jackknife_mean(
            values=values,
            block_size=4,
        )


def test_block_size_sensitivity_returns_points():
    values = np.arange(20, dtype=float)

    points = block_size_sensitivity(
        values=values,
        block_sizes=[1, 2, 4],
        n_bootstrap=50,
        seed=123,
    )

    assert len(points) == 3

    for point in points:
        assert isinstance(point, BlockSensitivityPoint)
        assert point.standard_error >= 0.0


def test_block_size_sensitivity_rejects_empty_sizes():
    values = np.arange(20, dtype=float)

    with pytest.raises(ValueError):
        block_size_sensitivity(
            values=values,
            block_sizes=[],
        )


def test_resampling_rejects_nonfinite_values():
    values = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError):
        naive_bootstrap_mean(values)
