import numpy as np
import pytest

from ymlab.covariance_fits import (
    CorrelatedCoshFitResult,
    CovarianceEstimate,
    bootstrap_connected_correlators,
    correlated_chi_squared,
    estimate_covariance,
    fit_correlated_periodic_cosh,
    regularized_inverse_covariance,
    shrink_covariance_to_diagonal,
)
from ymlab.spectroscopy import (
    periodic_cosh_correlator,
)


def test_estimate_covariance_known_shape():
    samples = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    result = estimate_covariance(
        samples
    )

    assert isinstance(
        result,
        CovarianceEstimate,
    )

    assert result.mean.shape == (2,)
    assert result.covariance.shape == (
        2,
        2,
    )
    assert result.correlation.shape == (
        2,
        2,
    )

    assert np.isclose(
        result.correlation[0, 1],
        1.0,
    )


def test_shrinkage_zero_preserves_covariance():
    covariance = np.array(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )

    result = shrink_covariance_to_diagonal(
        covariance,
        shrinkage=0.0,
    )

    assert np.allclose(
        result,
        covariance,
    )


def test_shrinkage_one_returns_diagonal():
    covariance = np.array(
        [
            [2.0, 0.5],
            [0.5, 1.0],
        ]
    )

    result = shrink_covariance_to_diagonal(
        covariance,
        shrinkage=1.0,
    )

    assert np.allclose(
        result,
        np.diag(
            np.diag(
                covariance
            )
        ),
    )


def test_regularized_inverse_recovers_full_rank_inverse():
    covariance = np.array(
        [
            [2.0, 0.3],
            [0.3, 1.0],
        ]
    )

    result = regularized_inverse_covariance(
        covariance,
        relative_cutoff=1e-14,
        absolute_cutoff=1e-18,
    )

    expected = np.linalg.inv(
        covariance
    )

    assert result.retained_rank == 2

    assert np.allclose(
        result.inverse_covariance,
        expected,
        atol=1e-10,
        rtol=1e-10,
    )


def test_regularized_inverse_truncates_near_null_mode():
    covariance = np.diag(
        [
            2.0,
            1.0,
            1e-18,
        ]
    )

    result = regularized_inverse_covariance(
        covariance,
        relative_cutoff=1e-10,
        absolute_cutoff=1e-16,
    )

    assert result.retained_rank == 2


def test_correlated_chi_squared_known_value():
    residuals = np.array(
        [
            1.0,
            2.0,
        ]
    )

    inverse_covariance = np.eye(
        2
    )

    value = correlated_chi_squared(
        residuals,
        inverse_covariance,
    )

    assert np.isclose(
        value,
        5.0,
    )


def test_exact_correlated_cosh_fit_recovers_parameters():
    temporal_extent = 12
    amplitude = 1.7
    mass = 0.62

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=amplitude,
        mass=mass,
        temporal_extent=temporal_extent,
    )

    covariance = np.eye(
        temporal_extent,
        dtype=float,
    ) * 1e-4

    result = fit_correlated_periodic_cosh(
        correlation=correlation,
        covariance=covariance,
        fit_start=1,
        fit_stop=6,
        shrinkage=0.0,
        relative_cutoff=1e-14,
    )

    assert isinstance(
        result,
        CorrelatedCoshFitResult,
    )

    assert result.success

    assert np.isclose(
        result.amplitude,
        amplitude,
        atol=1e-5,
        rtol=1e-5,
    )

    assert np.isclose(
        result.mass,
        mass,
        atol=1e-5,
        rtol=1e-5,
    )

    assert result.chi_squared < 1e-6


def test_correlated_fit_handles_nondiagonal_covariance():
    temporal_extent = 10
    amplitude = 1.2
    mass = 0.5

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=amplitude,
        mass=mass,
        temporal_extent=temporal_extent,
    )

    covariance = np.full(
        (
            temporal_extent,
            temporal_extent,
        ),
        2e-5,
        dtype=float,
    )

    covariance += np.eye(
        temporal_extent
    ) * 8e-5

    result = fit_correlated_periodic_cosh(
        correlation=correlation,
        covariance=covariance,
        fit_start=1,
        fit_stop=5,
        shrinkage=0.05,
    )

    assert result.success

    assert np.isclose(
        result.mass,
        mass,
        atol=1e-5,
    )


def test_bootstrap_connected_correlators_shape():
    rng = np.random.default_rng(
        123
    )

    ensemble = rng.normal(
        size=(
            20,
            6,
        )
    )

    replicates = (
        bootstrap_connected_correlators(
            operator_series_ensemble=ensemble,
            n_bootstrap=50,
            seed=123,
        )
    )

    assert replicates.shape == (
        50,
        6,
    )

    assert np.all(
        np.isfinite(
            replicates
        )
    )


def test_invalid_shrinkage_is_rejected():
    covariance = np.eye(
        3
    )

    with pytest.raises(ValueError):
        shrink_covariance_to_diagonal(
            covariance,
            shrinkage=1.5,
        )


def test_correlated_fit_rejects_too_short_window():
    correlation = np.ones(
        6
    )

    covariance = np.eye(
        6
    )

    with pytest.raises(ValueError):
        fit_correlated_periodic_cosh(
            correlation=correlation,
            covariance=covariance,
            fit_start=1,
            fit_stop=3,
        )
