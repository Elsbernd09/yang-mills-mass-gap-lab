import numpy as np
import pytest

from ymlab.principal_fits import (
    CorrelatedPrincipalFitResult,
    PrincipalBootstrapFitSummary,
    bootstrap_refit_principal_mass,
    fit_correlated_principal_mass,
    normalized_periodic_principal_correlator,
    principal_state_covariance,
    scan_principal_fit_windows,
)


def test_normalized_principal_model_is_one_at_reference():
    temporal_extent = 12
    reference_time = 1

    value = (
        normalized_periodic_principal_correlator(
            lag=np.array(
                [
                    reference_time
                ],
                dtype=float,
            ),
            mass=0.6,
            temporal_extent=temporal_extent,
            reference_time=reference_time,
        )
    )

    assert np.allclose(
        value,
        1.0,
        atol=1e-14,
        rtol=1e-14,
    )


def test_normalized_principal_model_is_periodic_symmetric():
    temporal_extent = 12

    lags = np.arange(
        temporal_extent,
        dtype=float,
    )

    values = (
        normalized_periodic_principal_correlator(
            lag=lags,
            mass=0.5,
            temporal_extent=temporal_extent,
            reference_time=1,
        )
    )

    for lag in range(
        temporal_extent
    ):
        partner = (
            temporal_extent - lag
        ) % temporal_extent

        assert np.isclose(
            values[lag],
            values[partner],
            atol=1e-14,
            rtol=1e-14,
        )


def test_exact_correlated_principal_fit_recovers_mass():
    temporal_extent = 14
    reference_time = 1
    exact_mass = 0.63

    principal = (
        normalized_periodic_principal_correlator(
            lag=np.arange(
                temporal_extent,
                dtype=float,
            ),
            mass=exact_mass,
            temporal_extent=temporal_extent,
            reference_time=reference_time,
        )
    )

    covariance = np.eye(
        temporal_extent,
        dtype=float,
    ) * 1e-4

    result = fit_correlated_principal_mass(
        principal_correlator=principal,
        covariance=covariance,
        reference_time=reference_time,
        fit_start=1,
        fit_stop=6,
        shrinkage=0.0,
        relative_cutoff=1e-14,
    )

    assert isinstance(
        result,
        CorrelatedPrincipalFitResult,
    )

    assert result.success

    assert np.isclose(
        result.mass,
        exact_mass,
        atol=1e-5,
        rtol=1e-5,
    )

    assert result.chi_squared < 1e-8


def test_exact_fit_handles_nondiagonal_covariance():
    temporal_extent = 12
    exact_mass = 0.47

    principal = (
        normalized_periodic_principal_correlator(
            lag=np.arange(
                temporal_extent,
                dtype=float,
            ),
            mass=exact_mass,
            temporal_extent=temporal_extent,
            reference_time=1,
        )
    )

    covariance = np.full(
        (
            temporal_extent,
            temporal_extent,
        ),
        2e-5,
    )

    covariance += np.eye(
        temporal_extent
    ) * 8e-5

    result = fit_correlated_principal_mass(
        principal_correlator=principal,
        covariance=covariance,
        reference_time=1,
        fit_start=1,
        fit_stop=6,
        shrinkage=0.05,
    )

    assert result.success

    assert np.isclose(
        result.mass,
        exact_mass,
        atol=1e-5,
        rtol=1e-5,
    )


def test_principal_state_covariance_shape():
    rng = np.random.default_rng(
        123
    )

    samples = rng.normal(
        size=(
            50,
            8,
            3,
        )
    )

    result = principal_state_covariance(
        principal_samples=samples,
        state=1,
    )

    assert result.state == 1

    assert result.estimate.covariance.shape == (
        8,
        8,
    )

    assert result.finite_replicates == 50


def test_principal_window_scan_finds_exact_fit():
    temporal_extent = 14
    exact_mass = 0.55

    principal = (
        normalized_periodic_principal_correlator(
            lag=np.arange(
                temporal_extent,
                dtype=float,
            ),
            mass=exact_mass,
            temporal_extent=temporal_extent,
            reference_time=1,
        )
    )

    covariance = np.eye(
        temporal_extent
    ) * 1e-4

    fits = scan_principal_fit_windows(
        principal_correlator=principal,
        covariance=covariance,
        reference_time=1,
        shrinkage=0.0,
        relative_cutoff=1e-14,
        minimum_points=3,
    )

    assert len(fits) > 0

    assert np.isclose(
        fits[0].mass,
        exact_mass,
        atol=1e-5,
    )


def test_bootstrap_refit_recovers_mass_distribution():
    rng = np.random.default_rng(
        123
    )

    temporal_extent = 12
    exact_mass = 0.58

    central = (
        normalized_periodic_principal_correlator(
            lag=np.arange(
                temporal_extent,
                dtype=float,
            ),
            mass=exact_mass,
            temporal_extent=temporal_extent,
            reference_time=1,
        )
    )

    covariance = np.eye(
        temporal_extent
    ) * 1e-4

    central_fit = fit_correlated_principal_mass(
        principal_correlator=central,
        covariance=covariance,
        reference_time=1,
        fit_start=1,
        fit_stop=6,
        shrinkage=0.0,
    )

    samples = []

    for _ in range(40):
        noisy = (
            central
            + rng.normal(
                scale=1e-3,
                size=temporal_extent,
            )
        )

        noisy[1] = 1.0

        samples.append(
            noisy
        )

    summary = bootstrap_refit_principal_mass(
        principal_samples=np.asarray(
            samples
        ),
        central_fit=central_fit,
        covariance=covariance,
        shrinkage=0.0,
    )

    assert isinstance(
        summary,
        PrincipalBootstrapFitSummary,
    )

    assert summary.accepted_fits > 20

    assert abs(
        summary.mean_mass
        - exact_mass
    ) < 0.05


def test_fit_rejects_negative_window_values():
    principal = np.ones(
        10
    )

    principal[3] = -1.0

    covariance = np.eye(
        10
    )

    with pytest.raises(ValueError):
        fit_correlated_principal_mass(
            principal_correlator=principal,
            covariance=covariance,
            reference_time=1,
            fit_start=1,
            fit_stop=5,
        )


def test_invalid_mass_bounds_rejected():
    principal = np.ones(
        10
    )

    covariance = np.eye(
        10
    )

    with pytest.raises(ValueError):
        fit_correlated_principal_mass(
            principal_correlator=principal,
            covariance=covariance,
            reference_time=1,
            fit_start=1,
            fit_stop=5,
            mass_bounds=(
                1.0,
                0.5,
            ),
        )
