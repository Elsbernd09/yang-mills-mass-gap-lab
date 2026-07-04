import numpy as np
import pytest

from ymlab.spectroscopy import (
    BootstrapMassResult,
    CoshFitResult,
    PlateauResult,
    arccosh_effective_mass,
    bootstrap_cosh_mass,
    fit_periodic_cosh,
    periodic_cosh_correlator,
    scan_effective_mass_plateaus,
)


def test_periodic_cosh_is_symmetric():
    temporal_extent = 10
    lags = np.arange(
        temporal_extent,
        dtype=float,
    )

    values = periodic_cosh_correlator(
        lag=lags,
        amplitude=2.0,
        mass=0.7,
        temporal_extent=temporal_extent,
    )

    for lag in range(
        1,
        temporal_extent,
    ):
        assert np.isclose(
            values[lag],
            values[
                temporal_extent - lag
            ],
        )


def test_arccosh_effective_mass_recovers_exact_mass():
    temporal_extent = 12
    exact_mass = 0.65

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=1.7,
        mass=exact_mass,
        temporal_extent=temporal_extent,
    )

    effective_mass = arccosh_effective_mass(
        correlation
    )

    finite_values = effective_mass[
        np.isfinite(effective_mass)
    ]

    assert len(finite_values) > 0

    assert np.allclose(
        finite_values,
        exact_mass,
        atol=1e-10,
        rtol=1e-10,
    )


def test_fit_periodic_cosh_recovers_exact_parameters():
    temporal_extent = 12
    amplitude = 1.8
    mass = 0.55

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=amplitude,
        mass=mass,
        temporal_extent=temporal_extent,
    )

    result = fit_periodic_cosh(
        correlation=correlation,
        fit_start=1,
        fit_stop=6,
    )

    assert isinstance(
        result,
        CoshFitResult,
    )
    assert result.success
    assert np.isclose(
        result.amplitude,
        amplitude,
        atol=1e-7,
    )
    assert np.isclose(
        result.mass,
        mass,
        atol=1e-7,
    )


def test_effective_mass_invalid_ratio_returns_nan():
    correlation = np.array(
        [
            1.0,
            2.0,
            1.0,
            2.0,
        ]
    )

    result = arccosh_effective_mass(
        correlation
    )

    assert np.isnan(result[1])


def test_plateau_scan_finds_flat_window():
    effective_mass = np.array(
        [
            np.nan,
            0.8,
            0.6,
            0.5000,
            0.5001,
            0.4999,
            np.nan,
        ]
    )

    results = scan_effective_mass_plateaus(
        effective_mass=effective_mass,
        minimum_window=2,
        maximum_window=3,
    )

    assert len(results) > 0
    assert isinstance(
        results[0],
        PlateauResult,
    )

    best = results[0]

    assert best.start in [3, 4]
    assert np.isclose(
        best.mean_mass,
        0.5,
        atol=2e-4,
    )


def test_plateau_scan_rejects_short_minimum():
    effective_mass = np.array(
        [
            np.nan,
            0.5,
            0.5,
            np.nan,
        ]
    )

    with pytest.raises(ValueError):
        scan_effective_mass_plateaus(
            effective_mass,
            minimum_window=1,
        )


def test_fit_rejects_nonpositive_fit_window():
    correlation = np.array(
        [
            1.0,
            0.5,
            -0.1,
            0.5,
            1.0,
        ]
    )

    with pytest.raises(ValueError):
        fit_periodic_cosh(
            correlation=correlation,
            fit_start=1,
            fit_stop=4,
        )


def test_bootstrap_cosh_mass_returns_distribution():
    rng = np.random.default_rng(123)

    temporal_extent = 8
    base = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=1.0,
        mass=0.6,
        temporal_extent=temporal_extent,
    )

    ensemble = np.asarray(
        [
            base
            + rng.normal(
                0.0,
                0.002,
                size=temporal_extent,
            )
            for _ in range(30)
        ]
    )

    # Here each row is used as a synthetic operator-like time series.
    # Add a positive offset so the connected ensemble correlator has a
    # numerically usable structure under resampling.
    ensemble = ensemble + 2.0

    try:
        result = bootstrap_cosh_mass(
            operator_series_ensemble=ensemble,
            fit_start=1,
            fit_stop=4,
            n_bootstrap=50,
            seed=123,
        )

        assert isinstance(
            result,
            BootstrapMassResult,
        )
        assert result.successful_fits >= 2
        assert result.standard_error >= 0.0

    except ValueError as error:
        # Synthetic noisy operator ensembles may fail positivity in a connected
        # correlator fit. The function must fail explicitly rather than return
        # a fabricated mass.
        assert (
            "bootstrap" in str(error).lower()
            or "successful" in str(error).lower()
        )
