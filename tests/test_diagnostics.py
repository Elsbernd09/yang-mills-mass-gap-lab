import numpy as np
import pytest

from ymlab.diagnostics import (
    AutocorrelationDiagnostics,
    remove_burn_in,
    autocorrelation_function,
    integrated_autocorrelation_time,
    effective_sample_size,
    diagnose_autocorrelation,
)


def test_remove_burn_in():
    values = np.array([1, 2, 3, 4, 5], dtype=float)

    cleaned = remove_burn_in(values, burn_in=2)

    assert np.allclose(cleaned, np.array([3, 4, 5], dtype=float))


def test_remove_burn_in_rejects_too_large():
    values = np.array([1, 2, 3], dtype=float)

    with pytest.raises(ValueError):
        remove_burn_in(values, burn_in=3)


def test_autocorrelation_lag_zero_is_one():
    values = np.array([1, 2, 3, 4, 5], dtype=float)

    autocorr = autocorrelation_function(values, max_lag=2)

    assert np.isclose(autocorr[0], 1.0)


def test_autocorrelation_constant_data():
    values = np.array([2, 2, 2, 2], dtype=float)

    autocorr = autocorrelation_function(values, max_lag=2)

    assert np.isclose(autocorr[0], 1.0)
    assert np.isclose(autocorr[1], 0.0)
    assert np.isclose(autocorr[2], 0.0)


def test_integrated_autocorrelation_time_positive():
    autocorr = np.array([1.0, 0.5, 0.25, -0.1])

    tau = integrated_autocorrelation_time(autocorr)

    assert tau >= 0.5


def test_effective_sample_size():
    ess = effective_sample_size(n_samples=100, tau_int=2.0)

    assert np.isclose(ess, 25.0)


def test_diagnose_autocorrelation_returns_object():
    values = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=float)

    result = diagnose_autocorrelation(values, burn_in=1, max_lag=3)

    assert isinstance(result, AutocorrelationDiagnostics)
    assert len(result.autocorrelation) == 4
    assert result.integrated_autocorrelation_time > 0
    assert result.effective_sample_size > 0


def test_autocorrelation_rejects_short_series():
    with pytest.raises(ValueError):
        autocorrelation_function(np.array([1.0]))
