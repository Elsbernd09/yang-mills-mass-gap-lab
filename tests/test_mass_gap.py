import numpy as np
import pytest

from ymlab.mass_gap import (
    effective_mass,
    fit_exponential_mass,
    exponential_decay,
    EffectiveMassResult,
    ExponentialFitResult,
)


def test_effective_mass_known_decay():
    mass = 0.5
    t = np.arange(6)
    correlation = np.exp(-mass * t)

    result = effective_mass(correlation)

    assert isinstance(result, EffectiveMassResult)
    assert np.allclose(result.effective_masses, mass)


def test_effective_mass_skips_nonpositive_values():
    correlation = np.array([1.0, 0.5, -0.1, 0.2])

    result = effective_mass(correlation)

    assert len(result.effective_masses) == 1


def test_exponential_decay_function():
    t = np.array([0, 1, 2])
    values = exponential_decay(t, amplitude=2.0, mass=1.0)

    assert np.allclose(values, 2.0 * np.exp(-t))


def test_fit_exponential_mass_known_decay():
    true_mass = 0.4
    t = np.arange(8)
    correlation = 3.0 * np.exp(-true_mass * t)

    result = fit_exponential_mass(correlation)

    assert isinstance(result, ExponentialFitResult)
    assert np.isclose(result.mass, true_mass, atol=1e-4)


def test_fit_exponential_mass_rejects_bad_range():
    correlation = np.exp(-0.2 * np.arange(5))

    with pytest.raises(ValueError):
        fit_exponential_mass(correlation, min_lag=3, max_lag=2)


def test_effective_mass_rejects_short_input():
    with pytest.raises(ValueError):
        effective_mass(np.array([1.0]))
