import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.correlations import (
    plaquette_slice_observable,
    plaquette_time_series,
    connected_autocorrelation,
    normalized_connected_autocorrelation,
)


def test_cold_start_plaquette_slice_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    value = plaquette_slice_observable(
        lattice=lattice,
        time_direction=0,
        time_index=0,
        mu=0,
        nu=1,
    )

    assert np.isclose(value, 0.0)


def test_plaquette_time_series_length():
    lattice = Lattice(shape=(5, 4), cold_start=True)

    series = plaquette_time_series(
        lattice=lattice,
        time_direction=0,
        mu=0,
        nu=1,
    )

    assert len(series) == 5


def test_connected_autocorrelation_basic_shape():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    corr = connected_autocorrelation(values)

    assert corr.shape == values.shape


def test_normalized_autocorrelation_starts_at_one_for_nonconstant_data():
    values = np.array([1.0, 2.0, 1.0, 2.0])

    corr = normalized_connected_autocorrelation(values)

    assert np.isclose(corr[0], 1.0)


def test_normalized_autocorrelation_constant_data_returns_zeros():
    values = np.array([2.0, 2.0, 2.0, 2.0])

    corr = normalized_connected_autocorrelation(values)

    assert np.allclose(corr, 0.0)


def test_invalid_autocorrelation_input_raises_error():
    with pytest.raises(ValueError):
        connected_autocorrelation(np.array([[1.0, 2.0], [3.0, 4.0]]))
