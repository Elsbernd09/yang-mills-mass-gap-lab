import numpy as np
import pytest

from ymlab.glueball import (
    GlueballCorrelatorResult,
    ensemble_connected_correlator,
    normalized_connected_correlator,
    periodic_configuration_correlation,
    scalar_glueball_operator,
    scalar_glueball_time_series,
    spatial_plane_pairs,
)
from ymlab.gauge_transformations import (
    gauge_transform_lattice,
    random_gauge_field,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep


def nontrivial_3d_lattice() -> Lattice:
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
        seed=2026,
    )

    for _ in range(3):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.15,
        )

    return lattice


def test_spatial_plane_pairs_3d():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    planes = spatial_plane_pairs(
        lattice=lattice,
        time_direction=0,
    )

    assert planes == [(1, 2)]


def test_spatial_plane_pairs_4d():
    lattice = Lattice(
        shape=(3, 3, 3, 3),
        cold_start=True,
    )

    planes = spatial_plane_pairs(
        lattice=lattice,
        time_direction=0,
    )

    assert planes == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_cold_start_scalar_operator_is_zero():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    value = scalar_glueball_operator(
        lattice=lattice,
        time_direction=0,
        time_index=0,
    )

    assert np.isclose(value, 0.0)


def test_scalar_time_series_has_temporal_extent():
    lattice = Lattice(
        shape=(5, 4, 4),
        cold_start=True,
    )

    series = scalar_glueball_time_series(
        lattice=lattice,
        time_direction=0,
    )

    assert series.shape == (5,)


def test_scalar_operator_requires_three_dimensions():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        scalar_glueball_operator(
            lattice=lattice,
            time_direction=0,
            time_index=0,
        )


def test_scalar_operator_is_gauge_invariant():
    lattice = nontrivial_3d_lattice()

    gauge_field = random_gauge_field(
        lattice=lattice,
        seed=314159,
    )

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    before = scalar_glueball_time_series(
        lattice=lattice,
        time_direction=0,
    )

    after = scalar_glueball_time_series(
        lattice=transformed,
        time_direction=0,
    )

    assert np.allclose(
        before,
        after,
        atol=1e-8,
        rtol=1e-8,
    )


def test_periodic_configuration_correlation_shape():
    series = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    correlation = periodic_configuration_correlation(
        series
    )

    assert correlation.shape == series.shape


def test_periodic_configuration_correlation_is_periodic_symmetric():
    series = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    correlation = periodic_configuration_correlation(
        series
    )

    assert np.isclose(
        correlation[1],
        correlation[-1],
    )


def test_ensemble_connected_correlator_returns_result():
    ensemble = np.array(
        [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 3.0],
            [1.5, 2.5, 1.5, 2.5],
        ]
    )

    result = ensemble_connected_correlator(
        ensemble
    )

    assert isinstance(
        result,
        GlueballCorrelatorResult,
    )
    assert result.temporal_extent == 4
    assert result.number_of_configurations == 3
    assert result.correlation.shape == (4,)


def test_connected_correlator_is_periodic_symmetric():
    ensemble = np.array(
        [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 3.0],
            [1.5, 2.5, 1.5, 2.5],
        ]
    )

    result = ensemble_connected_correlator(
        ensemble
    )

    assert np.isclose(
        result.correlation[1],
        result.correlation[-1],
    )


def test_normalized_connected_correlator_starts_at_one():
    correlation = np.array(
        [2.0, 1.0, 0.5]
    )

    normalized = normalized_connected_correlator(
        correlation
    )

    assert np.isclose(
        normalized[0],
        1.0,
    )


def test_normalized_zero_correlator_returns_zeros():
    correlation = np.zeros(4)

    normalized = normalized_connected_correlator(
        correlation
    )

    assert np.allclose(
        normalized,
        0.0,
    )


def test_ensemble_correlator_rejects_one_configuration():
    ensemble = np.array(
        [
            [1.0, 2.0, 3.0]
        ]
    )

    with pytest.raises(ValueError):
        ensemble_connected_correlator(
            ensemble
        )
