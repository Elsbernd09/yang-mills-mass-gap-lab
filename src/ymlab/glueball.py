"""
Gauge-invariant scalar glueball-style operators and Euclidean correlators.

This module constructs simple scalar operators from spatial plaquette action
densities on fixed Euclidean time slices.

The current implementation is an exploratory finite-lattice analogue of a
scalar 0++-style operator. It does not claim to perform a full irreducible
cubic-group projection or a continuum glueball spectroscopy calculation.

The purpose is to create a more physically meaningful mass-spectrum pipeline:

    gauge configurations
    -> gauge-invariant O(t)
    -> ensemble connected correlator C(t)
    -> effective mass analysis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice
from ymlab.plaquette import plaquette_action_density


@dataclass(frozen=True)
class GlueballCorrelatorResult:
    """Container for a connected Euclidean correlator."""

    lags: np.ndarray
    correlation: np.ndarray
    raw_correlation: np.ndarray
    mean_operator: float
    number_of_configurations: int
    temporal_extent: int


def _validate_time_direction(
    lattice: Lattice,
    time_direction: int,
) -> None:
    """Validate a Euclidean time direction."""
    if time_direction < 0 or time_direction >= lattice.dim:
        raise ValueError("Invalid time direction.")


def spatial_plane_pairs(
    lattice: Lattice,
    time_direction: int,
) -> list[tuple[int, int]]:
    """
    Return all plaquette planes not containing the Euclidean time direction.

    These are used to construct a simple scalar spatial plaquette operator.
    """
    _validate_time_direction(lattice, time_direction)

    spatial_directions = [
        direction
        for direction in range(lattice.dim)
        if direction != time_direction
    ]

    return [
        (mu, nu)
        for i, mu in enumerate(spatial_directions)
        for nu in spatial_directions[i + 1 :]
    ]


def scalar_glueball_operator(
    lattice: Lattice,
    time_direction: int,
    time_index: int,
) -> float:
    """
    Compute a simple scalar glueball-style operator O(t).

    O(t) is defined as the spatial average of plaquette action density across
    all purely spatial plaquette planes on the fixed time slice t.

    In a d-dimensional lattice, the time direction is excluded from the
    plaquette planes used in the operator.

    At least three lattice dimensions are required because a purely spatial
    plaquette needs two non-time directions.
    """
    _validate_time_direction(lattice, time_direction)

    temporal_extent = lattice.shape[time_direction]

    if time_index < 0 or time_index >= temporal_extent:
        raise ValueError("Invalid time index.")

    planes = spatial_plane_pairs(
        lattice=lattice,
        time_direction=time_direction,
    )

    if len(planes) == 0:
        raise ValueError(
            "Scalar spatial glueball-style operator requires "
            "at least three lattice dimensions."
        )

    total = 0.0
    count = 0

    for site in lattice.sites():
        if site[time_direction] != time_index:
            continue

        for mu, nu in planes:
            total += plaquette_action_density(
                lattice=lattice,
                site=site,
                mu=mu,
                nu=nu,
            )
            count += 1

    if count == 0:
        raise ValueError("No plaquettes found on requested time slice.")

    return float(total / count)


def scalar_glueball_time_series(
    lattice: Lattice,
    time_direction: int,
) -> np.ndarray:
    """
    Compute O(t) for every Euclidean time slice.
    """
    _validate_time_direction(lattice, time_direction)

    temporal_extent = lattice.shape[time_direction]

    return np.asarray(
        [
            scalar_glueball_operator(
                lattice=lattice,
                time_direction=time_direction,
                time_index=t,
            )
            for t in range(temporal_extent)
        ],
        dtype=float,
    )


def periodic_configuration_correlation(
    operator_series: np.ndarray,
) -> np.ndarray:
    """
    Compute the periodic raw correlation for one configuration.

    For a time series O(t),

        C_raw(tau) =
            (1/T) sum_t O(t) O(t + tau),

    with periodic indexing in Euclidean time.
    """
    operator_series = np.asarray(
        operator_series,
        dtype=float,
    )

    if operator_series.ndim != 1:
        raise ValueError(
            "operator_series must be one-dimensional."
        )

    if len(operator_series) < 2:
        raise ValueError(
            "Need at least two Euclidean time slices."
        )

    if not np.all(np.isfinite(operator_series)):
        raise ValueError(
            "operator_series must contain finite values."
        )

    temporal_extent = len(operator_series)

    return np.asarray(
        [
            np.mean(
                operator_series
                * np.roll(operator_series, -lag)
            )
            for lag in range(temporal_extent)
        ],
        dtype=float,
    )


def ensemble_connected_correlator(
    operator_series_ensemble: np.ndarray,
) -> GlueballCorrelatorResult:
    """
    Compute an ensemble-connected Euclidean correlator.

    Input shape:

        (number_of_configurations, temporal_extent)

    The raw correlator is

        < O(t) O(t + tau) >,

    averaged over both Euclidean time origins and configurations.

    The connected correlator is

        C(tau) =
            < O(t) O(t + tau) > - <O>^2.
    """
    ensemble = np.asarray(
        operator_series_ensemble,
        dtype=float,
    )

    if ensemble.ndim != 2:
        raise ValueError(
            "Expected a 2D array with shape "
            "(configurations, temporal_extent)."
        )

    number_of_configurations, temporal_extent = ensemble.shape

    if number_of_configurations < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if temporal_extent < 2:
        raise ValueError(
            "Need at least two Euclidean time slices."
        )

    if not np.all(np.isfinite(ensemble)):
        raise ValueError(
            "Ensemble must contain finite values."
        )

    raw_correlations = np.asarray(
        [
            periodic_configuration_correlation(series)
            for series in ensemble
        ],
        dtype=float,
    )

    raw_correlation = np.mean(
        raw_correlations,
        axis=0,
    )

    mean_operator = float(np.mean(ensemble))

    connected = (
        raw_correlation
        - mean_operator ** 2
    )

    return GlueballCorrelatorResult(
        lags=np.arange(
            temporal_extent,
            dtype=int,
        ),
        correlation=np.asarray(
            connected,
            dtype=float,
        ),
        raw_correlation=np.asarray(
            raw_correlation,
            dtype=float,
        ),
        mean_operator=mean_operator,
        number_of_configurations=number_of_configurations,
        temporal_extent=temporal_extent,
    )


def normalized_connected_correlator(
    correlation: np.ndarray,
) -> np.ndarray:
    """
    Normalize a connected correlator by C(0).

    If C(0) is numerically zero, return zeros.
    """
    correlation = np.asarray(
        correlation,
        dtype=float,
    )

    if correlation.ndim != 1:
        raise ValueError(
            "correlation must be one-dimensional."
        )

    if len(correlation) == 0:
        raise ValueError(
            "correlation cannot be empty."
        )

    if np.isclose(correlation[0], 0.0):
        return np.zeros_like(correlation)

    return correlation / correlation[0]
