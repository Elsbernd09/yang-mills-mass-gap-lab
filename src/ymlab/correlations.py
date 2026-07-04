"""
Correlation functions for SU(2) lattice gauge theory.

This module builds simple gauge-invariant correlation diagnostics from
plaquette action densities.

The goal is to create numerical observables that can be used for
mass-gap-style effective mass estimation.

This is exploratory and finite-lattice based. It is not a proof of the
Yang-Mills mass gap.
"""

from __future__ import annotations

import numpy as np

from ymlab.lattice import Lattice
from ymlab.plaquette import plaquette_action_density


def plaquette_slice_observable(
    lattice: Lattice,
    time_direction: int,
    time_index: int,
    mu: int = 0,
    nu: int = 1,
) -> float:
    """
    Average plaquette action density over a fixed time slice.

    Parameters
    ----------
    lattice:
        Lattice configuration.
    time_direction:
        Direction treated as the Euclidean time direction.
    time_index:
        Slice index in the time direction.
    mu, nu:
        Plaquette plane directions.

    Returns
    -------
    float
        Average plaquette action density on the selected slice.
    """
    if time_direction < 0 or time_direction >= lattice.dim:
        raise ValueError("Invalid time direction.")

    if time_index < 0 or time_index >= lattice.shape[time_direction]:
        raise ValueError("Invalid time index.")

    if mu == nu:
        raise ValueError("Plaquette directions must differ.")

    total = 0.0
    count = 0

    for site in lattice.sites():
        if site[time_direction] == time_index:
            total += plaquette_action_density(lattice, site, mu, nu)
            count += 1

    if count == 0:
        raise ValueError("No sites found in requested time slice.")

    return total / count


def plaquette_time_series(
    lattice: Lattice,
    time_direction: int,
    mu: int = 0,
    nu: int = 1,
) -> np.ndarray:
    """
    Compute a time series of plaquette action density averages.

    Each entry is the average plaquette action density on one time slice.
    """
    length = lattice.shape[time_direction]

    values = [
        plaquette_slice_observable(
            lattice=lattice,
            time_direction=time_direction,
            time_index=t,
            mu=mu,
            nu=nu,
        )
        for t in range(length)
    ]

    return np.asarray(values, dtype=float)


def connected_autocorrelation(values: np.ndarray) -> np.ndarray:
    """
    Compute the connected periodic autocorrelation of a one-dimensional series.

    The connected correlation subtracts the mean:

        C(t) = <(x_s - mean)(x_{s+t} - mean)>_s

    Periodic boundary conditions are used.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")

    if len(values) < 2:
        raise ValueError("Need at least two values to compute autocorrelation.")

    centered = values - np.mean(values)
    n = len(centered)

    correlations = []

    for lag in range(n):
        shifted = np.roll(centered, -lag)
        correlations.append(float(np.mean(centered * shifted)))

    return np.asarray(correlations, dtype=float)


def normalized_connected_autocorrelation(values: np.ndarray) -> np.ndarray:
    """
    Compute connected autocorrelation normalized by C(0).

    If C(0) is numerically zero, return zeros to avoid division errors.
    """
    corr = connected_autocorrelation(values)

    if np.isclose(corr[0], 0.0):
        return np.zeros_like(corr)

    return corr / corr[0]
