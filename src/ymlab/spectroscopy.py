"""
Periodic Euclidean correlator spectroscopy utilities.

This module provides:

1. Periodic single-state cosh-like correlator models.
2. Arccosh effective-mass estimation.
3. Periodic correlator fitting.
4. Effective-mass plateau-window diagnostics.
5. Configuration-level bootstrap mass estimation.

The implementation is intended for exploratory finite-lattice spectroscopy.
It does not establish a rigorous continuum Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from ymlab.glueball import ensemble_connected_correlator


@dataclass(frozen=True)
class CoshFitResult:
    """Result of a periodic single-state correlator fit."""

    amplitude: float
    mass: float
    amplitude_error: float
    mass_error: float
    fit_start: int
    fit_stop: int
    fit_lags: np.ndarray
    fit_values: np.ndarray
    model_values: np.ndarray
    success: bool


@dataclass(frozen=True)
class PlateauResult:
    """Candidate effective-mass plateau window."""

    start: int
    stop: int
    number_of_points: int
    mean_mass: float
    standard_deviation: float
    relative_spread: float
    score: float


@dataclass(frozen=True)
class BootstrapMassResult:
    """Configuration-bootstrap mass estimate."""

    estimate: float
    standard_error: float
    lower_ci: float
    upper_ci: float
    successful_fits: int
    attempted_fits: int
    bootstrap_masses: np.ndarray


def periodic_cosh_correlator(
    lag: np.ndarray,
    amplitude: float,
    mass: float,
    temporal_extent: int,
) -> np.ndarray:
    """
    Evaluate a periodic single-state Euclidean correlator.

    C(t) = A [exp(-m t) + exp(-m (T - t))].
    """
    lag = np.asarray(lag, dtype=float)

    if temporal_extent <= 1:
        raise ValueError(
            "temporal_extent must be greater than one."
        )

    if mass < 0.0:
        raise ValueError("mass must be nonnegative.")

    return amplitude * (
        np.exp(-mass * lag)
        + np.exp(
            -mass * (
                float(temporal_extent) - lag
            )
        )
    )


def arccosh_effective_mass(
    correlation: np.ndarray,
) -> np.ndarray:
    """
    Compute the periodic arccosh effective-mass estimator.

    For interior times,

        m_eff(t) =
            arccosh(
                [C(t-1) + C(t+1)]
                / [2 C(t)]
            ).

    Invalid points are returned as NaN.

    The endpoints are also NaN because the estimator is reported only for the
    ordinary interior range rather than wrapping them periodically.
    """
    correlation = np.asarray(
        correlation,
        dtype=float,
    )

    if correlation.ndim != 1:
        raise ValueError(
            "correlation must be one-dimensional."
        )

    if len(correlation) < 3:
        raise ValueError(
            "Need at least three correlator values."
        )

    if not np.all(np.isfinite(correlation)):
        raise ValueError(
            "correlation must contain finite values."
        )

    effective_mass = np.full(
        len(correlation),
        np.nan,
        dtype=float,
    )

    for t in range(1, len(correlation) - 1):
        denominator = 2.0 * correlation[t]

        if np.isclose(denominator, 0.0):
            continue

        ratio = (
            correlation[t - 1]
            + correlation[t + 1]
        ) / denominator

        if not np.isfinite(ratio):
            continue

        if ratio < 1.0:
            continue

        effective_mass[t] = float(
            np.arccosh(ratio)
        )

    return effective_mass


def fit_periodic_cosh(
    correlation: np.ndarray,
    fit_start: int = 1,
    fit_stop: Optional[int] = None,
) -> CoshFitResult:
    """
    Fit a connected correlator to a periodic single-state model.

    The fit uses only the specified lag window. fit_stop is exclusive.

    Positive amplitude and positive mass bounds are imposed.
    """
    correlation = np.asarray(
        correlation,
        dtype=float,
    )

    if correlation.ndim != 1:
        raise ValueError(
            "correlation must be one-dimensional."
        )

    if len(correlation) < 4:
        raise ValueError(
            "Need at least four correlator values."
        )

    if not np.all(np.isfinite(correlation)):
        raise ValueError(
            "correlation must contain finite values."
        )

    temporal_extent = len(correlation)

    if fit_stop is None:
        fit_stop = (
            temporal_extent // 2
            + 1
        )

    if fit_start < 0:
        raise ValueError(
            "fit_start must be nonnegative."
        )

    if fit_stop > temporal_extent:
        raise ValueError(
            "fit_stop cannot exceed temporal extent."
        )

    if fit_stop - fit_start < 2:
        raise ValueError(
            "Fit window must contain at least two points."
        )

    fit_lags = np.arange(
        fit_start,
        fit_stop,
        dtype=float,
    )

    fit_values = correlation[
        fit_start:fit_stop
    ]

    if np.any(fit_values <= 0.0):
        raise ValueError(
            "Periodic cosh fit requires positive "
            "correlator values in the fit window."
        )

    initial_amplitude = max(
        float(fit_values[0]) / 2.0,
        np.finfo(float).eps,
    )

    initial_mass = 0.5

    def model(
        lag: np.ndarray,
        amplitude: float,
        mass: float,
    ) -> np.ndarray:
        return periodic_cosh_correlator(
            lag=lag,
            amplitude=amplitude,
            mass=mass,
            temporal_extent=temporal_extent,
        )

    try:
        parameters, covariance = curve_fit(
            model,
            fit_lags,
            fit_values,
            p0=[
                initial_amplitude,
                initial_mass,
            ],
            bounds=(
                [0.0, 1e-12],
                [np.inf, np.inf],
            ),
            maxfev=20000,
        )

        amplitude = float(parameters[0])
        mass = float(parameters[1])

        if covariance.shape == (2, 2):
            diagonal = np.diag(covariance)

            amplitude_error = float(
                np.sqrt(diagonal[0])
            ) if diagonal[0] >= 0.0 else np.nan

            mass_error = float(
                np.sqrt(diagonal[1])
            ) if diagonal[1] >= 0.0 else np.nan
        else:
            amplitude_error = np.nan
            mass_error = np.nan

        model_values = model(
            fit_lags,
            amplitude,
            mass,
        )

        success = bool(
            np.isfinite(amplitude)
            and np.isfinite(mass)
            and mass > 0.0
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ):
        amplitude = np.nan
        mass = np.nan
        amplitude_error = np.nan
        mass_error = np.nan
        model_values = np.full(
            len(fit_lags),
            np.nan,
            dtype=float,
        )
        success = False

    return CoshFitResult(
        amplitude=amplitude,
        mass=mass,
        amplitude_error=amplitude_error,
        mass_error=mass_error,
        fit_start=fit_start,
        fit_stop=fit_stop,
        fit_lags=fit_lags,
        fit_values=np.asarray(
            fit_values,
            dtype=float,
        ),
        model_values=np.asarray(
            model_values,
            dtype=float,
        ),
        success=success,
    )


def scan_effective_mass_plateaus(
    effective_mass: np.ndarray,
    minimum_window: int = 2,
    maximum_window: Optional[int] = None,
) -> list[PlateauResult]:
    """
    Scan contiguous finite effective-mass windows.

    Candidate windows are ranked by a simple stability score:

        score = relative_spread / sqrt(number_of_points).

    Smaller scores indicate flatter and/or longer candidate windows.

    This is a diagnostic heuristic, not an automatic proof that a true
    asymptotic ground-state plateau has been isolated.
    """
    effective_mass = np.asarray(
        effective_mass,
        dtype=float,
    )

    if effective_mass.ndim != 1:
        raise ValueError(
            "effective_mass must be one-dimensional."
        )

    if minimum_window < 2:
        raise ValueError(
            "minimum_window must be at least two."
        )

    if maximum_window is None:
        maximum_window = len(effective_mass)

    if maximum_window < minimum_window:
        raise ValueError(
            "maximum_window must be at least minimum_window."
        )

    results = []

    for start in range(len(effective_mass)):
        for stop in range(
            start + minimum_window,
            min(
                len(effective_mass),
                start + maximum_window,
            ) + 1,
        ):
            window = effective_mass[
                start:stop
            ]

            if not np.all(
                np.isfinite(window)
            ):
                continue

            if np.any(window <= 0.0):
                continue

            mean_mass = float(
                np.mean(window)
            )

            standard_deviation = float(
                np.std(
                    window,
                    ddof=1,
                )
            )

            relative_spread = (
                standard_deviation
                / abs(mean_mass)
                if not np.isclose(
                    mean_mass,
                    0.0,
                )
                else np.inf
            )

            number_of_points = len(window)

            score = float(
                relative_spread
                / np.sqrt(number_of_points)
            )

            results.append(
                PlateauResult(
                    start=start,
                    stop=stop,
                    number_of_points=number_of_points,
                    mean_mass=mean_mass,
                    standard_deviation=standard_deviation,
                    relative_spread=float(
                        relative_spread
                    ),
                    score=score,
                )
            )

    return sorted(
        results,
        key=lambda result: result.score,
    )


def bootstrap_cosh_mass(
    operator_series_ensemble: np.ndarray,
    fit_start: int = 1,
    fit_stop: Optional[int] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapMassResult:
    """
    Bootstrap the periodic-cosh mass at the configuration level.

    Entire configurations are resampled with replacement.

    This preserves each configuration's Euclidean time-slice operator vector.
    """
    ensemble = np.asarray(
        operator_series_ensemble,
        dtype=float,
    )

    if ensemble.ndim != 2:
        raise ValueError(
            "Expected a 2D configuration-by-time ensemble."
        )

    number_of_configurations = ensemble.shape[0]

    if number_of_configurations < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    rng = np.random.default_rng(seed)

    bootstrap_masses = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            number_of_configurations,
            size=number_of_configurations,
        )

        bootstrap_ensemble = ensemble[
            indices
        ]

        correlator = ensemble_connected_correlator(
            bootstrap_ensemble
        )

        try:
            fit = fit_periodic_cosh(
                correlation=correlator.correlation,
                fit_start=fit_start,
                fit_stop=fit_stop,
            )
        except ValueError:
            continue

        if (
            fit.success
            and np.isfinite(fit.mass)
            and fit.mass > 0.0
        ):
            bootstrap_masses.append(
                fit.mass
            )

    masses = np.asarray(
        bootstrap_masses,
        dtype=float,
    )

    if len(masses) < 2:
        raise ValueError(
            "Fewer than two successful bootstrap mass fits."
        )

    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            masses,
            alpha / 2.0,
        )
    )
    upper = float(
        np.quantile(
            masses,
            1.0 - alpha / 2.0,
        )
    )

    return BootstrapMassResult(
        estimate=float(np.mean(masses)),
        standard_error=float(
            np.std(
                masses,
                ddof=1,
            )
        ),
        lower_ci=lower,
        upper_ci=upper,
        successful_fits=len(masses),
        attempted_fits=n_bootstrap,
        bootstrap_masses=masses,
    )
