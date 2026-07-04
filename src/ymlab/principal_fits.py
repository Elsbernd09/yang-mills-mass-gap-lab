"""
Covariance-aware fitting of GEVP principal correlators.

For a generalized-eigenvalue problem

    C(t) v_n = lambda_n(t, t0) C(t0) v_n,

the principal correlators satisfy

    lambda_n(t0, t0) = 1

for retained variational states.

A periodic single-state model can therefore be normalized at the reference
time:

    lambda(t; m, t0, T)
        =
        [
            exp(-m t)
            + exp(-m (T - t))
        ]
        /
        [
            exp(-m t0)
            + exp(-m (T - t0))
        ].

This removes the free amplitude parameter.

The module provides:

1. normalized periodic principal-correlator models,
2. bootstrap covariance estimation for matched principal states,
3. shrinkage and regularized inverse covariance,
4. correlated one-parameter mass fits,
5. fit-window scans,
6. bootstrap-refit mass distributions.

This is exploratory finite-lattice variational spectroscopy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from ymlab.covariance_fits import (
    CovarianceEstimate,
    estimate_covariance,
    regularized_inverse_covariance,
    shrink_covariance_to_diagonal,
)


@dataclass(frozen=True)
class CorrelatedPrincipalFitResult:
    """Correlated normalized principal-correlator fit."""

    mass: float
    mass_error: float
    parameter_variance: float
    fit_start: int
    fit_stop: int
    reference_time: int
    temporal_extent: int
    fit_lags: np.ndarray
    fit_values: np.ndarray
    model_values: np.ndarray
    residuals: np.ndarray
    chi_squared: float
    effective_degrees_of_freedom: int
    reduced_chi_squared: float
    covariance_rank: int
    covariance_condition_number: float
    success: bool
    optimizer_message: str


@dataclass(frozen=True)
class PrincipalCovarianceResult:
    """Bootstrap covariance for one matched principal state."""

    state: int
    estimate: CovarianceEstimate
    finite_replicates: int


@dataclass(frozen=True)
class PrincipalBootstrapFitSummary:
    """Bootstrap-refit mass distribution summary."""

    central_mass: float
    bootstrap_masses: np.ndarray
    accepted_fits: int
    rejected_fits: int
    requested_fits: int
    mean_mass: float
    standard_error: float
    lower_95: float
    median_mass: float
    upper_95: float


def normalized_periodic_principal_correlator(
    lag: np.ndarray | float,
    mass: float,
    temporal_extent: int,
    reference_time: int,
) -> np.ndarray:
    """
    Evaluate a periodic single-state principal-correlator model normalized at t0.
    """
    if mass <= 0.0:
        raise ValueError(
            "mass must be positive."
        )

    if temporal_extent < 3:
        raise ValueError(
            "temporal_extent must be at least three."
        )

    if (
        reference_time < 0
        or reference_time >= temporal_extent
    ):
        raise ValueError(
            "Invalid reference_time."
        )

    lag = np.asarray(
        lag,
        dtype=float,
    )

    numerator = (
        np.exp(
            -mass * lag
        )
        + np.exp(
            -mass
            * (
                temporal_extent - lag
            )
        )
    )

    denominator = (
        np.exp(
            -mass * reference_time
        )
        + np.exp(
            -mass
            * (
                temporal_extent - reference_time
            )
        )
    )

    return numerator / denominator


def principal_state_covariance(
    principal_samples: np.ndarray,
    state: int,
) -> PrincipalCovarianceResult:
    """
    Estimate time-time covariance for one matched bootstrap principal state.

    Input shape:

        (
            bootstrap_replicates,
            temporal_extent,
            states,
        ).
    """
    samples = np.asarray(
        principal_samples,
        dtype=float,
    )

    if samples.ndim != 3:
        raise ValueError(
            "Expected principal_samples shape "
            "(bootstrap, temporal_extent, states)."
        )

    if (
        state < 0
        or state >= samples.shape[2]
    ):
        raise ValueError(
            "Invalid state index."
        )

    state_samples = samples[
        :,
        :,
        state,
    ]

    finite_rows = np.all(
        np.isfinite(
            state_samples
        ),
        axis=1,
    )

    state_samples = state_samples[
        finite_rows
    ]

    if len(state_samples) < 2:
        raise ValueError(
            "Need at least two fully finite bootstrap replicates."
        )

    estimate = estimate_covariance(
        state_samples
    )

    return PrincipalCovarianceResult(
        state=int(
            state
        ),
        estimate=estimate,
        finite_replicates=len(
            state_samples
        ),
    )


def _principal_model_derivative_mass(
    lags: np.ndarray,
    mass: float,
    temporal_extent: int,
    reference_time: int,
) -> np.ndarray:
    """
    Analytic derivative of the normalized periodic principal model with respect
    to mass.
    """
    lags = np.asarray(
        lags,
        dtype=float,
    )

    temporal_extent_float = float(
        temporal_extent
    )

    forward = np.exp(
        -mass * lags
    )

    backward_distance = (
        temporal_extent_float
        - lags
    )

    backward = np.exp(
        -mass
        * backward_distance
    )

    numerator = (
        forward + backward
    )

    numerator_derivative = (
        -lags * forward
        - backward_distance
        * backward
    )

    reference_forward = np.exp(
        -mass * reference_time
    )

    reference_backward_distance = (
        temporal_extent_float
        - reference_time
    )

    reference_backward = np.exp(
        -mass
        * reference_backward_distance
    )

    denominator = (
        reference_forward
        + reference_backward
    )

    denominator_derivative = (
        -reference_time
        * reference_forward
        - reference_backward_distance
        * reference_backward
    )

    return (
        numerator_derivative
        * denominator
        - numerator
        * denominator_derivative
    ) / (
        denominator ** 2
    )


def fit_correlated_principal_mass(
    principal_correlator: np.ndarray,
    covariance: np.ndarray,
    reference_time: int,
    fit_start: int,
    fit_stop: int,
    shrinkage: float = 0.10,
    relative_cutoff: float = 1e-10,
    absolute_cutoff: float = 1e-18,
    mass_bounds: tuple[float, float] = (
        1e-6,
        10.0,
    ),
) -> CorrelatedPrincipalFitResult:
    """
    Fit one normalized principal correlator using correlated chi-squared.
    """
    principal = np.asarray(
        principal_correlator,
        dtype=float,
    )

    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if principal.ndim != 1:
        raise ValueError(
            "principal_correlator must be one-dimensional."
        )

    temporal_extent = len(
        principal
    )

    if covariance.shape != (
        temporal_extent,
        temporal_extent,
    ):
        raise ValueError(
            "covariance shape must match principal-correlator length."
        )

    if not np.all(
        np.isfinite(
            principal
        )
    ):
        raise ValueError(
            "principal_correlator must contain finite values."
        )

    if (
        reference_time < 0
        or reference_time >= temporal_extent
    ):
        raise ValueError(
            "Invalid reference_time."
        )

    if fit_start < 0:
        raise ValueError(
            "fit_start must be nonnegative."
        )

    if fit_stop > temporal_extent:
        raise ValueError(
            "fit_stop exceeds temporal extent."
        )

    if fit_stop - fit_start < 2:
        raise ValueError(
            "Principal fit window must contain at least two points."
        )

    lower_mass, upper_mass = (
        mass_bounds
    )

    if (
        lower_mass <= 0.0
        or upper_mass <= lower_mass
    ):
        raise ValueError(
            "Invalid positive mass_bounds."
        )

    fit_lags = np.arange(
        fit_start,
        fit_stop,
        dtype=float,
    )

    fit_values = principal[
        fit_start:fit_stop
    ]

    if np.any(
        fit_values <= 0.0
    ):
        raise ValueError(
            "Principal fit requires positive values in the fit window."
        )

    fit_covariance = covariance[
        fit_start:fit_stop,
        fit_start:fit_stop,
    ]

    shrunk_covariance = (
        shrink_covariance_to_diagonal(
            covariance=fit_covariance,
            shrinkage=shrinkage,
        )
    )

    inverse_result = (
        regularized_inverse_covariance(
            covariance=shrunk_covariance,
            relative_cutoff=relative_cutoff,
            absolute_cutoff=absolute_cutoff,
        )
    )

    inverse_covariance = (
        inverse_result.inverse_covariance
    )

    def objective(
        mass: float,
    ) -> float:
        model = (
            normalized_periodic_principal_correlator(
                lag=fit_lags,
                mass=mass,
                temporal_extent=temporal_extent,
                reference_time=reference_time,
            )
        )

        residuals = (
            fit_values - model
        )

        return float(
            residuals.T
            @ inverse_covariance
            @ residuals
        )

    optimization = minimize_scalar(
        objective,
        bounds=(
            lower_mass,
            upper_mass,
        ),
        method="bounded",
    )

    mass = float(
        optimization.x
    )

    model_values = (
        normalized_periodic_principal_correlator(
            lag=fit_lags,
            mass=mass,
            temporal_extent=temporal_extent,
            reference_time=reference_time,
        )
    )

    residuals = (
        fit_values
        - model_values
    )

    chi_squared = float(
        residuals.T
        @ inverse_covariance
        @ residuals
    )

    derivative = (
        _principal_model_derivative_mass(
            lags=fit_lags,
            mass=mass,
            temporal_extent=temporal_extent,
            reference_time=reference_time,
        )
    )

    information = float(
        derivative.T
        @ inverse_covariance
        @ derivative
    )

    if (
        np.isfinite(
            information
        )
        and information > 0.0
    ):
        parameter_variance = (
            1.0 / information
        )

        mass_error = float(
            np.sqrt(
                parameter_variance
            )
        )
    else:
        parameter_variance = np.nan
        mass_error = np.nan

    effective_degrees_of_freedom = (
        inverse_result.retained_rank
        - 1
    )

    reduced_chi_squared = (
        chi_squared
        / effective_degrees_of_freedom
        if effective_degrees_of_freedom > 0
        else np.nan
    )

    success = bool(
        optimization.success
        and np.isfinite(
            mass
        )
        and mass > 0.0
    )

    return CorrelatedPrincipalFitResult(
        mass=mass,
        mass_error=mass_error,
        parameter_variance=float(
            parameter_variance
        ),
        fit_start=int(
            fit_start
        ),
        fit_stop=int(
            fit_stop
        ),
        reference_time=int(
            reference_time
        ),
        temporal_extent=int(
            temporal_extent
        ),
        fit_lags=np.asarray(
            fit_lags,
            dtype=float,
        ),
        fit_values=np.asarray(
            fit_values,
            dtype=float,
        ),
        model_values=np.asarray(
            model_values,
            dtype=float,
        ),
        residuals=np.asarray(
            residuals,
            dtype=float,
        ),
        chi_squared=chi_squared,
        effective_degrees_of_freedom=int(
            effective_degrees_of_freedom
        ),
        reduced_chi_squared=float(
            reduced_chi_squared
        ),
        covariance_rank=(
            inverse_result.retained_rank
        ),
        covariance_condition_number=(
            inverse_result.condition_number
        ),
        success=success,
        optimizer_message=str(
            optimization.message
        ),
    )


def scan_principal_fit_windows(
    principal_correlator: np.ndarray,
    covariance: np.ndarray,
    reference_time: int,
    shrinkage: float = 0.10,
    relative_cutoff: float = 1e-10,
    minimum_points: int = 3,
) -> list[
    CorrelatedPrincipalFitResult
]:
    """
    Scan positive principal-correlator fit windows.

    Successful fits are sorted by:

    1. distance of reduced chi-squared from one,
    2. longer fit window,
    3. smaller mass error.
    """
    principal = np.asarray(
        principal_correlator,
        dtype=float,
    )

    temporal_extent = len(
        principal
    )

    if minimum_points < 2:
        raise ValueError(
            "minimum_points must be at least two."
        )

    maximum_stop = (
        temporal_extent // 2
        + 1
    )

    fits = []

    candidate_starts = sorted(
        set(
            [
                reference_time,
                reference_time + 1,
                1,
                2,
            ]
        )
    )

    for fit_start in candidate_starts:
        if fit_start < 0:
            continue

        for fit_stop in range(
            fit_start + minimum_points,
            maximum_stop + 1,
        ):
            try:
                fit = fit_correlated_principal_mass(
                    principal_correlator=principal,
                    covariance=covariance,
                    reference_time=reference_time,
                    fit_start=fit_start,
                    fit_stop=fit_stop,
                    shrinkage=shrinkage,
                    relative_cutoff=relative_cutoff,
                )
            except ValueError:
                continue

            if fit.success:
                fits.append(
                    fit
                )

    def fit_key(
        fit: CorrelatedPrincipalFitResult,
    ) -> tuple[float, int, float]:
        reduced_score = (
            abs(
                fit.reduced_chi_squared
                - 1.0
            )
            if np.isfinite(
                fit.reduced_chi_squared
            )
            else np.inf
        )

        length_score = -(
            fit.fit_stop
            - fit.fit_start
        )

        error_score = (
            fit.mass_error
            if np.isfinite(
                fit.mass_error
            )
            else np.inf
        )

        return (
            reduced_score,
            length_score,
            error_score,
        )

    fits.sort(
        key=fit_key
    )

    return fits


def bootstrap_refit_principal_mass(
    principal_samples: np.ndarray,
    central_fit: CorrelatedPrincipalFitResult,
    covariance: np.ndarray,
    shrinkage: float = 0.10,
    relative_cutoff: float = 1e-10,
) -> PrincipalBootstrapFitSummary:
    """
    Refit every finite bootstrap principal-correlator replicate in a fixed
    central fit window.

    The same covariance matrix and fit window are used for all replicates.
    """
    samples = np.asarray(
        principal_samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "Expected principal_samples shape "
            "(bootstrap, temporal_extent)."
        )

    if covariance.shape != (
        samples.shape[1],
        samples.shape[1],
    ):
        raise ValueError(
            "covariance shape does not match principal samples."
        )

    masses = []
    rejected = 0

    for sample in samples:
        if not np.all(
            np.isfinite(
                sample
            )
        ):
            rejected += 1
            continue

        try:
            fit = fit_correlated_principal_mass(
                principal_correlator=sample,
                covariance=covariance,
                reference_time=(
                    central_fit.reference_time
                ),
                fit_start=(
                    central_fit.fit_start
                ),
                fit_stop=(
                    central_fit.fit_stop
                ),
                shrinkage=shrinkage,
                relative_cutoff=relative_cutoff,
            )
        except ValueError:
            rejected += 1
            continue

        if not fit.success:
            rejected += 1
            continue

        masses.append(
            fit.mass
        )

    if len(masses) < 2:
        raise ValueError(
            "Fewer than two bootstrap principal fits were accepted."
        )

    masses = np.asarray(
        masses,
        dtype=float,
    )

    return PrincipalBootstrapFitSummary(
        central_mass=float(
            central_fit.mass
        ),
        bootstrap_masses=masses,
        accepted_fits=len(
            masses
        ),
        rejected_fits=int(
            rejected
        ),
        requested_fits=samples.shape[0],
        mean_mass=float(
            np.mean(
                masses
            )
        ),
        standard_error=float(
            np.std(
                masses,
                ddof=1,
            )
        ),
        lower_95=float(
            np.quantile(
                masses,
                0.025,
            )
        ),
        median_mass=float(
            np.quantile(
                masses,
                0.5,
            )
        ),
        upper_95=float(
            np.quantile(
                masses,
                0.975,
            )
        ),
    )
