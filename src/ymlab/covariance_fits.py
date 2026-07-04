"""
Covariance estimation and correlated spectroscopy fitting.

Euclidean correlator values measured at different time separations are
statistically correlated because they are estimated from the same underlying
gauge configurations.

For a correlator vector C and model f(theta), the correlated objective is

    chi^2(theta)
        =
        [C - f(theta)]^T
        Sigma^{-1}
        [C - f(theta)],

where Sigma is the covariance matrix of the correlator estimator.

Finite ensembles can produce poorly conditioned covariance matrices. This
module therefore provides:

1. Configuration-bootstrap correlator replicates.
2. Sample covariance and correlation matrices.
3. Diagonal shrinkage regularization.
4. Eigenvalue-truncated covariance pseudoinverses.
5. Correlated periodic-cosh fitting.
6. Correlated chi-squared and parameter-covariance diagnostics.

This is finite-lattice numerical spectroscopy infrastructure. It does not prove
a continuum Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ymlab.glueball import ensemble_connected_correlator
from ymlab.spectroscopy import periodic_cosh_correlator


@dataclass(frozen=True)
class CovarianceEstimate:
    """Covariance and correlation summary."""

    mean: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    standard_deviations: np.ndarray
    eigenvalues: np.ndarray
    numerical_rank: int
    condition_number: float


@dataclass(frozen=True)
class InverseCovarianceResult:
    """Regularized covariance pseudoinverse."""

    covariance: np.ndarray
    inverse_covariance: np.ndarray
    eigenvalues: np.ndarray
    retained_mask: np.ndarray
    retained_eigenvalues: np.ndarray
    retained_rank: int
    eigenvalue_cutoff: float
    condition_number: float
    identity_projection_error: float


@dataclass(frozen=True)
class CorrelatedCoshFitResult:
    """Result of a correlated periodic-cosh fit."""

    amplitude: float
    mass: float
    amplitude_error: float
    mass_error: float
    parameter_covariance: np.ndarray
    fit_start: int
    fit_stop: int
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


def _validate_samples(
    samples: np.ndarray,
) -> np.ndarray:
    """Validate a sample-by-feature matrix."""
    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape "
            "(number_of_samples, number_of_features)."
        )

    if samples.shape[0] < 2:
        raise ValueError(
            "Need at least two samples."
        )

    if samples.shape[1] < 1:
        raise ValueError(
            "Need at least one feature."
        )

    if not np.all(
        np.isfinite(samples)
    ):
        raise ValueError(
            "samples must contain finite values."
        )

    return samples


def estimate_covariance(
    samples: np.ndarray,
) -> CovarianceEstimate:
    """
    Estimate sample covariance and correlation matrices.

    Samples are rows and measured components are columns.
    """
    samples = _validate_samples(
        samples
    )

    mean = np.mean(
        samples,
        axis=0,
    )

    covariance = np.atleast_2d(
        np.cov(
            samples,
            rowvar=False,
            ddof=1,
        )
    )

    covariance = 0.5 * (
        covariance
        + covariance.T
    )

    variances = np.diag(
        covariance
    )

    standard_deviations = np.sqrt(
        np.maximum(
            variances,
            0.0,
        )
    )

    denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = np.zeros_like(
        covariance
    )

    valid = denominator > 0.0

    correlation[valid] = (
        covariance[valid]
        / denominator[valid]
    )

    diagonal_indices = np.arange(
        covariance.shape[0]
    )

    correlation[
        diagonal_indices,
        diagonal_indices,
    ] = np.where(
        standard_deviations > 0.0,
        1.0,
        0.0,
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    positive = eigenvalues[
        eigenvalues > 0.0
    ]

    if len(positive) == 0:
        numerical_rank = 0
        condition_number = np.inf
    else:
        tolerance = max(
            np.finfo(float).eps
            * covariance.shape[0]
            * float(
                np.max(
                    np.abs(
                        eigenvalues
                    )
                )
            ),
            0.0,
        )

        retained = eigenvalues > tolerance
        numerical_rank = int(
            np.sum(
                retained
            )
        )

        retained_values = eigenvalues[
            retained
        ]

        condition_number = (
            float(
                np.max(
                    retained_values
                )
                / np.min(
                    retained_values
                )
            )
            if len(retained_values) > 0
            else np.inf
        )

    return CovarianceEstimate(
        mean=np.asarray(
            mean,
            dtype=float,
        ),
        covariance=np.asarray(
            covariance,
            dtype=float,
        ),
        correlation=np.asarray(
            correlation,
            dtype=float,
        ),
        standard_deviations=np.asarray(
            standard_deviations,
            dtype=float,
        ),
        eigenvalues=np.asarray(
            eigenvalues,
            dtype=float,
        ),
        numerical_rank=numerical_rank,
        condition_number=float(
            condition_number
        ),
    )


def shrink_covariance_to_diagonal(
    covariance: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """
    Apply linear shrinkage toward the covariance diagonal.

    Sigma_shrunk
        =
        (1 - alpha) Sigma
        +
        alpha diag(Sigma).

    alpha = 0 preserves the original covariance.

    alpha = 1 removes all estimated cross-covariance terms.
    """
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be two-dimensional."
        )

    if (
        covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance must be square."
        )

    if not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError(
            "covariance must contain finite values."
        )

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(
            "shrinkage must satisfy 0 <= shrinkage <= 1."
        )

    symmetric = 0.5 * (
        covariance
        + covariance.T
    )

    diagonal = np.diag(
        np.diag(
            symmetric
        )
    )

    shrunk = (
        (1.0 - shrinkage)
        * symmetric
        + shrinkage
        * diagonal
    )

    return 0.5 * (
        shrunk
        + shrunk.T
    )


def regularized_inverse_covariance(
    covariance: np.ndarray,
    relative_cutoff: float = 1e-10,
    absolute_cutoff: float = 1e-18,
) -> InverseCovarianceResult:
    """
    Construct an eigenvalue-truncated symmetric covariance pseudoinverse.

    Only covariance eigenmodes satisfying

        lambda > max(
            absolute_cutoff,
            relative_cutoff * largest_positive_lambda
        )

    are retained.
    """
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be two-dimensional."
        )

    if (
        covariance.shape[0]
        != covariance.shape[1]
    ):
        raise ValueError(
            "covariance must be square."
        )

    if not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError(
            "covariance must contain finite values."
        )

    if relative_cutoff < 0.0:
        raise ValueError(
            "relative_cutoff must be nonnegative."
        )

    if absolute_cutoff < 0.0:
        raise ValueError(
            "absolute_cutoff must be nonnegative."
        )

    symmetric = 0.5 * (
        covariance
        + covariance.T
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            symmetric
        )
    )

    positive = eigenvalues[
        eigenvalues > 0.0
    ]

    if len(positive) == 0:
        raise ValueError(
            "Covariance matrix has no positive eigenmodes."
        )

    largest_positive = float(
        np.max(
            positive
        )
    )

    cutoff = max(
        float(
            absolute_cutoff
        ),
        float(
            relative_cutoff
            * largest_positive
        ),
    )

    retained_mask = (
        eigenvalues > cutoff
    )

    retained_eigenvalues = eigenvalues[
        retained_mask
    ]

    retained_eigenvectors = eigenvectors[
        :,
        retained_mask,
    ]

    if len(retained_eigenvalues) == 0:
        raise ValueError(
            "Regularization removed every covariance mode."
        )

    inverse_covariance = (
        retained_eigenvectors
        @ np.diag(
            1.0
            / retained_eigenvalues
        )
        @ retained_eigenvectors.T
    )

    inverse_covariance = 0.5 * (
        inverse_covariance
        + inverse_covariance.T
    )

    projector = (
        retained_eigenvectors
        @ retained_eigenvectors.T
    )

    reconstructed_projector = (
        symmetric
        @ inverse_covariance
    )

    identity_projection_error = float(
        np.max(
            np.abs(
                reconstructed_projector
                - projector
            )
        )
    )

    condition_number = float(
        np.max(
            retained_eigenvalues
        )
        / np.min(
            retained_eigenvalues
        )
    )

    return InverseCovarianceResult(
        covariance=np.asarray(
            symmetric,
            dtype=float,
        ),
        inverse_covariance=np.asarray(
            inverse_covariance,
            dtype=float,
        ),
        eigenvalues=np.asarray(
            eigenvalues,
            dtype=float,
        ),
        retained_mask=np.asarray(
            retained_mask,
            dtype=bool,
        ),
        retained_eigenvalues=np.asarray(
            retained_eigenvalues,
            dtype=float,
        ),
        retained_rank=len(
            retained_eigenvalues
        ),
        eigenvalue_cutoff=float(
            cutoff
        ),
        condition_number=condition_number,
        identity_projection_error=(
            identity_projection_error
        ),
    )


def correlated_chi_squared(
    residuals: np.ndarray,
    inverse_covariance: np.ndarray,
) -> float:
    """
    Evaluate r^T Sigma^{-1} r.
    """
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    inverse_covariance = np.asarray(
        inverse_covariance,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError(
            "residuals must be one-dimensional."
        )

    if inverse_covariance.shape != (
        len(residuals),
        len(residuals),
    ):
        raise ValueError(
            "inverse_covariance shape does not match residuals."
        )

    value = (
        residuals.T
        @ inverse_covariance
        @ residuals
    )

    return float(
        value
    )


def bootstrap_connected_correlators(
    operator_series_ensemble: np.ndarray,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate configuration-bootstrap connected-correlator replicates.

    The output shape is

        (n_bootstrap, temporal_extent).

    Entire configuration time-series vectors are resampled with replacement.
    """
    ensemble = np.asarray(
        operator_series_ensemble,
        dtype=float,
    )

    if ensemble.ndim != 2:
        raise ValueError(
            "Expected ensemble shape "
            "(configurations, temporal_extent)."
        )

    number_of_configurations = ensemble.shape[0]

    if number_of_configurations < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if n_bootstrap <= 1:
        raise ValueError(
            "n_bootstrap must be greater than one."
        )

    if not np.all(
        np.isfinite(ensemble)
    ):
        raise ValueError(
            "Ensemble must contain finite values."
        )

    rng = np.random.default_rng(
        seed
    )

    replicates = []

    for _ in range(
        n_bootstrap
    ):
        indices = rng.integers(
            0,
            number_of_configurations,
            size=number_of_configurations,
        )

        resampled = ensemble[
            indices
        ]

        correlator = (
            ensemble_connected_correlator(
                resampled
            )
        )

        replicates.append(
            correlator.correlation
        )

    return np.asarray(
        replicates,
        dtype=float,
    )


def _periodic_cosh_jacobian(
    lags: np.ndarray,
    amplitude: float,
    mass: float,
    temporal_extent: int,
) -> np.ndarray:
    """
    Analytic Jacobian of the periodic cosh-like model with respect to A and m.
    """
    lags = np.asarray(
        lags,
        dtype=float,
    )

    backward_distance = (
        float(
            temporal_extent
        )
        - lags
    )

    forward = np.exp(
        -mass * lags
    )

    backward = np.exp(
        -mass
        * backward_distance
    )

    derivative_amplitude = (
        forward + backward
    )

    derivative_mass = amplitude * (
        -lags * forward
        - backward_distance
        * backward
    )

    return np.column_stack(
        [
            derivative_amplitude,
            derivative_mass,
        ]
    )


def fit_correlated_periodic_cosh(
    correlation: np.ndarray,
    covariance: np.ndarray,
    fit_start: int = 1,
    fit_stop: Optional[int] = None,
    shrinkage: float = 0.05,
    relative_cutoff: float = 1e-10,
    absolute_cutoff: float = 1e-18,
) -> CorrelatedCoshFitResult:
    """
    Fit a periodic single-state model using a correlated chi-squared objective.

    The covariance matrix is first restricted to the fit window, shrunk toward
    its diagonal, and then pseudoinverted with eigenvalue truncation.
    """
    correlation = np.asarray(
        correlation,
        dtype=float,
    )

    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if correlation.ndim != 1:
        raise ValueError(
            "correlation must be one-dimensional."
        )

    temporal_extent = len(
        correlation
    )

    if covariance.shape != (
        temporal_extent,
        temporal_extent,
    ):
        raise ValueError(
            "covariance shape must match the correlator length."
        )

    if not np.all(
        np.isfinite(
            correlation
        )
    ):
        raise ValueError(
            "correlation must contain finite values."
        )

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

    if fit_stop - fit_start < 3:
        raise ValueError(
            "Correlated fit window must contain "
            "at least three points."
        )

    fit_lags = np.arange(
        fit_start,
        fit_stop,
        dtype=float,
    )

    fit_values = correlation[
        fit_start:fit_stop
    ]

    if np.any(
        fit_values <= 0.0
    ):
        raise ValueError(
            "Correlated periodic-cosh fit requires "
            "positive correlator values in the fit window."
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

    initial_amplitude = max(
        float(
            fit_values[0]
        )
        / 2.0,
        np.finfo(float).eps,
    )

    initial_mass = 0.5

    def decode(
        parameters: np.ndarray,
    ) -> tuple[float, float]:
        amplitude = float(
            np.exp(
                parameters[0]
            )
        )

        mass = float(
            np.exp(
                parameters[1]
            )
        )

        return amplitude, mass

    def objective(
        parameters: np.ndarray,
    ) -> float:
        amplitude, mass = decode(
            parameters
        )

        model_values = periodic_cosh_correlator(
            lag=fit_lags,
            amplitude=amplitude,
            mass=mass,
            temporal_extent=temporal_extent,
        )

        residuals = (
            fit_values
            - model_values
        )

        return correlated_chi_squared(
            residuals=residuals,
            inverse_covariance=inverse_covariance,
        )

    optimization = minimize(
        objective,
        x0=np.log(
            [
                initial_amplitude,
                initial_mass,
            ]
        ),
        method="L-BFGS-B",
        bounds=[
            (-50.0, 50.0),
            (-20.0, 10.0),
        ],
    )

    amplitude, mass = decode(
        optimization.x
    )

    model_values = periodic_cosh_correlator(
        lag=fit_lags,
        amplitude=amplitude,
        mass=mass,
        temporal_extent=temporal_extent,
    )

    residuals = (
        fit_values
        - model_values
    )

    chi_squared = correlated_chi_squared(
        residuals=residuals,
        inverse_covariance=inverse_covariance,
    )

    jacobian = _periodic_cosh_jacobian(
        lags=fit_lags,
        amplitude=amplitude,
        mass=mass,
        temporal_extent=temporal_extent,
    )

    fisher = (
        jacobian.T
        @ inverse_covariance
        @ jacobian
    )

    parameter_covariance = np.linalg.pinv(
        fisher,
        rcond=1e-12,
    )

    parameter_covariance = 0.5 * (
        parameter_covariance
        + parameter_covariance.T
    )

    parameter_variances = np.diag(
        parameter_covariance
    )

    amplitude_error = (
        float(
            np.sqrt(
                parameter_variances[0]
            )
        )
        if parameter_variances[0] >= 0.0
        else np.nan
    )

    mass_error = (
        float(
            np.sqrt(
                parameter_variances[1]
            )
        )
        if parameter_variances[1] >= 0.0
        else np.nan
    )

    effective_degrees_of_freedom = (
        inverse_result.retained_rank
        - 2
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
            amplitude
        )
        and np.isfinite(
            mass
        )
        and amplitude > 0.0
        and mass > 0.0
    )

    return CorrelatedCoshFitResult(
        amplitude=amplitude,
        mass=mass,
        amplitude_error=amplitude_error,
        mass_error=mass_error,
        parameter_covariance=np.asarray(
            parameter_covariance,
            dtype=float,
        ),
        fit_start=fit_start,
        fit_stop=fit_stop,
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
        chi_squared=float(
            chi_squared
        ),
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
