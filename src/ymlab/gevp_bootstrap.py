"""
Bootstrap uncertainty analysis for generalized-eigenvalue spectroscopy.

The central variational analysis solves

    C(t) v_n(t, t0)
        =
        lambda_n(t, t0) C(t0) v_n(t, t0).

A central GEVP solution alone does not quantify uncertainty in:

1. principal correlators,
2. variational effective masses,
3. generalized eigenvectors,
4. numerical retained rank,
5. state identity across resampled ensembles.

This module resamples complete configuration-level operator measurements,
reconstructs the full correlator matrix, resolves the regularized GEVP, and
matches bootstrap states to the central variational solution.

State matching is performed in the original operator basis using a
metric-normalized overlap defined by the central reference correlator.

For central vector v and bootstrap vector w,

    overlap(v, w)
        =
        |v^T C_ref w|
        /
        sqrt[
            (v^T C_ref v)
            (w^T C_ref w)
        ].

A one-to-one maximum-overlap assignment is selected using the Hungarian
algorithm.

Bootstrap replicates whose retained numerical rank differs from the central
GEVP are recorded and rejected from fixed-rank state uncertainty summaries.

Both iid configuration bootstrap and circular block bootstrap are supported.

This is finite-lattice numerical uncertainty analysis. It does not establish a
continuum Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ymlab.operator_basis import (
    ensemble_correlator_matrix,
    symmetrize_correlator_matrix,
)
from ymlab.variational import (
    GEVPResult,
    principal_arccosh_effective_masses,
    principal_log_effective_masses,
    solve_regularized_gevp,
)


BootstrapMode = Literal[
    "iid",
    "circular_block",
]


@dataclass(frozen=True)
class StateMatchResult:
    """One state-assignment comparison."""

    permutation: np.ndarray
    overlap_matrix: np.ndarray
    matched_overlaps: np.ndarray
    matched_vectors: np.ndarray
    matched_values: np.ndarray


@dataclass(frozen=True)
class GEVPBootstrapResult:
    """Bootstrap distributions for variational spectroscopy."""

    central_result: GEVPResult
    principal_samples: np.ndarray
    log_mass_samples: np.ndarray
    arccosh_mass_samples: np.ndarray
    overlap_samples: np.ndarray
    accepted_replicates: int
    rejected_rank_replicates: int
    rejected_numerical_replicates: int
    requested_replicates: int
    bootstrap_mode: str
    block_size: int
    central_rank: int


@dataclass(frozen=True)
class BootstrapInterval:
    """Pointwise bootstrap summary."""

    mean: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    finite_counts: np.ndarray
    confidence_level: float


def _validate_operator_ensemble(
    operator_ensemble: np.ndarray,
) -> np.ndarray:
    """Validate configuration-by-operator-by-time data."""
    ensemble = np.asarray(
        operator_ensemble,
        dtype=float,
    )

    if ensemble.ndim != 3:
        raise ValueError(
            "Expected operator ensemble shape "
            "(configurations, operators, temporal_extent)."
        )

    if ensemble.shape[0] < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if ensemble.shape[1] < 1:
        raise ValueError(
            "Need at least one operator."
        )

    if ensemble.shape[2] < 3:
        raise ValueError(
            "Need at least three Euclidean time slices."
        )

    if not np.all(
        np.isfinite(
            ensemble
        )
    ):
        raise ValueError(
            "Operator ensemble must contain finite values."
        )

    return ensemble


def circular_block_bootstrap_indices(
    number_of_configurations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one circular moving-block bootstrap index sample.

    Complete consecutive configuration blocks are sampled with replacement.
    The final concatenated sample is truncated to the original chain length.
    """
    if number_of_configurations < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if block_size < 1:
        raise ValueError(
            "block_size must be positive."
        )

    if block_size > number_of_configurations:
        raise ValueError(
            "block_size cannot exceed the number of configurations."
        )

    number_of_blocks = int(
        np.ceil(
            number_of_configurations
            / block_size
        )
    )

    blocks = []

    for _ in range(
        number_of_blocks
    ):
        start = int(
            rng.integers(
                0,
                number_of_configurations,
            )
        )

        block = (
            start
            + np.arange(
                block_size,
                dtype=int,
            )
        ) % number_of_configurations

        blocks.append(
            block
        )

    return np.concatenate(
        blocks
    )[:number_of_configurations]


def bootstrap_configuration_indices(
    number_of_configurations: int,
    rng: np.random.Generator,
    mode: BootstrapMode = "iid",
    block_size: int = 1,
) -> np.ndarray:
    """
    Generate one configuration-level bootstrap index sample.
    """
    if number_of_configurations < 2:
        raise ValueError(
            "Need at least two configurations."
        )

    if mode == "iid":
        return rng.integers(
            0,
            number_of_configurations,
            size=number_of_configurations,
        )

    if mode == "circular_block":
        return circular_block_bootstrap_indices(
            number_of_configurations=(
                number_of_configurations
            ),
            block_size=block_size,
            rng=rng,
        )

    raise ValueError(
        "mode must be 'iid' or 'circular_block'."
    )


def metric_overlap_matrix(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    metric_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute normalized overlaps in a supplied operator-space metric.

    Output shape:

        (
            number_of_reference_states,
            number_of_candidate_states,
        ).
    """
    reference_vectors = np.asarray(
        reference_vectors,
        dtype=float,
    )

    candidate_vectors = np.asarray(
        candidate_vectors,
        dtype=float,
    )

    metric_matrix = np.asarray(
        metric_matrix,
        dtype=float,
    )

    if (
        reference_vectors.ndim != 2
        or candidate_vectors.ndim != 2
    ):
        raise ValueError(
            "State-vector arrays must be two-dimensional."
        )

    operator_dimension = (
        reference_vectors.shape[0]
    )

    if (
        candidate_vectors.shape[0]
        != operator_dimension
    ):
        raise ValueError(
            "Reference and candidate vectors must use "
            "the same operator dimension."
        )

    if metric_matrix.shape != (
        operator_dimension,
        operator_dimension,
    ):
        raise ValueError(
            "Metric matrix shape does not match operator dimension."
        )

    symmetric_metric = 0.5 * (
        metric_matrix
        + metric_matrix.T
    )

    numerator = np.abs(
        reference_vectors.T
        @ symmetric_metric
        @ candidate_vectors
    )

    reference_norm_squared = np.diag(
        reference_vectors.T
        @ symmetric_metric
        @ reference_vectors
    )

    candidate_norm_squared = np.diag(
        candidate_vectors.T
        @ symmetric_metric
        @ candidate_vectors
    )

    if np.any(
        reference_norm_squared <= 0.0
    ):
        raise ValueError(
            "Reference vectors have nonpositive metric norm."
        )

    if np.any(
        candidate_norm_squared <= 0.0
    ):
        raise ValueError(
            "Candidate vectors have nonpositive metric norm."
        )

    denominator = np.sqrt(
        np.outer(
            reference_norm_squared,
            candidate_norm_squared,
        )
    )

    return numerator / denominator


def match_states_to_reference(
    reference_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_values: np.ndarray,
    metric_matrix: np.ndarray,
) -> StateMatchResult:
    """
    Match candidate states to central states by maximum metric overlap.

    Candidate eigenvector signs are aligned after assignment.
    """
    candidate_values = np.asarray(
        candidate_values,
        dtype=float,
    )

    if (
        reference_vectors.shape[1]
        != candidate_vectors.shape[1]
    ):
        raise ValueError(
            "State matching requires equal state counts."
        )

    number_of_states = (
        reference_vectors.shape[1]
    )

    if candidate_values.shape != (
        number_of_states,
    ):
        raise ValueError(
            "candidate_values shape does not match state count."
        )

    overlaps = metric_overlap_matrix(
        reference_vectors=reference_vectors,
        candidate_vectors=candidate_vectors,
        metric_matrix=metric_matrix,
    )

    rows, columns = (
        linear_sum_assignment(
            -overlaps
        )
    )

    permutation = np.empty(
        number_of_states,
        dtype=int,
    )

    permutation[rows] = columns

    matched_vectors = np.array(
        candidate_vectors[
            :,
            permutation,
        ],
        dtype=float,
        copy=True,
    )

    matched_values = np.asarray(
        candidate_values[
            permutation
        ],
        dtype=float,
    )

    symmetric_metric = 0.5 * (
        metric_matrix
        + metric_matrix.T
    )

    for state in range(
        number_of_states
    ):
        signed_overlap = float(
            reference_vectors[
                :,
                state,
            ].T
            @ symmetric_metric
            @ matched_vectors[
                :,
                state,
            ]
        )

        if signed_overlap < 0.0:
            matched_vectors[
                :,
                state,
            ] *= -1.0

    matched_overlaps = overlaps[
        np.arange(
            number_of_states
        ),
        permutation,
    ]

    return StateMatchResult(
        permutation=np.asarray(
            permutation,
            dtype=int,
        ),
        overlap_matrix=np.asarray(
            overlaps,
            dtype=float,
        ),
        matched_overlaps=np.asarray(
            matched_overlaps,
            dtype=float,
        ),
        matched_vectors=np.asarray(
            matched_vectors,
            dtype=float,
        ),
        matched_values=np.asarray(
            matched_values,
            dtype=float,
        ),
    )


def bootstrap_gevp(
    operator_ensemble: np.ndarray,
    n_bootstrap: int = 500,
    reference_time: int = 1,
    relative_cutoff: float = 1e-8,
    absolute_cutoff: float = 1e-14,
    mode: BootstrapMode = "iid",
    block_size: int = 1,
    seed: Optional[int] = None,
) -> GEVPBootstrapResult:
    """
    Bootstrap the complete correlator-matrix and GEVP pipeline.

    Every accepted replicate performs:

    1. configuration resampling,
    2. correlator-matrix reconstruction,
    3. matrix symmetrization,
    4. reference-metric regularization,
    5. GEVP solution,
    6. state matching to the central solution,
    7. principal-correlator effective-mass reconstruction.

    Replicates with retained rank different from the central solution are not
    inserted into fixed-state uncertainty arrays.
    """
    ensemble = _validate_operator_ensemble(
        operator_ensemble
    )

    if n_bootstrap <= 1:
        raise ValueError(
            "n_bootstrap must be greater than one."
        )

    if mode == "iid":
        effective_block_size = 1
    else:
        if block_size < 1:
            raise ValueError(
                "block_size must be positive."
            )

        effective_block_size = int(
            block_size
        )

    central_matrix_result = (
        ensemble_correlator_matrix(
            ensemble
        )
    )

    central_matrices = (
        symmetrize_correlator_matrix(
            central_matrix_result.correlation_matrices
        )
    )

    central_result = solve_regularized_gevp(
        correlation_matrices=central_matrices,
        reference_time=reference_time,
        relative_cutoff=relative_cutoff,
        absolute_cutoff=absolute_cutoff,
        track_states=True,
    )

    central_rank = (
        central_result.retained_rank
    )

    central_metric_matrix = (
        central_matrices[
            reference_time
        ]
    )

    rng = np.random.default_rng(
        seed
    )

    principal_samples = []
    log_mass_samples = []
    arccosh_mass_samples = []
    overlap_samples = []

    rejected_rank = 0
    rejected_numerical = 0

    number_of_configurations = (
        ensemble.shape[0]
    )

    for _ in range(
        n_bootstrap
    ):
        indices = bootstrap_configuration_indices(
            number_of_configurations=(
                number_of_configurations
            ),
            rng=rng,
            mode=mode,
            block_size=effective_block_size,
        )

        resampled = ensemble[
            indices
        ]

        try:
            matrix_result = (
                ensemble_correlator_matrix(
                    resampled
                )
            )

            matrices = (
                symmetrize_correlator_matrix(
                    matrix_result.correlation_matrices
                )
            )

            bootstrap_result = (
                solve_regularized_gevp(
                    correlation_matrices=matrices,
                    reference_time=reference_time,
                    relative_cutoff=relative_cutoff,
                    absolute_cutoff=absolute_cutoff,
                    track_states=True,
                )
            )
        except (
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            rejected_numerical += 1
            continue

        if (
            bootstrap_result.retained_rank
            != central_rank
        ):
            rejected_rank += 1
            continue

        matched_principal = np.zeros_like(
            bootstrap_result.principal_correlators
        )

        replicate_overlaps = np.zeros_like(
            bootstrap_result.principal_correlators
        )

        matching_failed = False

        for lag in range(
            bootstrap_result.principal_correlators.shape[0]
        ):
            try:
                match = match_states_to_reference(
                    reference_vectors=(
                        central_result.generalized_eigenvectors[
                            lag
                        ]
                    ),
                    candidate_vectors=(
                        bootstrap_result.generalized_eigenvectors[
                            lag
                        ]
                    ),
                    candidate_values=(
                        bootstrap_result.principal_correlators[
                            lag
                        ]
                    ),
                    metric_matrix=central_metric_matrix,
                )
            except ValueError:
                matching_failed = True
                break

            matched_principal[
                lag
            ] = match.matched_values

            replicate_overlaps[
                lag
            ] = match.matched_overlaps

        if matching_failed:
            rejected_numerical += 1
            continue

        principal_samples.append(
            matched_principal
        )

        log_mass_samples.append(
            principal_log_effective_masses(
                matched_principal
            )
        )

        arccosh_mass_samples.append(
            principal_arccosh_effective_masses(
                matched_principal
            )
        )

        overlap_samples.append(
            replicate_overlaps
        )

    if len(
        principal_samples
    ) == 0:
        raise ValueError(
            "No bootstrap GEVP replicates were accepted."
        )

    return GEVPBootstrapResult(
        central_result=central_result,
        principal_samples=np.asarray(
            principal_samples,
            dtype=float,
        ),
        log_mass_samples=np.asarray(
            log_mass_samples,
            dtype=float,
        ),
        arccosh_mass_samples=np.asarray(
            arccosh_mass_samples,
            dtype=float,
        ),
        overlap_samples=np.asarray(
            overlap_samples,
            dtype=float,
        ),
        accepted_replicates=len(
            principal_samples
        ),
        rejected_rank_replicates=(
            rejected_rank
        ),
        rejected_numerical_replicates=(
            rejected_numerical
        ),
        requested_replicates=n_bootstrap,
        bootstrap_mode=str(
            mode
        ),
        block_size=effective_block_size,
        central_rank=central_rank,
    )


def summarize_bootstrap_distribution(
    samples: np.ndarray,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """
    Compute pointwise finite-value bootstrap summaries.

    NaN values are ignored independently at each array position.
    """
    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim < 2:
        raise ValueError(
            "samples must contain a bootstrap axis "
            "and at least one measured axis."
        )

    if samples.shape[0] < 2:
        raise ValueError(
            "Need at least two bootstrap samples."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between zero and one."
        )

    alpha = (
        1.0 - confidence_level
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0 - alpha / 2.0
    )

    finite_counts = np.sum(
        np.isfinite(
            samples
        ),
        axis=0,
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        mean = np.nanmean(
            samples,
            axis=0,
        )

        standard_error = np.nanstd(
            samples,
            axis=0,
            ddof=1,
        )

        lower = np.nanquantile(
            samples,
            lower_quantile,
            axis=0,
        )

        median = np.nanquantile(
            samples,
            0.5,
            axis=0,
        )

        upper = np.nanquantile(
            samples,
            upper_quantile,
            axis=0,
        )

    mean = np.where(
        finite_counts > 0,
        mean,
        np.nan,
    )

    standard_error = np.where(
        finite_counts > 1,
        standard_error,
        np.nan,
    )

    lower = np.where(
        finite_counts > 0,
        lower,
        np.nan,
    )

    median = np.where(
        finite_counts > 0,
        median,
        np.nan,
    )

    upper = np.where(
        finite_counts > 0,
        upper,
        np.nan,
    )

    return BootstrapInterval(
        mean=np.asarray(
            mean,
            dtype=float,
        ),
        standard_error=np.asarray(
            standard_error,
            dtype=float,
        ),
        lower=np.asarray(
            lower,
            dtype=float,
        ),
        median=np.asarray(
            median,
            dtype=float,
        ),
        upper=np.asarray(
            upper,
            dtype=float,
        ),
        finite_counts=np.asarray(
            finite_counts,
            dtype=int,
        ),
        confidence_level=float(
            confidence_level
        ),
    )
