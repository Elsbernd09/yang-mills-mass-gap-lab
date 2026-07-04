"""
Regularized generalized-eigenvalue spectroscopy.

For an operator correlator matrix C(t), the variational method studies

    C(t) v_n(t, t0)
        =
        lambda_n(t, t0) C(t0) v_n(t, t0).

The reference matrix C(t0) can be poorly conditioned when the operator basis
contains strongly correlated or nearly linearly dependent operators.

This module therefore avoids direct inversion of C(t0).

The algorithm is:

1. Symmetrize C(t).
2. Diagonalize the reference matrix C(t0).
3. Retain only sufficiently positive metric eigenmodes.
4. Construct a whitening map W satisfying

       W^T C(t0) W = I

   in the retained subspace.

5. Solve the ordinary symmetric eigenproblem

       W^T C(t) W q_n = lambda_n q_n.

6. Map retained eigenvectors back to the original operator basis.
7. Optionally track states between adjacent Euclidean times by maximizing
   eigenvector overlap.

The resulting generalized eigenvalues are called principal correlators.

This is an exploratory finite-lattice variational spectroscopy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from ymlab.operator_basis import symmetrize_correlator_matrix


@dataclass(frozen=True)
class ReferenceMetric:
    """Regularized reference correlator metric."""

    reference_time: int
    original_dimension: int
    retained_rank: int
    eigenvalues: np.ndarray
    retained_eigenvalues: np.ndarray
    retained_mask: np.ndarray
    whitening_matrix: np.ndarray
    condition_number: float
    eigenvalue_cutoff: float


@dataclass(frozen=True)
class GEVPResult:
    """Generalized-eigenvalue spectroscopy result."""

    lags: np.ndarray
    principal_correlators: np.ndarray
    generalized_eigenvectors: np.ndarray
    whitened_eigenvectors: np.ndarray
    reference_metric: ReferenceMetric
    reference_time: int
    retained_rank: int
    maximum_reference_identity_error: float


def build_reference_metric(
    reference_matrix: np.ndarray,
    reference_time: int,
    relative_cutoff: float = 1e-10,
    absolute_cutoff: float = 1e-14,
) -> ReferenceMetric:
    """
    Construct a regularized whitening transform from C(t0).

    Only eigenvalues satisfying

        eigenvalue > max(
            absolute_cutoff,
            relative_cutoff * largest_positive_eigenvalue
        )

    are retained.
    """
    matrix = np.asarray(
        reference_matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "reference_matrix must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "reference_matrix must be square."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "reference_matrix must contain finite values."
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
        matrix + matrix.T
    )

    eigenvalues, eigenvectors = eigh(
        symmetric
    )

    positive = eigenvalues[
        eigenvalues > 0.0
    ]

    if len(positive) == 0:
        raise ValueError(
            "Reference correlator has no positive metric modes."
        )

    largest_positive = float(
        np.max(positive)
    )

    cutoff = max(
        float(absolute_cutoff),
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
            "Regularization removed every reference metric mode."
        )

    whitening_matrix = (
        retained_eigenvectors
        @ np.diag(
            1.0
            / np.sqrt(
                retained_eigenvalues
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

    return ReferenceMetric(
        reference_time=int(
            reference_time
        ),
        original_dimension=matrix.shape[0],
        retained_rank=len(
            retained_eigenvalues
        ),
        eigenvalues=np.asarray(
            eigenvalues,
            dtype=float,
        ),
        retained_eigenvalues=np.asarray(
            retained_eigenvalues,
            dtype=float,
        ),
        retained_mask=np.asarray(
            retained_mask,
            dtype=bool,
        ),
        whitening_matrix=np.asarray(
            whitening_matrix,
            dtype=float,
        ),
        condition_number=condition_number,
        eigenvalue_cutoff=float(
            cutoff
        ),
    )


def whitened_matrix(
    matrix: np.ndarray,
    metric: ReferenceMetric,
) -> np.ndarray:
    """
    Transform an operator-space matrix into the retained whitened basis.
    """
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    expected_shape = (
        metric.original_dimension,
        metric.original_dimension,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            "Matrix shape does not match reference metric."
        )

    symmetric = 0.5 * (
        matrix + matrix.T
    )

    w = metric.whitening_matrix

    transformed = (
        w.T
        @ symmetric
        @ w
    )

    return 0.5 * (
        transformed
        + transformed.T
    )


def _sort_descending(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs from largest to smallest eigenvalue."""
    order = np.argsort(
        eigenvalues
    )[::-1]

    return (
        eigenvalues[order],
        eigenvectors[:, order],
    )


def _match_states_by_overlap(
    previous_vectors: np.ndarray,
    current_vectors: np.ndarray,
    current_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match current states to previous states by maximum absolute overlap.

    The Hungarian assignment algorithm finds the one-to-one assignment that
    maximizes total absolute eigenvector overlap.

    Eigenvector signs are then aligned with the previous time slice.
    """
    overlap = np.abs(
        previous_vectors.T
        @ current_vectors
    )

    row_indices, column_indices = (
        linear_sum_assignment(
            -overlap
        )
    )

    permutation = np.empty(
        len(column_indices),
        dtype=int,
    )

    permutation[row_indices] = (
        column_indices
    )

    matched_vectors = current_vectors[
        :,
        permutation,
    ]

    matched_values = current_values[
        permutation
    ]

    for state in range(
        matched_vectors.shape[1]
    ):
        signed_overlap = float(
            previous_vectors[:, state]
            @ matched_vectors[:, state]
        )

        if signed_overlap < 0.0:
            matched_vectors[:, state] *= -1.0

    return (
        matched_values,
        matched_vectors,
    )


def solve_regularized_gevp(
    correlation_matrices: np.ndarray,
    reference_time: int = 1,
    relative_cutoff: float = 1e-10,
    absolute_cutoff: float = 1e-14,
    track_states: bool = True,
) -> GEVPResult:
    """
    Solve a regularized generalized eigenvalue problem for all time lags.

    Input shape:

        (temporal_extent, operators, operators).

    Principal correlator output shape:

        (temporal_extent, retained_rank).
    """
    matrices = np.asarray(
        correlation_matrices,
        dtype=float,
    )

    if matrices.ndim != 3:
        raise ValueError(
            "Expected correlation matrix shape "
            "(lags, operators, operators)."
        )

    if (
        matrices.shape[1]
        != matrices.shape[2]
    ):
        raise ValueError(
            "Operator correlation matrices must be square."
        )

    if not np.all(
        np.isfinite(matrices)
    ):
        raise ValueError(
            "Correlation matrices must contain finite values."
        )

    temporal_extent = matrices.shape[0]

    if (
        reference_time < 0
        or reference_time >= temporal_extent
    ):
        raise ValueError(
            "Invalid reference_time."
        )

    symmetric = symmetrize_correlator_matrix(
        matrices
    )

    metric = build_reference_metric(
        reference_matrix=symmetric[
            reference_time
        ],
        reference_time=reference_time,
        relative_cutoff=relative_cutoff,
        absolute_cutoff=absolute_cutoff,
    )

    retained_rank = metric.retained_rank

    principal = np.zeros(
        (
            temporal_extent,
            retained_rank,
        ),
        dtype=float,
    )

    whitened_vectors = np.zeros(
        (
            temporal_extent,
            retained_rank,
            retained_rank,
        ),
        dtype=float,
    )

    generalized_vectors = np.zeros(
        (
            temporal_extent,
            metric.original_dimension,
            retained_rank,
        ),
        dtype=float,
    )

    previous_vectors = None

    for lag in range(
        temporal_extent
    ):
        reduced_matrix = whitened_matrix(
            symmetric[lag],
            metric,
        )

        values, vectors = eigh(
            reduced_matrix
        )

        values, vectors = _sort_descending(
            values,
            vectors,
        )

        if (
            track_states
            and previous_vectors is not None
        ):
            values, vectors = (
                _match_states_by_overlap(
                    previous_vectors=previous_vectors,
                    current_vectors=vectors,
                    current_values=values,
                )
            )

        principal[lag] = values
        whitened_vectors[lag] = vectors

        generalized_vectors[lag] = (
            metric.whitening_matrix
            @ vectors
        )

        previous_vectors = vectors.copy()

    reference_identity = whitened_matrix(
        symmetric[reference_time],
        metric,
    )

    maximum_reference_identity_error = float(
        np.max(
            np.abs(
                reference_identity
                - np.eye(
                    retained_rank,
                    dtype=float,
                )
            )
        )
    )

    return GEVPResult(
        lags=np.arange(
            temporal_extent,
            dtype=int,
        ),
        principal_correlators=principal,
        generalized_eigenvectors=generalized_vectors,
        whitened_eigenvectors=whitened_vectors,
        reference_metric=metric,
        reference_time=reference_time,
        retained_rank=retained_rank,
        maximum_reference_identity_error=(
            maximum_reference_identity_error
        ),
    )


def principal_log_effective_masses(
    principal_correlators: np.ndarray,
) -> np.ndarray:
    """
    Compute log-ratio effective masses for each principal correlator.

    For positive adjacent principal correlator values,

        m_eff(t)
            =
            log[
                lambda(t) / lambda(t + 1)
            ].

    Invalid values are returned as NaN.
    """
    principal = np.asarray(
        principal_correlators,
        dtype=float,
    )

    if principal.ndim != 2:
        raise ValueError(
            "principal_correlators must be two-dimensional."
        )

    if principal.shape[0] < 2:
        raise ValueError(
            "Need at least two Euclidean time lags."
        )

    effective = np.full(
        principal.shape,
        np.nan,
        dtype=float,
    )

    for lag in range(
        principal.shape[0] - 1
    ):
        current = principal[lag]
        following = principal[
            lag + 1
        ]

        valid = (
            np.isfinite(current)
            & np.isfinite(following)
            & (current > 0.0)
            & (following > 0.0)
        )

        effective[
            lag,
            valid,
        ] = np.log(
            current[valid]
            / following[valid]
        )

    return effective


def principal_arccosh_effective_masses(
    principal_correlators: np.ndarray,
) -> np.ndarray:
    """
    Compute an arccosh effective mass independently for each principal state.

    Invalid time-state points are returned as NaN.
    """
    principal = np.asarray(
        principal_correlators,
        dtype=float,
    )

    if principal.ndim != 2:
        raise ValueError(
            "principal_correlators must be two-dimensional."
        )

    if principal.shape[0] < 3:
        raise ValueError(
            "Need at least three Euclidean time lags."
        )

    effective = np.full(
        principal.shape,
        np.nan,
        dtype=float,
    )

    for state in range(
        principal.shape[1]
    ):
        series = principal[
            :,
            state,
        ]

        for lag in range(
            1,
            len(series) - 1,
        ):
            denominator = (
                2.0 * series[lag]
            )

            if (
                not np.isfinite(denominator)
                or np.isclose(
                    denominator,
                    0.0,
                )
            ):
                continue

            ratio = (
                series[lag - 1]
                + series[lag + 1]
            ) / denominator

            if (
                not np.isfinite(ratio)
                or ratio < 1.0
            ):
                continue

            effective[
                lag,
                state,
            ] = float(
                np.arccosh(
                    ratio
                )
            )

    return effective


def metric_orthonormality_error(
    vectors: np.ndarray,
    reference_matrix: np.ndarray,
) -> float:
    """
    Measure max error in V^T C(t0) V = I.
    """
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    reference_matrix = np.asarray(
        reference_matrix,
        dtype=float,
    )

    gram = (
        vectors.T
        @ reference_matrix
        @ vectors
    )

    identity = np.eye(
        vectors.shape[1],
        dtype=float,
    )

    return float(
        np.max(
            np.abs(
                gram - identity
            )
        )
    )
