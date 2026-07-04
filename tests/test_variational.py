import numpy as np
import pytest

from ymlab.variational import (
    GEVPResult,
    ReferenceMetric,
    build_reference_metric,
    metric_orthonormality_error,
    principal_arccosh_effective_masses,
    principal_log_effective_masses,
    solve_regularized_gevp,
    whitened_matrix,
)


def exact_mixed_correlator(
    energies,
    mixing_matrix,
    temporal_extent,
):
    energies = np.asarray(
        energies,
        dtype=float,
    )

    mixing_matrix = np.asarray(
        mixing_matrix,
        dtype=float,
    )

    matrices = []

    for lag in range(
        temporal_extent
    ):
        diagonal = np.diag(
            np.exp(
                -energies * lag
            )
        )

        matrices.append(
            mixing_matrix
            @ diagonal
            @ mixing_matrix.T
        )

    return np.asarray(
        matrices,
        dtype=float,
    )


def test_build_reference_metric_whitens_matrix():
    reference = np.array(
        [
            [4.0, 1.0],
            [1.0, 2.0],
        ]
    )

    metric = build_reference_metric(
        reference_matrix=reference,
        reference_time=1,
    )

    assert isinstance(
        metric,
        ReferenceMetric,
    )

    whitened = whitened_matrix(
        reference,
        metric,
    )

    assert np.allclose(
        whitened,
        np.eye(
            metric.retained_rank
        ),
        atol=1e-10,
        rtol=1e-10,
    )


def test_reference_metric_truncates_near_null_mode():
    reference = np.diag(
        [
            2.0,
            1.0,
            1e-16,
        ]
    )

    metric = build_reference_metric(
        reference_matrix=reference,
        reference_time=1,
        relative_cutoff=1e-10,
        absolute_cutoff=1e-14,
    )

    assert metric.retained_rank == 2


def test_reference_metric_rejects_nonpositive_matrix():
    reference = np.diag(
        [
            -1.0,
            -2.0,
        ]
    )

    with pytest.raises(ValueError):
        build_reference_metric(
            reference_matrix=reference,
            reference_time=1,
        )


def test_exact_gevp_recovers_principal_correlators():
    energies = np.array(
        [
            0.35,
            0.80,
        ]
    )

    mixing = np.array(
        [
            [1.0, 0.4],
            [0.3, 1.2],
        ]
    )

    matrices = exact_mixed_correlator(
        energies=energies,
        mixing_matrix=mixing,
        temporal_extent=8,
    )

    result = solve_regularized_gevp(
        correlation_matrices=matrices,
        reference_time=1,
        track_states=False,
    )

    assert isinstance(
        result,
        GEVPResult,
    )

    expected = np.column_stack(
        [
            np.exp(
                -energies[0]
                * (
                    result.lags - 1
                )
            ),
            np.exp(
                -energies[1]
                * (
                    result.lags - 1
                )
            ),
        ]
    )

    # Largest principal correlator corresponds to the smaller energy
    # for lags later than t0.
    assert np.allclose(
        result.principal_correlators[
            1:,
        ],
        expected[1:],
        atol=1e-9,
        rtol=1e-9,
    )


def test_principal_log_mass_recovers_exact_energies():
    energies = np.array(
        [
            0.4,
            0.9,
        ]
    )

    lags = np.arange(
        8,
        dtype=float,
    )

    principal = np.column_stack(
        [
            np.exp(
                -energies[0] * lags
            ),
            np.exp(
                -energies[1] * lags
            ),
        ]
    )

    effective = principal_log_effective_masses(
        principal
    )

    assert np.allclose(
        effective[:-1, 0],
        energies[0],
        atol=1e-12,
    )

    assert np.allclose(
        effective[:-1, 1],
        energies[1],
        atol=1e-12,
    )


def test_principal_arccosh_mass_recovers_periodic_mass():
    temporal_extent = 10
    masses = np.array(
        [
            0.45,
            0.9,
        ]
    )

    lags = np.arange(
        temporal_extent,
        dtype=float,
    )

    principal = np.column_stack(
        [
            np.exp(
                -mass * lags
            )
            + np.exp(
                -mass
                * (
                    temporal_extent - lags
                )
            )
            for mass in masses
        ]
    )

    effective = (
        principal_arccosh_effective_masses(
            principal
        )
    )

    for state, mass in enumerate(
        masses
    ):
        finite = effective[
            np.isfinite(
                effective[:, state]
            ),
            state,
        ]

        assert len(finite) > 0

        assert np.allclose(
            finite,
            mass,
            atol=1e-10,
            rtol=1e-10,
        )


def test_reference_generalized_vectors_are_metric_orthonormal():
    energies = np.array(
        [
            0.3,
            0.7,
        ]
    )

    mixing = np.array(
        [
            [1.0, 0.2],
            [0.4, 1.1],
        ]
    )

    matrices = exact_mixed_correlator(
        energies=energies,
        mixing_matrix=mixing,
        temporal_extent=6,
    )

    result = solve_regularized_gevp(
        correlation_matrices=matrices,
        reference_time=1,
    )

    error = metric_orthonormality_error(
        vectors=result.generalized_eigenvectors[
            result.reference_time
        ],
        reference_matrix=matrices[
            result.reference_time
        ],
    )

    assert error < 1e-9


def test_reference_principal_correlators_are_one():
    energies = np.array(
        [
            0.3,
            0.7,
        ]
    )

    mixing = np.array(
        [
            [1.0, 0.2],
            [0.4, 1.1],
        ]
    )

    matrices = exact_mixed_correlator(
        energies=energies,
        mixing_matrix=mixing,
        temporal_extent=6,
    )

    result = solve_regularized_gevp(
        correlation_matrices=matrices,
        reference_time=1,
    )

    assert np.allclose(
        result.principal_correlators[
            result.reference_time
        ],
        1.0,
        atol=1e-10,
        rtol=1e-10,
    )


def test_gevp_rejects_invalid_reference_time():
    matrices = np.zeros(
        (
            4,
            2,
            2,
        )
    )

    with pytest.raises(ValueError):
        solve_regularized_gevp(
            correlation_matrices=matrices,
            reference_time=5,
        )
