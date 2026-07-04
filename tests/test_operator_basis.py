import numpy as np
import pytest

from ymlab.gauge_transformations import (
    gauge_transform_lattice,
    random_gauge_field,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.operator_basis import (
    CorrelatorMatrixResult,
    OperatorBasis,
    correlation_matrix_eigenvalues,
    create_smearing_basis,
    ensemble_correlator_matrix,
    maximum_matrix_asymmetry,
    measure_operator_basis,
    periodic_cross_correlation,
    symmetrize_correlator_matrix,
)


def nontrivial_lattice() -> Lattice:
    lattice = Lattice(
        shape=(4, 3, 3),
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


def test_create_smearing_basis():
    basis = create_smearing_basis(
        smearing_levels=[0, 2, 5],
        alpha=0.5,
        time_direction=0,
    )

    assert isinstance(
        basis,
        OperatorBasis,
    )

    assert basis.names == (
        "scalar_smear_0",
        "scalar_smear_2",
        "scalar_smear_5",
    )

    assert basis.smearing_levels == (
        0,
        2,
        5,
    )


def test_basis_rejects_duplicate_levels():
    with pytest.raises(ValueError):
        create_smearing_basis(
            smearing_levels=[0, 2, 2],
            alpha=0.5,
            time_direction=0,
        )


def test_measure_operator_basis_shape():
    lattice = nontrivial_lattice()

    basis = create_smearing_basis(
        smearing_levels=[0, 2, 5],
        alpha=0.5,
        time_direction=0,
    )

    measurements = measure_operator_basis(
        lattice=lattice,
        basis=basis,
    )

    assert measurements.shape == (
        3,
        4,
    )


def test_operator_basis_is_gauge_invariant():
    lattice = nontrivial_lattice()

    gauge_field = random_gauge_field(
        lattice=lattice,
        seed=314159,
    )

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    basis = create_smearing_basis(
        smearing_levels=[0, 2, 5],
        alpha=0.5,
        time_direction=0,
    )

    before = measure_operator_basis(
        lattice=lattice,
        basis=basis,
    )

    after = measure_operator_basis(
        lattice=transformed,
        basis=basis,
    )

    assert np.allclose(
        before,
        after,
        atol=1e-8,
        rtol=1e-8,
    )


def test_periodic_cross_correlation_shape():
    a = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    b = np.array(
        [2.0, 3.0, 4.0, 5.0]
    )

    result = periodic_cross_correlation(
        a,
        b,
    )

    assert result.shape == (4,)


def test_cross_correlation_reversal_identity():
    a = np.array(
        [1.0, 2.0, 4.0, 3.0]
    )

    b = np.array(
        [3.0, 1.0, 2.0, 5.0]
    )

    ab = periodic_cross_correlation(
        a,
        b,
    )

    ba = periodic_cross_correlation(
        b,
        a,
    )

    temporal_extent = len(a)

    for lag in range(
        temporal_extent
    ):
        partner = (
            temporal_extent - lag
        ) % temporal_extent

        assert np.isclose(
            ab[lag],
            ba[partner],
        )


def test_ensemble_correlator_matrix_shape():
    ensemble = np.array(
        [
            [
                [1.0, 2.0, 1.0, 2.0],
                [2.0, 3.0, 2.0, 3.0],
            ],
            [
                [1.5, 2.5, 1.5, 2.5],
                [2.5, 3.5, 2.5, 3.5],
            ],
            [
                [2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 3.0, 4.0],
            ],
        ]
    )

    result = ensemble_correlator_matrix(
        ensemble
    )

    assert isinstance(
        result,
        CorrelatorMatrixResult,
    )

    assert result.correlation_matrices.shape == (
        4,
        2,
        2,
    )

    assert result.operator_means.shape == (
        2,
    )


def test_zero_lag_correlation_matrix_is_symmetric():
    ensemble = np.array(
        [
            [
                [1.0, 2.0, 1.0, 2.0],
                [2.0, 3.0, 2.0, 3.0],
            ],
            [
                [1.5, 2.5, 1.5, 2.5],
                [2.5, 3.5, 2.5, 3.5],
            ],
            [
                [2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 3.0, 4.0],
            ],
        ]
    )

    result = ensemble_correlator_matrix(
        ensemble
    )

    assert np.allclose(
        result.correlation_matrices[0],
        result.correlation_matrices[0].T,
    )


def test_symmetrize_correlator_matrix():
    matrices = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )

    symmetric = symmetrize_correlator_matrix(
        matrices
    )

    assert np.allclose(
        symmetric[0],
        symmetric[0].T,
    )

    assert np.isclose(
        symmetric[0, 0, 1],
        2.5,
    )


def test_maximum_matrix_asymmetry():
    matrices = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )

    error = maximum_matrix_asymmetry(
        matrices
    )

    assert np.isclose(
        error,
        1.0,
    )


def test_correlation_matrix_eigenvalues():
    matrices = np.array(
        [
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [1.0, 0.0],
                [0.0, 0.5],
            ],
        ]
    )

    eigenvalues = (
        correlation_matrix_eigenvalues(
            matrices
        )
    )

    assert eigenvalues.shape == (
        2,
        2,
    )

    assert np.allclose(
        eigenvalues[0],
        [1.0, 2.0],
    )


def test_correlator_matrix_rejects_single_configuration():
    ensemble = np.zeros(
        (
            1,
            2,
            4,
        )
    )

    with pytest.raises(ValueError):
        ensemble_correlator_matrix(
            ensemble
        )
