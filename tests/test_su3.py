import numpy as np

from ymlab.su3 import (
    identity,
    dagger,
    determinant,
    is_unitary,
    is_su3,
    project_to_su3,
    random_su3,
    small_random_su3,
    reunitarize,
    real_trace,
    gell_mann_matrices,
    is_hermitian,
    is_traceless,
)


def test_identity_is_su3():
    assert is_su3(identity())


def test_dagger_inverse_for_identity():
    assert np.allclose(dagger(identity()) @ identity(), identity())


def test_random_su3_properties():
    rng = np.random.default_rng(123)

    for _ in range(20):
        u = random_su3(rng)

        assert is_unitary(u)
        assert np.isclose(determinant(u), 1.0)
        assert is_su3(u)


def test_project_to_su3_properties():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    u = project_to_su3(z)

    assert is_su3(u)


def test_small_random_su3_is_su3():
    rng = np.random.default_rng(7)

    u = small_random_su3(epsilon=0.01, rng=rng)

    assert is_su3(u)


def test_reunitarize_returns_su3():
    rng = np.random.default_rng(123)
    u = random_su3(rng)
    noisy = u + 1e-5 * rng.normal(size=(3, 3))

    projected = reunitarize(noisy)

    assert is_su3(projected, atol=1e-8)


def test_real_trace_identity():
    assert np.isclose(real_trace(identity()), 3.0)


def test_gell_mann_count():
    matrices = gell_mann_matrices()

    assert len(matrices) == 8


def test_gell_mann_matrices_are_hermitian_and_traceless():
    matrices = gell_mann_matrices()

    for matrix in matrices:
        assert matrix.shape == (3, 3)
        assert is_hermitian(matrix)
        assert is_traceless(matrix)


def test_gell_mann_normalization_trace():
    matrices = gell_mann_matrices()

    for i, a in enumerate(matrices):
        for j, b in enumerate(matrices):
            trace_value = np.trace(a @ b)

            if i == j:
                assert np.isclose(trace_value, 2.0)
            else:
                assert np.isclose(trace_value, 0.0)
