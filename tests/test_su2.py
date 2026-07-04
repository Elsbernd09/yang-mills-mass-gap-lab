import numpy as np

from ymlab.su2 import (
    identity,
    dagger,
    is_unitary,
    is_su2,
    determinant,
    random_su2,
    small_random_su2,
    from_quaternion,
)


def test_identity_is_su2():
    assert is_su2(identity())


def test_dagger_inverse_for_random_su2():
    rng = np.random.default_rng(42)
    u = random_su2(rng)

    assert np.allclose(dagger(u) @ u, identity())


def test_random_su2_properties():
    rng = np.random.default_rng(123)

    for _ in range(100):
        u = random_su2(rng)

        assert is_unitary(u)
        assert np.isclose(determinant(u), 1.0)


def test_small_random_su2_is_su2():
    rng = np.random.default_rng(7)
    u = small_random_su2(epsilon=0.01, rng=rng)

    assert is_su2(u)


def test_from_quaternion_normalizes_input():
    u = from_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))

    assert np.allclose(u, identity())
