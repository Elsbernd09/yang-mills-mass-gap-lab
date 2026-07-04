import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.staples import staple, local_link_action, local_action_difference
from ymlab.su2 import identity, small_random_su2, is_su2


def test_cold_start_staple_shape():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    st = staple(lattice, site=(0, 0), mu=0)

    assert st.shape == (2, 2)


def test_cold_start_2d_staple_equals_two_identities():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    st = staple(lattice, site=(0, 0), mu=0)

    assert np.allclose(st, 2.0 * identity())


def test_cold_start_3d_staple_equals_four_identities():
    lattice = Lattice(shape=(4, 4, 4), cold_start=True)

    st = staple(lattice, site=(0, 0, 0), mu=0)

    assert np.allclose(st, 4.0 * identity())


def test_local_link_action_is_finite():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    value = local_link_action(lattice, site=(0, 0), mu=0, beta=2.0)

    assert np.isfinite(value)


def test_local_action_difference_identity_proposal_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    delta = local_action_difference(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        proposal=identity(),
        beta=2.0,
    )

    assert np.isclose(delta, 0.0)


def test_local_action_difference_for_su2_proposal_is_finite():
    lattice = Lattice(shape=(4, 4), cold_start=True, seed=123)
    proposal = small_random_su2(epsilon=0.1, rng=lattice.rng)

    assert is_su2(proposal)

    delta = local_action_difference(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        proposal=proposal,
        beta=2.0,
    )

    assert np.isfinite(delta)


def test_invalid_direction_raises_error():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    with pytest.raises(ValueError):
        staple(lattice, site=(0, 0), mu=4)


def test_negative_beta_raises_error():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    with pytest.raises(ValueError):
        local_link_action(lattice, site=(0, 0), mu=0, beta=-1.0)
