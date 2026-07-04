import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.wilson_action import (
    wilson_action,
    action_per_plaquette,
    number_of_plaquettes,
)


def test_number_of_plaquettes_2d():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert number_of_plaquettes(lattice) == 16


def test_number_of_plaquettes_3d():
    lattice = Lattice(shape=(4, 4, 4), cold_start=True)

    # In 3D, each site has C(3, 2) = 3 plaquette planes.
    assert number_of_plaquettes(lattice) == 64 * 3


def test_cold_start_wilson_action_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert np.isclose(wilson_action(lattice, beta=2.0), 0.0)


def test_cold_start_action_per_plaquette_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert np.isclose(action_per_plaquette(lattice, beta=2.0), 0.0)


def test_negative_beta_raises_error():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    with pytest.raises(ValueError):
        wilson_action(lattice, beta=-1.0)


def test_random_start_action_is_nonnegative():
    lattice = Lattice(shape=(4, 4), cold_start=False, seed=123)

    assert wilson_action(lattice, beta=2.0) >= 0.0
