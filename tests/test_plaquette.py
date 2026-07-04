import numpy as np

from ymlab.lattice import Lattice
from ymlab.plaquette import (
    plaquette,
    plaquette_action_density,
    average_plaquette,
    average_plaquette_action_density,
)
from ymlab.su2 import identity, is_su2


def test_cold_start_plaquette_is_identity():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    up = plaquette(lattice, site=(0, 0), mu=0, nu=1)

    assert np.allclose(up, identity())


def test_cold_start_plaquette_is_su2():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    up = plaquette(lattice, site=(0, 0), mu=0, nu=1)

    assert is_su2(up)


def test_cold_start_action_density_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    density = plaquette_action_density(lattice, site=(0, 0), mu=0, nu=1)

    assert np.isclose(density, 0.0)


def test_cold_start_average_plaquette_is_one():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert np.isclose(average_plaquette(lattice), 1.0)


def test_cold_start_average_action_density_is_zero():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert np.isclose(average_plaquette_action_density(lattice), 0.0)
