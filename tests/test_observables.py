import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.observables import (
    rectangular_wilson_loop,
    average_rectangular_wilson_loop,
    wilson_loop_table,
)


def test_cold_start_1x1_wilson_loop_is_one():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    value = rectangular_wilson_loop(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        nu=1,
        width=1,
        height=1,
    )

    assert np.isclose(value, 1.0)


def test_cold_start_larger_wilson_loop_is_one():
    lattice = Lattice(shape=(6, 6), cold_start=True)

    value = rectangular_wilson_loop(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        nu=1,
        width=3,
        height=2,
    )

    assert np.isclose(value, 1.0)


def test_average_cold_start_wilson_loop_is_one():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    value = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=0,
        nu=1,
        width=2,
        height=2,
    )

    assert np.isclose(value, 1.0)


def test_wilson_loop_table_size():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    table = wilson_loop_table(
        lattice=lattice,
        mu=0,
        nu=1,
        max_width=3,
        max_height=2,
    )

    assert len(table) == 6


def test_wilson_loop_table_contains_expected_keys():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    table = wilson_loop_table(
        lattice=lattice,
        mu=0,
        nu=1,
        max_width=2,
        max_height=2,
    )

    row = table[0]

    assert "width" in row
    assert "height" in row
    assert "area" in row
    assert "perimeter" in row
    assert "value" in row


def test_invalid_wilson_loop_directions_raise_error():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    with pytest.raises(ValueError):
        rectangular_wilson_loop(
            lattice=lattice,
            site=(0, 0),
            mu=0,
            nu=0,
            width=1,
            height=1,
        )


def test_invalid_wilson_loop_size_raises_error():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    with pytest.raises(ValueError):
        rectangular_wilson_loop(
            lattice=lattice,
            site=(0, 0),
            mu=0,
            nu=1,
            width=0,
            height=1,
        )
