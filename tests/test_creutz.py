import numpy as np
import pytest

from ymlab.creutz import (
    CreutzRatio,
    creutz_ratio_from_values,
    creutz_ratio,
    creutz_ratio_table,
    valid_creutz_values,
    estimate_string_tension,
    creutz_results_as_rows,
)
from ymlab.lattice import Lattice


def test_creutz_ratio_from_exact_area_law():
    sigma = 0.25

    def w(r, t):
        return np.exp(-sigma * r * t)

    result = creutz_ratio_from_values(
        w_rt=w(2, 2),
        w_r1_t1=w(1, 1),
        w_rt1=w(2, 1),
        w_r1_t=w(1, 2),
        width=2,
        height=2,
    )

    assert result.valid
    assert np.isclose(result.value, sigma)


def test_creutz_ratio_rejects_small_loop():
    result = creutz_ratio_from_values(
        w_rt=1.0,
        w_r1_t1=1.0,
        w_rt1=1.0,
        w_r1_t=1.0,
        width=1,
        height=2,
    )

    assert not result.valid


def test_creutz_ratio_rejects_nonpositive_values():
    result = creutz_ratio_from_values(
        w_rt=-1.0,
        w_r1_t1=1.0,
        w_rt1=1.0,
        w_r1_t=1.0,
        width=2,
        height=2,
    )

    assert not result.valid


def test_cold_start_creutz_ratio_is_zero():
    lattice = Lattice(shape=(6, 6), cold_start=True)

    result = creutz_ratio(
        lattice=lattice,
        mu=0,
        nu=1,
        width=2,
        height=2,
    )

    assert result.valid
    assert np.isclose(result.value, 0.0)


def test_creutz_ratio_table_size():
    lattice = Lattice(shape=(6, 6), cold_start=True)

    results = creutz_ratio_table(
        lattice=lattice,
        mu=0,
        nu=1,
        max_width=4,
        max_height=3,
    )

    assert len(results) == 6


def test_valid_creutz_values_extracts_only_valid():
    results = [
        CreutzRatio(width=2, height=2, value=0.1, valid=True),
        CreutzRatio(width=2, height=3, value=float("nan"), valid=False),
        CreutzRatio(width=3, height=3, value=0.2, valid=True),
    ]

    values = valid_creutz_values(results)

    assert np.allclose(values, np.array([0.1, 0.2]))


def test_estimate_string_tension_mean():
    results = [
        CreutzRatio(width=2, height=2, value=0.1, valid=True),
        CreutzRatio(width=3, height=3, value=0.3, valid=True),
    ]

    assert np.isclose(estimate_string_tension(results), 0.2)


def test_estimate_string_tension_rejects_no_valid_values():
    results = [
        CreutzRatio(width=2, height=2, value=float("nan"), valid=False),
    ]

    with pytest.raises(ValueError):
        estimate_string_tension(results)


def test_creutz_results_as_rows():
    results = [
        CreutzRatio(width=2, height=3, value=0.1, valid=True),
    ]

    rows = creutz_results_as_rows(results)

    assert rows[0]["width"] == 2
    assert rows[0]["height"] == 3
    assert rows[0]["area"] == 6
    assert rows[0]["valid"] is True
