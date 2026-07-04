import numpy as np
import pytest

from ymlab.gauge_transformations import (
    GaugeInvarianceComparison,
    compare_scalar_observable,
    gauge_transform_lattice,
    identity_gauge_field,
    max_link_membership_error,
    random_gauge_field,
    transform_link,
    validate_gauge_field,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.observables import (
    average_rectangular_wilson_loop,
    rectangular_wilson_loop,
)
from ymlab.plaquette import average_plaquette
from ymlab.su2 import is_su2
from ymlab.wilson_action import wilson_action


def thermalized_test_lattice() -> Lattice:
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
        seed=2026,
    )

    for _ in range(4):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.15,
        )

    return lattice


def test_random_gauge_field_is_valid():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    gauge_field = random_gauge_field(lattice, seed=123)

    assert validate_gauge_field(lattice, gauge_field)


def test_identity_gauge_field_is_valid():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    gauge_field = identity_gauge_field(lattice)

    assert validate_gauge_field(lattice, gauge_field)


def test_transform_link_preserves_su2():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = transform_link(
        lattice=lattice,
        gauge_field=gauge_field,
        site=(0, 0),
        direction=0,
    )

    assert is_su2(transformed, atol=1e-8)


def test_identity_gauge_transformation_preserves_every_link():
    lattice = thermalized_test_lattice()
    gauge_field = identity_gauge_field(lattice)

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    for site in lattice.sites():
        for direction in range(lattice.dim):
            assert np.allclose(
                lattice.get_link(site, direction),
                transformed.get_link(site, direction),
            )


def test_random_gauge_transformation_preserves_wilson_action():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = gauge_transform_lattice(lattice, gauge_field)

    before = wilson_action(lattice, beta=2.0)
    after = wilson_action(transformed, beta=2.0)

    assert np.isclose(before, after, atol=1e-8, rtol=1e-8)


def test_random_gauge_transformation_preserves_average_plaquette():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = gauge_transform_lattice(lattice, gauge_field)

    before = average_plaquette(lattice)
    after = average_plaquette(transformed)

    assert np.isclose(before, after, atol=1e-8, rtol=1e-8)


def test_random_gauge_transformation_preserves_single_wilson_loop():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = gauge_transform_lattice(lattice, gauge_field)

    before = rectangular_wilson_loop(
        lattice=lattice,
        site=(1, 2),
        mu=0,
        nu=1,
        width=2,
        height=3,
    )

    after = rectangular_wilson_loop(
        lattice=transformed,
        site=(1, 2),
        mu=0,
        nu=1,
        width=2,
        height=3,
    )

    assert np.isclose(before, after, atol=1e-8, rtol=1e-8)


def test_random_gauge_transformation_preserves_average_wilson_loop():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = gauge_transform_lattice(lattice, gauge_field)

    before = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=0,
        nu=1,
        width=2,
        height=2,
    )

    after = average_rectangular_wilson_loop(
        lattice=transformed,
        mu=0,
        nu=1,
        width=2,
        height=2,
    )

    assert np.isclose(before, after, atol=1e-8, rtol=1e-8)


def test_transformed_lattice_link_membership_error_is_small():
    lattice = thermalized_test_lattice()
    gauge_field = random_gauge_field(lattice, seed=123)

    transformed = gauge_transform_lattice(lattice, gauge_field)

    assert max_link_membership_error(transformed) < 1e-8


def test_compare_scalar_observable_returns_comparison():
    result = compare_scalar_observable(
        name="test",
        before=1.0,
        after=1.0 + 1e-12,
    )

    assert isinstance(result, GaugeInvarianceComparison)
    assert result.invariant


def test_invalid_gauge_field_is_rejected():
    lattice = Lattice(shape=(4, 4), cold_start=True)
    gauge_field = random_gauge_field(lattice, seed=123)

    gauge_field.pop((0, 0))

    assert not validate_gauge_field(lattice, gauge_field)

    with pytest.raises(ValueError):
        gauge_transform_lattice(lattice, gauge_field)
