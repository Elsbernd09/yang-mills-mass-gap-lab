import numpy as np
import pytest

from ymlab.gauge_transformations import (
    gauge_transform_lattice,
    random_gauge_field,
)
from ymlab.glueball import (
    scalar_glueball_time_series,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.smearing import (
    SmearingSummary,
    copy_lattice,
    maximum_su2_link_error,
    smear_spatial_link,
    smear_spatial_links,
    smear_with_summary,
    spatial_directions,
    spatial_side_path_sum,
    spatial_smearing_step,
)
from ymlab.su2 import (
    identity,
    is_su2,
)


def nontrivial_lattice() -> Lattice:
    lattice = Lattice(
        shape=(4, 4, 4),
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


def test_spatial_directions():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    assert spatial_directions(
        lattice,
        time_direction=0,
    ) == [1, 2]


def test_copy_lattice_is_independent():
    lattice = Lattice(
        shape=(3, 3, 3),
        cold_start=True,
    )

    copied = copy_lattice(
        lattice
    )

    copied.set_link(
        (0, 0, 0),
        1,
        -identity(),
    )

    assert not np.allclose(
        lattice.get_link(
            (0, 0, 0),
            1,
        ),
        copied.get_link(
            (0, 0, 0),
            1,
        ),
    )


def test_cold_spatial_side_path_sum_3d():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    result = spatial_side_path_sum(
        lattice=lattice,
        site=(0, 0, 0),
        mu=1,
        time_direction=0,
    )

    assert np.allclose(
        result,
        2.0 * identity(),
    )


def test_cold_smearing_preserves_identity_links():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=5,
    )

    for site in smeared.sites():
        for direction in range(
            smeared.dim
        ):
            assert np.allclose(
                smeared.get_link(
                    site,
                    direction,
                ),
                identity(),
            )


def test_zero_smearing_steps_returns_equal_copy():
    lattice = nontrivial_lattice()

    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=0,
    )

    assert smeared is not lattice

    for site in lattice.sites():
        for direction in range(
            lattice.dim
        ):
            assert np.allclose(
                lattice.get_link(
                    site,
                    direction,
                ),
                smeared.get_link(
                    site,
                    direction,
                ),
            )


def test_temporal_links_are_unchanged():
    lattice = nontrivial_lattice()

    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=4,
    )

    for site in lattice.sites():
        assert np.allclose(
            lattice.get_link(
                site,
                0,
            ),
            smeared.get_link(
                site,
                0,
            ),
        )


def test_smearing_preserves_su2():
    lattice = nontrivial_lattice()

    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=5,
    )

    for site in smeared.sites():
        for direction in range(
            smeared.dim
        ):
            assert is_su2(
                smeared.get_link(
                    site,
                    direction,
                ),
                atol=1e-8,
            )

    assert (
        maximum_su2_link_error(
            smeared
        )
        < 1e-8
    )


def test_smearing_step_is_simultaneous_and_nonmutating():
    lattice = nontrivial_lattice()

    before = {
        (
            site,
            direction,
        ): np.array(
            lattice.get_link(
                site,
                direction,
            ),
            copy=True,
        )
        for site in lattice.sites()
        for direction in range(
            lattice.dim
        )
    }

    spatial_smearing_step(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
    )

    for key, value in before.items():
        site, direction = key

        assert np.allclose(
            lattice.get_link(
                site,
                direction,
            ),
            value,
        )


def test_smeared_scalar_operator_is_gauge_invariant():
    lattice = nontrivial_lattice()

    gauge_field = random_gauge_field(
        lattice=lattice,
        seed=314159,
    )

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    smeared_before = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=3,
    )

    smeared_after = smear_spatial_links(
        lattice=transformed,
        time_direction=0,
        alpha=0.5,
        steps=3,
    )

    operator_before = (
        scalar_glueball_time_series(
            lattice=smeared_before,
            time_direction=0,
        )
    )

    operator_after = (
        scalar_glueball_time_series(
            lattice=smeared_after,
            time_direction=0,
        )
    )

    assert np.allclose(
        operator_before,
        operator_after,
        atol=1e-8,
        rtol=1e-8,
    )


def test_smear_with_summary():
    lattice = nontrivial_lattice()

    smeared, summary = smear_with_summary(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=2,
    )

    assert isinstance(
        summary,
        SmearingSummary,
    )

    assert summary.steps == 2
    assert np.isclose(
        summary.alpha,
        0.5,
    )
    assert (
        summary.maximum_link_membership_error
        < 1e-8
    )

    assert smeared.shape == lattice.shape


def test_smear_spatial_link_rejects_temporal_direction():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        smear_spatial_link(
            lattice=lattice,
            site=(0, 0, 0),
            mu=0,
            time_direction=0,
            alpha=0.5,
        )


def test_smearing_rejects_2d_lattice():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        spatial_smearing_step(
            lattice=lattice,
            time_direction=0,
            alpha=0.5,
        )


def test_smearing_rejects_invalid_alpha():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        smear_spatial_links(
            lattice=lattice,
            time_direction=0,
            alpha=1.5,
            steps=2,
        )
