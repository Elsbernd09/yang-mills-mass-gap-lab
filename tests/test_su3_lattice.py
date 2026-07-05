import numpy as np

from ymlab.gauge_lattice import GaugeLattice
from ymlab.generic_gauge import (
    generic_average_plaquette,
    generic_metropolis_sweep,
    generic_rectangular_wilson_loop,
    generic_wilson_action,
)
from ymlab.generic_gauge_transformations import (
    gauge_transform_generic_lattice,
    identity_generic_gauge_field,
    maximum_generic_link_membership_error,
    random_generic_gauge_field,
)
from ymlab.generic_validation import (
    generic_global_action_difference,
    compare_generic_action_differences,
)
from ymlab.group_interface import su3_group


def nontrivial_su3_lattice(
    sweeps=5,
    seed=2026,
):
    lattice = GaugeLattice(
        shape=(3, 3, 3),
        group=su3_group(),
        cold_start=True,
        seed=seed,
    )

    for _ in range(
        sweeps
    ):
        generic_metropolis_sweep(
            lattice=lattice,
            beta=5.5,
            epsilon=0.05,
        )

    return lattice


def test_su3_local_global_action_difference_agrees():
    lattice = nontrivial_su3_lattice(
        sweeps=5,
        seed=2026,
    )

    rng = np.random.default_rng(
        314159
    )

    cases = [
        ((0, 0, 0), 0),
        ((1, 2, 0), 1),
        ((2, 1, 2), 2),
    ]

    for site, mu in cases:
        proposal = (
            lattice.group.small_random(
                0.05,
                rng,
            )
            @ lattice.get_link(
                site,
                mu,
            )
        )

        comparison = (
            compare_generic_action_differences(
                lattice=lattice,
                site=site,
                mu=mu,
                proposal=proposal,
                beta=5.5,
                atol=1e-9,
                rtol=1e-9,
            )
        )

        assert comparison.consistent

        assert (
            comparison.absolute_error
            < 1e-8
        )


def test_su3_global_delta_restores_original_link():
    lattice = nontrivial_su3_lattice(
        sweeps=4,
        seed=2027,
    )

    site = (
        1,
        1,
        2,
    )

    mu = 2

    old_link = np.array(
        lattice.get_link(
            site,
            mu,
        ),
        copy=True,
    )

    proposal = (
        lattice.group.small_random(
            0.04,
            np.random.default_rng(
                99
            ),
        )
        @ old_link
    )

    generic_global_action_difference(
        lattice=lattice,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=5.5,
    )

    assert np.allclose(
        lattice.get_link(
            site,
            mu,
        ),
        old_link,
        atol=1e-12,
        rtol=1e-12,
    )


def test_su3_identity_gauge_transform_is_unchanged():
    lattice = nontrivial_su3_lattice(
        sweeps=3,
        seed=2028,
    )

    field = identity_generic_gauge_field(
        lattice
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=field,
        )
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            assert np.allclose(
                transformed.get_link(
                    site,
                    mu,
                ),
                lattice.get_link(
                    site,
                    mu,
                ),
                atol=1e-12,
                rtol=1e-12,
            )


def test_su3_wilson_action_is_gauge_invariant():
    lattice = nontrivial_su3_lattice(
        sweeps=5,
        seed=2029,
    )

    field = random_generic_gauge_field(
        lattice=lattice,
        rng=np.random.default_rng(
            77
        ),
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=field,
        )
    )

    assert np.isclose(
        generic_wilson_action(
            lattice,
            beta=5.5,
        ),
        generic_wilson_action(
            transformed,
            beta=5.5,
        ),
        atol=1e-9,
        rtol=1e-10,
    )


def test_su3_average_plaquette_is_gauge_invariant():
    lattice = nontrivial_su3_lattice(
        sweeps=5,
        seed=2030,
    )

    field = random_generic_gauge_field(
        lattice=lattice,
        rng=np.random.default_rng(
            88
        ),
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=field,
        )
    )

    assert np.isclose(
        generic_average_plaquette(
            lattice
        ),
        generic_average_plaquette(
            transformed
        ),
        atol=1e-10,
        rtol=1e-10,
    )


def test_su3_closed_wilson_loop_is_gauge_invariant():
    lattice = nontrivial_su3_lattice(
        sweeps=5,
        seed=2031,
    )

    field = random_generic_gauge_field(
        lattice=lattice,
        rng=np.random.default_rng(
            101
        ),
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=field,
        )
    )

    original_loop = (
        generic_rectangular_wilson_loop(
            lattice=lattice,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    transformed_loop = (
        generic_rectangular_wilson_loop(
            lattice=transformed,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    assert np.isclose(
        original_loop,
        transformed_loop,
        atol=1e-10,
        rtol=1e-10,
    )


def test_su3_transformed_links_remain_in_group():
    lattice = nontrivial_su3_lattice(
        sweeps=4,
        seed=2032,
    )

    field = random_generic_gauge_field(
        lattice=lattice,
        rng=np.random.default_rng(
            2026
        ),
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=field,
        )
    )

    for site in transformed.sites():
        for mu in range(
            transformed.dim
        ):
            assert transformed.group.is_member(
                transformed.get_link(
                    site,
                    mu,
                )
            )

    assert (
        maximum_generic_link_membership_error(
            transformed
        )
        < 1e-8
    )


def test_short_su3_chain_preserves_link_membership():
    lattice = GaugeLattice(
        shape=(3, 3, 3),
        group=su3_group(),
        cold_start=True,
        seed=2033,
    )

    acceptance_rates = []

    for _ in range(
        10
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=5.5,
            epsilon=0.05,
        )

        acceptance_rates.append(
            result.acceptance_rate
        )

    assert np.all(
        np.asarray(
            acceptance_rates
        ) >= 0.0
    )

    assert np.all(
        np.asarray(
            acceptance_rates
        ) <= 1.0
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            assert lattice.group.is_member(
                lattice.get_link(
                    site,
                    mu,
                )
            )
