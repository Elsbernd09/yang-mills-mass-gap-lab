import numpy as np

from ymlab.lattice import Lattice
from ymlab.staples import (
    local_action_difference,
    staple,
)
from ymlab.su2 import (
    dagger,
    identity,
    small_random_su2,
)
from ymlab.validation import (
    ActionDifferenceComparison,
    compare_action_differences,
    global_action_difference,
)


def test_cold_2d_staple_equals_two_identity():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
    )

    result = staple(
        lattice=lattice,
        site=(0, 0),
        mu=0,
    )

    assert np.allclose(
        result,
        2.0 * identity(),
    )


def test_cold_3d_staple_equals_four_identity():
    lattice = Lattice(
        shape=(3, 3, 3),
        cold_start=True,
    )

    result = staple(
        lattice=lattice,
        site=(0, 0, 0),
        mu=0,
    )

    assert np.allclose(
        result,
        4.0 * identity(),
    )


def test_cold_4d_staple_equals_six_identity():
    lattice = Lattice(
        shape=(2, 2, 2, 2),
        cold_start=True,
    )

    result = staple(
        lattice=lattice,
        site=(0, 0, 0, 0),
        mu=0,
    )

    assert np.allclose(
        result,
        6.0 * identity(),
    )


def test_identity_proposal_has_zero_local_difference():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
    )

    delta = local_action_difference(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        proposal=identity(),
        beta=2.0,
    )

    assert np.isclose(
        delta,
        0.0,
    )


def test_global_action_difference_restores_original_link():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=False,
        seed=2026,
    )

    site = (1, 2)
    mu = 0

    original = np.array(
        lattice.get_link(site, mu),
        copy=True,
    )

    rng = np.random.default_rng(123)

    proposal = (
        small_random_su2(
            epsilon=0.1,
            rng=rng,
        )
        @ original
    )

    global_action_difference(
        lattice=lattice,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=2.0,
    )

    assert np.allclose(
        lattice.get_link(site, mu),
        original,
    )


def test_local_global_delta_agree_on_cold_lattice():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
    )

    rng = np.random.default_rng(123)

    proposal = small_random_su2(
        epsilon=0.15,
        rng=rng,
    )

    comparison = compare_action_differences(
        lattice=lattice,
        site=(1, 2),
        mu=0,
        proposal=proposal,
        beta=2.0,
    )

    assert isinstance(
        comparison,
        ActionDifferenceComparison,
    )

    assert comparison.consistent

    assert comparison.absolute_error < 1e-10


def test_local_global_delta_agree_random_2d():
    lattice = Lattice(
        shape=(4, 4),
        cold_start=False,
        seed=2026,
    )

    rng = np.random.default_rng(314159)

    sites = list(
        lattice.sites()
    )

    for _ in range(20):
        site = sites[
            int(
                rng.integers(
                    0,
                    len(sites),
                )
            )
        ]

        mu = int(
            rng.integers(
                0,
                lattice.dim,
            )
        )

        old_link = lattice.get_link(
            site,
            mu,
        )

        proposal = (
            small_random_su2(
                epsilon=0.2,
                rng=rng,
            )
            @ old_link
        )

        comparison = compare_action_differences(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=1.7,
        )

        assert comparison.consistent
        assert comparison.absolute_error < 1e-9


def test_local_global_delta_agree_random_3d():
    lattice = Lattice(
        shape=(3, 3, 3),
        cold_start=False,
        seed=2027,
    )

    rng = np.random.default_rng(271828)

    sites = list(
        lattice.sites()
    )

    for _ in range(15):
        site = sites[
            int(
                rng.integers(
                    0,
                    len(sites),
                )
            )
        ]

        mu = int(
            rng.integers(
                0,
                lattice.dim,
            )
        )

        old_link = lattice.get_link(
            site,
            mu,
        )

        proposal = (
            small_random_su2(
                epsilon=0.15,
                rng=rng,
            )
            @ old_link
        )

        comparison = compare_action_differences(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=2.3,
        )

        assert comparison.consistent
        assert comparison.absolute_error < 1e-9


def test_local_global_delta_agree_random_4d():
    lattice = Lattice(
        shape=(2, 2, 2, 2),
        cold_start=False,
        seed=2028,
    )

    rng = np.random.default_rng(161803)

    sites = list(
        lattice.sites()
    )

    for _ in range(12):
        site = sites[
            int(
                rng.integers(
                    0,
                    len(sites),
                )
            )
        ]

        mu = int(
            rng.integers(
                0,
                lattice.dim,
            )
        )

        old_link = lattice.get_link(
            site,
            mu,
        )

        proposal = (
            small_random_su2(
                epsilon=0.1,
                rng=rng,
            )
            @ old_link
        )

        comparison = compare_action_differences(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=2.0,
        )

        assert comparison.consistent
        assert comparison.absolute_error < 1e-9


def test_staple_dagger_sanity_is_finite():
    lattice = Lattice(
        shape=(3, 3, 3),
        cold_start=False,
        seed=2026,
    )

    value = staple(
        lattice=lattice,
        site=(1, 1, 1),
        mu=2,
    )

    assert np.all(
        np.isfinite(value)
    )

    assert np.all(
        np.isfinite(
            dagger(value)
        )
    )
