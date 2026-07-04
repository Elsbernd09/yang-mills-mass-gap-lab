import numpy as np
import pytest

from ymlab.gauge_lattice import GaugeLattice
from ymlab.generic_gauge import (
    GenericMetropolisSweepResult,
    generic_average_plaquette,
    generic_local_action_difference,
    generic_metropolis_sweep,
    generic_number_of_plaquettes,
    generic_plaquette,
    generic_rectangular_wilson_loop,
    generic_staple,
    generic_wilson_action,
)
from ymlab.group_interface import (
    su2_group,
    su3_group,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.observables import rectangular_wilson_loop
from ymlab.plaquette import (
    average_plaquette,
    plaquette,
)
from ymlab.staples import (
    local_action_difference,
    staple,
)
from ymlab.su2 import small_random_su2
from ymlab.wilson_action import (
    number_of_plaquettes,
    wilson_action,
)


def copy_su2_configuration_to_generic(
    lattice: Lattice,
) -> GaugeLattice:
    generic = GaugeLattice(
        shape=lattice.shape,
        group=su2_group(),
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            generic.set_link(
                site,
                mu,
                lattice.get_link(
                    site,
                    mu,
                ),
            )

    return generic


def nontrivial_su2_pair():
    lattice = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
        seed=2026,
    )

    for _ in range(4):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.18,
        )

    generic = (
        copy_su2_configuration_to_generic(
            lattice
        )
    )

    return lattice, generic


def test_generic_su2_cold_start():
    lattice = GaugeLattice(
        shape=(4, 4),
        group=su2_group(),
        cold_start=True,
    )

    assert lattice.number_of_sites() == 16
    assert lattice.number_of_links() == 32

    assert np.isclose(
        generic_average_plaquette(
            lattice
        ),
        1.0,
    )

    assert np.isclose(
        generic_wilson_action(
            lattice,
            beta=2.0,
        ),
        0.0,
    )


def test_generic_su3_cold_start():
    lattice = GaugeLattice(
        shape=(3, 3, 3),
        group=su3_group(),
        cold_start=True,
    )

    assert np.isclose(
        generic_average_plaquette(
            lattice
        ),
        1.0,
    )

    assert np.isclose(
        generic_wilson_action(
            lattice,
            beta=5.5,
        ),
        0.0,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            assert lattice.get_link(
                site,
                mu,
            ).shape == (
                3,
                3,
            )


def test_generic_plaquette_count_matches_old_su2():
    old, generic = nontrivial_su2_pair()

    assert (
        generic_number_of_plaquettes(
            generic
        )
        == number_of_plaquettes(
            old
        )
    )


def test_generic_su2_plaquette_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    for site, mu, nu in [
        ((0, 0, 0), 0, 1),
        ((1, 2, 3), 1, 2),
        ((3, 1, 2), 0, 2),
    ]:
        assert np.allclose(
            generic_plaquette(
                lattice=generic,
                site=site,
                mu=mu,
                nu=nu,
            ),
            plaquette(
                lattice=old,
                site=site,
                mu=mu,
                nu=nu,
            ),
            atol=1e-12,
            rtol=1e-12,
        )


def test_generic_su2_average_plaquette_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    assert np.isclose(
        generic_average_plaquette(
            generic
        ),
        average_plaquette(
            old
        ),
        atol=1e-12,
        rtol=1e-12,
    )


def test_generic_su2_wilson_action_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    assert np.isclose(
        generic_wilson_action(
            generic,
            beta=2.0,
        ),
        wilson_action(
            old,
            beta=2.0,
        ),
        atol=1e-10,
        rtol=1e-12,
    )


def test_generic_su2_staple_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    for site, mu in [
        ((0, 0, 0), 0),
        ((1, 2, 3), 1),
        ((3, 1, 2), 2),
    ]:
        assert np.allclose(
            generic_staple(
                lattice=generic,
                site=site,
                mu=mu,
            ),
            staple(
                lattice=old,
                site=site,
                mu=mu,
            ),
            atol=1e-12,
            rtol=1e-12,
        )


def test_generic_su2_local_delta_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    rng = np.random.default_rng(
        314159
    )

    site = (
        1,
        2,
        3,
    )

    mu = 1

    proposal = (
        small_random_su2(
            epsilon=0.15,
            rng=rng,
        )
        @ old.get_link(
            site,
            mu,
        )
    )

    old_delta = local_action_difference(
        lattice=old,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=2.0,
    )

    generic_delta = (
        generic_local_action_difference(
            lattice=generic,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=2.0,
        )
    )

    assert np.isclose(
        generic_delta,
        old_delta,
        atol=1e-10,
        rtol=1e-10,
    )


def test_generic_su2_wilson_loop_matches_old_backend():
    old, generic = nontrivial_su2_pair()

    old_value = rectangular_wilson_loop(
        lattice=old,
        site=(0, 1, 2),
        mu=0,
        nu=2,
        width=2,
        height=2,
    )

    generic_value = (
        generic_rectangular_wilson_loop(
            lattice=generic,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    assert np.isclose(
        generic_value,
        old_value,
        atol=1e-12,
        rtol=1e-12,
    )


def test_generic_su2_metropolis_sweep_preserves_group():
    lattice = GaugeLattice(
        shape=(4, 4, 4),
        group=su2_group(),
        cold_start=True,
        seed=2026,
    )

    result = generic_metropolis_sweep(
        lattice=lattice,
        beta=2.0,
        epsilon=0.18,
    )

    assert isinstance(
        result,
        GenericMetropolisSweepResult,
    )

    assert (
        0.0
        <= result.acceptance_rate
        <= 1.0
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


def test_generic_su3_metropolis_sweep_preserves_group():
    lattice = GaugeLattice(
        shape=(3, 3, 3),
        group=su3_group(),
        cold_start=True,
        seed=2026,
    )

    result = generic_metropolis_sweep(
        lattice=lattice,
        beta=5.5,
        epsilon=0.05,
    )

    assert (
        0.0
        <= result.acceptance_rate
        <= 1.0
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


def test_generic_lattice_rejects_wrong_matrix_shape():
    lattice = GaugeLattice(
        shape=(4, 4),
        group=su3_group(),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        lattice.set_link(
            (0, 0),
            0,
            np.eye(
                2,
                dtype=complex,
            ),
        )


def test_generic_wilson_action_rejects_negative_beta():
    lattice = GaugeLattice(
        shape=(4, 4),
        group=su2_group(),
        cold_start=True,
    )

    with pytest.raises(ValueError):
        generic_wilson_action(
            lattice,
            beta=-1.0,
        )
