import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.monte_carlo import (
    metropolis_sweep,
)
from ymlab.overrelaxation import (
    HybridSweepResult,
    OverrelaxationLinkResult,
    OverrelaxationSweepResult,
    hybrid_metropolis_overrelaxation_sweep,
    normalized_su2_staple,
    overrelaxation_link_update,
    overrelaxation_proposal,
    overrelaxation_sweep,
)
from ymlab.su2 import (
    identity,
    is_su2,
)
from ymlab.wilson_action import (
    wilson_action,
)


def nontrivial_lattice() -> Lattice:
    lattice = Lattice(
        shape=(
            4,
            4,
            4,
        ),
        cold_start=True,
        seed=2026,
    )

    for _ in range(5):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.18,
        )

    return lattice


def test_cold_normalized_staple_is_identity():
    lattice = Lattice(
        shape=(
            4,
            4,
            4,
        ),
        cold_start=True,
    )

    normalized, scale = (
        normalized_su2_staple(
            lattice=lattice,
            site=(
                0,
                0,
                0,
            ),
            mu=1,
        )
    )

    assert np.allclose(
        normalized,
        identity(),
    )

    assert np.isclose(
        scale,
        4.0,
    )


def test_overrelaxation_proposal_is_su2():
    lattice = nontrivial_lattice()

    proposal, scale = (
        overrelaxation_proposal(
            lattice=lattice,
            site=(
                1,
                2,
                3,
            ),
            mu=1,
        )
    )

    assert scale > 0.0

    assert is_su2(
        proposal,
        atol=1e-8,
    )


def test_single_link_preserves_local_action():
    lattice = nontrivial_lattice()

    result = (
        overrelaxation_link_update(
            lattice=lattice,
            site=(
                1,
                2,
                3,
            ),
            mu=1,
            beta=2.0,
        )
    )

    assert isinstance(
        result,
        OverrelaxationLinkResult,
    )

    assert np.isclose(
        result.local_action_before,
        result.local_action_after,
        atol=1e-10,
        rtol=1e-10,
    )

    assert (
        result.absolute_action_error
        < 1e-9
    )


def test_single_link_preserves_global_wilson_action():
    lattice = nontrivial_lattice()

    before = wilson_action(
        lattice,
        beta=2.0,
    )

    overrelaxation_link_update(
        lattice=lattice,
        site=(
            2,
            1,
            0,
        ),
        mu=2,
        beta=2.0,
    )

    after = wilson_action(
        lattice,
        beta=2.0,
    )

    assert np.isclose(
        before,
        after,
        atol=1e-9,
        rtol=1e-10,
    )


def test_full_sweep_preserves_global_wilson_action():
    lattice = nontrivial_lattice()

    before = wilson_action(
        lattice,
        beta=2.0,
    )

    result = overrelaxation_sweep(
        lattice=lattice,
        beta=2.0,
    )

    after = wilson_action(
        lattice,
        beta=2.0,
    )

    assert isinstance(
        result,
        OverrelaxationSweepResult,
    )

    assert result.updated_links > 0

    assert np.isclose(
        before,
        after,
        atol=1e-8,
        rtol=1e-10,
    )


def test_overrelaxation_preserves_all_links_in_su2():
    lattice = nontrivial_lattice()

    overrelaxation_sweep(
        lattice=lattice,
        beta=2.0,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            assert is_su2(
                lattice.get_link(
                    site,
                    mu,
                ),
                atol=1e-8,
            )


def test_cold_lattice_is_fixed_point():
    lattice = Lattice(
        shape=(
            4,
            4,
            4,
        ),
        cold_start=True,
    )

    overrelaxation_sweep(
        lattice=lattice,
        beta=2.0,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            assert np.allclose(
                lattice.get_link(
                    site,
                    mu,
                ),
                identity(),
                atol=1e-12,
                rtol=1e-12,
            )


def test_hybrid_sweep_runs():
    lattice = nontrivial_lattice()

    result = (
        hybrid_metropolis_overrelaxation_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.18,
            overrelaxation_sweeps=2,
        )
    )

    assert isinstance(
        result,
        HybridSweepResult,
    )

    assert (
        0.0
        <= result.metropolis_acceptance_rate
        <= 1.0
    )

    assert result.overrelaxation_sweeps == 2

    assert (
        result.overrelaxation_updated_links
        > 0
    )


def test_negative_overrelaxation_count_rejected():
    lattice = nontrivial_lattice()

    with pytest.raises(ValueError):
        hybrid_metropolis_overrelaxation_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.18,
            overrelaxation_sweeps=-1,
        )


def test_negative_beta_rejected():
    lattice = nontrivial_lattice()

    with pytest.raises(ValueError):
        overrelaxation_sweep(
            lattice=lattice,
            beta=-1.0,
        )
