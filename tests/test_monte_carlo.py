import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.monte_carlo import (
    metropolis_link_update,
    metropolis_sweep,
    run_metropolis,
    MetropolisResult,
)
from ymlab.su2 import is_su2


def test_single_metropolis_update_preserves_su2():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    metropolis_link_update(
        lattice=lattice,
        site=(0, 0),
        direction=0,
        beta=2.0,
        epsilon=0.1,
    )

    assert is_su2(lattice.get_link((0, 0), 0))


def test_metropolis_sweep_acceptance_rate_is_probability():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    acceptance_rate = metropolis_sweep(
        lattice=lattice,
        beta=2.0,
        epsilon=0.1,
    )

    assert 0.0 <= acceptance_rate <= 1.0


def test_run_metropolis_returns_result_object():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    result = run_metropolis(
        lattice=lattice,
        beta=2.0,
        sweeps=5,
        epsilon=0.1,
        measurement_interval=1,
    )

    assert isinstance(result, MetropolisResult)
    assert len(result.actions) == 5
    assert len(result.acceptance_rates) == 5
    assert len(result.average_plaquettes) == 5


def test_run_metropolis_measurement_interval():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    result = run_metropolis(
        lattice=lattice,
        beta=2.0,
        sweeps=10,
        epsilon=0.1,
        measurement_interval=2,
    )

    assert len(result.actions) == 5
    assert len(result.acceptance_rates) == 5
    assert len(result.average_plaquettes) == 5


def test_run_metropolis_rejects_bad_sweep_count():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    with pytest.raises(ValueError):
        run_metropolis(
            lattice=lattice,
            beta=2.0,
            sweeps=0,
            epsilon=0.1,
        )


def test_run_metropolis_observables_are_finite():
    lattice = Lattice(shape=(3, 3), cold_start=True, seed=123)

    result = run_metropolis(
        lattice=lattice,
        beta=2.0,
        sweeps=5,
        epsilon=0.1,
        measurement_interval=1,
    )

    assert np.all(np.isfinite(result.actions))
    assert np.all(np.isfinite(result.acceptance_rates))
    assert np.all(np.isfinite(result.average_plaquettes))
