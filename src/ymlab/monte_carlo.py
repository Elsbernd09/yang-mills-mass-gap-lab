"""
Metropolis-Hastings sampler for SU(2) lattice gauge theory.

This module implements a simple global-action Metropolis update:
1. Choose a lattice link.
2. Propose a small random SU(2) perturbation.
3. Compute the change in Wilson action.
4. Accept the update with probability min(1, exp(-Delta S)).

This is intentionally simple and transparent. Later versions can optimize the
action-difference calculation locally using staples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import small_random_su2
from ymlab.wilson_action import wilson_action


@dataclass
class MetropolisResult:
    """Container for Monte Carlo simulation diagnostics."""

    actions: list[float]
    acceptance_rates: list[float]
    average_plaquettes: list[float]


def metropolis_link_update(
    lattice: Lattice,
    site: Site,
    direction: int,
    beta: float,
    epsilon: float = 0.1,
) -> bool:
    """
    Attempt one Metropolis update on a single link.

    Parameters
    ----------
    lattice:
        Lattice configuration.
    site:
        Site where the link begins.
    direction:
        Link direction.
    beta:
        Wilson action coupling parameter.
    epsilon:
        Size of the random SU(2) perturbation.

    Returns
    -------
    bool
        True if the proposal is accepted, False otherwise.
    """
    old_link = lattice.get_link(site, direction).copy()
    old_action = wilson_action(lattice, beta)

    proposal = small_random_su2(epsilon=epsilon, rng=lattice.rng) @ old_link
    lattice.set_link(site, direction, proposal)

    new_action = wilson_action(lattice, beta)
    delta_action = new_action - old_action

    if delta_action <= 0:
        return True

    acceptance_probability = np.exp(-delta_action)

    if lattice.rng.random() < acceptance_probability:
        return True

    lattice.set_link(site, direction, old_link)
    return False


def metropolis_sweep(
    lattice: Lattice,
    beta: float,
    epsilon: float = 0.1,
) -> float:
    """
    Perform one full Metropolis sweep over every link.

    Returns
    -------
    float
        Acceptance rate for the sweep.
    """
    accepted = 0
    total = 0

    for site in list(lattice.sites()):
        for direction in range(lattice.dim):
            did_accept = metropolis_link_update(
                lattice=lattice,
                site=site,
                direction=direction,
                beta=beta,
                epsilon=epsilon,
            )
            accepted += int(did_accept)
            total += 1

    return accepted / total


def run_metropolis(
    lattice: Lattice,
    beta: float,
    sweeps: int,
    epsilon: float = 0.1,
    measurement_interval: int = 1,
) -> MetropolisResult:
    """
    Run a Metropolis-Hastings simulation.

    Parameters
    ----------
    lattice:
        Initial lattice configuration.
    beta:
        Wilson action coupling parameter.
    sweeps:
        Number of full lattice sweeps.
    epsilon:
        Proposal size.
    measurement_interval:
        Record diagnostics every this many sweeps.

    Returns
    -------
    MetropolisResult
        Actions, acceptance rates, and average plaquette measurements.
    """
    if sweeps <= 0:
        raise ValueError("sweeps must be positive.")

    if measurement_interval <= 0:
        raise ValueError("measurement_interval must be positive.")

    from ymlab.plaquette import average_plaquette

    actions: list[float] = []
    acceptance_rates: list[float] = []
    average_plaquettes: list[float] = []

    for sweep in range(1, sweeps + 1):
        acceptance_rate = metropolis_sweep(lattice, beta=beta, epsilon=epsilon)

        if sweep % measurement_interval == 0:
            actions.append(wilson_action(lattice, beta))
            acceptance_rates.append(acceptance_rate)
            average_plaquettes.append(average_plaquette(lattice))

    return MetropolisResult(
        actions=actions,
        acceptance_rates=acceptance_rates,
        average_plaquettes=average_plaquettes,
    )
