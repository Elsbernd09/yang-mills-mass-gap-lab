"""
Metropolis-Hastings sampler for SU(2) lattice gauge theory.

This module implements local staple-based Metropolis updates:
1. Choose a lattice link U_mu(x).
2. Compute the staple around that link.
3. Propose a small random SU(2) perturbation.
4. Compute the local Wilson action difference.
5. Accept with probability min(1, exp(-Delta S)).

The simulation runner supports burn-in and measurement intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.plaquette import average_plaquette
from ymlab.staples import local_action_difference
from ymlab.su2 import small_random_su2
from ymlab.wilson_action import wilson_action


@dataclass
class MetropolisResult:
    """Container for Monte Carlo simulation diagnostics."""

    actions: list[float]
    acceptance_rates: list[float]
    average_plaquettes: list[float]
    burn_in: int = 0
    total_sweeps: int = 0
    measurement_interval: int = 1


def metropolis_link_update(
    lattice: Lattice,
    site: Site,
    direction: int,
    beta: float,
    epsilon: float = 0.1,
) -> bool:
    """
    Attempt one local staple-based Metropolis update on a single link.

    Returns True if the proposal is accepted and False otherwise.
    """
    old_link = lattice.get_link(site, direction).copy()
    proposal = small_random_su2(epsilon=epsilon, rng=lattice.rng) @ old_link

    delta_action = local_action_difference(
        lattice=lattice,
        site=site,
        mu=direction,
        proposal=proposal,
        beta=beta,
    )

    if delta_action <= 0:
        lattice.set_link(site, direction, proposal)
        return True

    acceptance_probability = np.exp(-delta_action)

    if lattice.rng.random() < acceptance_probability:
        lattice.set_link(site, direction, proposal)
        return True

    return False


def metropolis_sweep(
    lattice: Lattice,
    beta: float,
    epsilon: float = 0.1,
) -> float:
    """
    Perform one full Metropolis sweep over every positive directed link.

    Returns the acceptance rate.
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
    burn_in: int = 0,
) -> MetropolisResult:
    """
    Run a Metropolis-Hastings simulation and record diagnostics.

    Parameters
    ----------
    lattice:
        Lattice configuration.
    beta:
        Wilson action coupling parameter.
    sweeps:
        Total number of sweeps, including burn-in.
    epsilon:
        Proposal size.
    measurement_interval:
        Record measurements every this many sweeps after burn-in.
    burn_in:
        Number of initial sweeps discarded before measurements begin.
    """
    if sweeps <= 0:
        raise ValueError("sweeps must be positive.")

    if measurement_interval <= 0:
        raise ValueError("measurement_interval must be positive.")

    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative.")

    if burn_in >= sweeps:
        raise ValueError("burn_in must be smaller than sweeps.")

    actions: list[float] = []
    acceptance_rates: list[float] = []
    average_plaquettes: list[float] = []

    for sweep in range(1, sweeps + 1):
        acceptance_rate = metropolis_sweep(lattice, beta=beta, epsilon=epsilon)

        if sweep > burn_in and (sweep - burn_in) % measurement_interval == 0:
            actions.append(wilson_action(lattice, beta))
            acceptance_rates.append(acceptance_rate)
            average_plaquettes.append(average_plaquette(lattice))

    return MetropolisResult(
        actions=actions,
        acceptance_rates=acceptance_rates,
        average_plaquettes=average_plaquettes,
        burn_in=burn_in,
        total_sweeps=sweeps,
        measurement_interval=measurement_interval,
    )
