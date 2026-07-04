"""
Wilson action for SU(2) lattice gauge theory.

For SU(2), the Wilson action is

    S_W(U) = beta * sum_p [1 - (1/2) Re Tr(U_p)]

where U_p is the plaquette matrix around an elementary square.
"""

from __future__ import annotations

from ymlab.lattice import Lattice
from ymlab.plaquette import plaquette_action_density


def wilson_action(lattice: Lattice, beta: float) -> float:
    """
    Compute the Wilson action of an SU(2) lattice configuration.

    Parameters
    ----------
    lattice:
        Lattice object containing SU(2) link variables.
    beta:
        Inverse coupling parameter. Larger beta corresponds to weaker coupling.

    Returns
    -------
    float
        Wilson action.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    total = 0.0

    for site in lattice.sites():
        for mu in range(lattice.dim):
            for nu in range(mu + 1, lattice.dim):
                total += plaquette_action_density(lattice, site, mu, nu)

    return beta * total


def action_per_plaquette(lattice: Lattice, beta: float) -> float:
    """
    Compute Wilson action divided by the number of plaquettes.

    This gives a normalized measure that is easier to compare across lattice sizes.
    """
    number_of_plaquettes = 0

    for _site in lattice.sites():
        for mu in range(lattice.dim):
            for nu in range(mu + 1, lattice.dim):
                number_of_plaquettes += 1

    if number_of_plaquettes == 0:
        raise ValueError("Lattice has no plaquettes.")

    return wilson_action(lattice, beta) / number_of_plaquettes


def number_of_plaquettes(lattice: Lattice) -> int:
    """
    Count the number of positively oriented plaquettes.

    In d dimensions, each site has d choose 2 coordinate planes.
    """
    count = 0

    for _site in lattice.sites():
        for mu in range(lattice.dim):
            for nu in range(mu + 1, lattice.dim):
                count += 1

    return count
