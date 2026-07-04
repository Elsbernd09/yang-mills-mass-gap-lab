"""
Plaquette construction for SU(2) lattice gauge theory.

The plaquette is the ordered product around an elementary square:

    U_mu(x)
    U_nu(x + mu)
    U_mu(x + nu)^dagger
    U_nu(x)^dagger

This is the lattice analogue of curvature.
"""

from __future__ import annotations

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, real_trace


def plaquette(lattice: Lattice, site: Site, mu: int, nu: int) -> np.ndarray:
    """
    Compute the plaquette U_{mu,nu}(x).

    Parameters
    ----------
    lattice:
        Lattice object.
    site:
        Starting site x.
    mu:
        First direction.
    nu:
        Second direction.

    Returns
    -------
    np.ndarray
        2x2 SU(2) plaquette matrix.
    """
    if mu == nu:
        raise ValueError("Plaquette directions must be different.")

    if mu < 0 or mu >= lattice.dim or nu < 0 or nu >= lattice.dim:
        raise ValueError("Invalid plaquette direction.")

    x = site
    x_plus_mu = lattice.shift(x, mu, 1)
    x_plus_nu = lattice.shift(x, nu, 1)

    u_mu_x = lattice.get_link(x, mu)
    u_nu_x_plus_mu = lattice.get_link(x_plus_mu, nu)
    u_mu_x_plus_nu = lattice.get_link(x_plus_nu, mu)
    u_nu_x = lattice.get_link(x, nu)

    return u_mu_x @ u_nu_x_plus_mu @ dagger(u_mu_x_plus_nu) @ dagger(u_nu_x)


def plaquette_action_density(lattice: Lattice, site: Site, mu: int, nu: int) -> float:
    """
    Compute the local Wilson plaquette action density for SU(2):

        1 - (1/2) Re Tr(U_p)
    """
    up = plaquette(lattice, site, mu, nu)
    return 1.0 - 0.5 * real_trace(up)


def average_plaquette(lattice: Lattice) -> float:
    """
    Compute the average normalized plaquette value:

        (1/2) Re Tr(U_p)

    For a cold start lattice, this should be 1.
    """
    total = 0.0
    count = 0

    for site in lattice.sites():
        for mu in range(lattice.dim):
            for nu in range(mu + 1, lattice.dim):
                up = plaquette(lattice, site, mu, nu)
                total += 0.5 * real_trace(up)
                count += 1

    return total / count


def average_plaquette_action_density(lattice: Lattice) -> float:
    """
    Compute the average Wilson plaquette action density.

    For a cold start lattice, this should be 0.
    """
    total = 0.0
    count = 0

    for site in lattice.sites():
        for mu in range(lattice.dim):
            for nu in range(mu + 1, lattice.dim):
                total += plaquette_action_density(lattice, site, mu, nu)
                count += 1

    return total / count
