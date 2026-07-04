"""
Staple construction for SU(2) lattice gauge theory.

In lattice gauge theory, a single link U_mu(x) appears only in the plaquettes
touching that link. The sum of the neighboring path products is called the
staple.

For each direction nu != mu, there are two staples:
1. Forward staple in the +nu direction.
2. Backward staple in the -nu direction.

The local Wilson contribution involving U_mu(x) can be written using

    Re Tr(U_mu(x) * staple)

This allows Metropolis updates to compute local action differences instead of
recomputing the full Wilson action.
"""

from __future__ import annotations

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, real_trace


def staple(lattice: Lattice, site: Site, mu: int) -> np.ndarray:
    """
    Compute the SU(2) staple matrix associated with link U_mu(site).

    Parameters
    ----------
    lattice:
        Lattice configuration.
    site:
        Starting site x.
    mu:
        Link direction.

    Returns
    -------
    np.ndarray
        2x2 complex staple matrix.
    """
    if mu < 0 or mu >= lattice.dim:
        raise ValueError("Invalid link direction.")

    total = np.zeros((2, 2), dtype=np.complex128)
    x = site

    for nu in range(lattice.dim):
        if nu == mu:
            continue

        # Forward staple:
        # U_nu(x + mu) U_mu(x + nu)^dagger U_nu(x)^dagger
        x_plus_mu = lattice.shift(x, mu, 1)
        x_plus_nu = lattice.shift(x, nu, 1)

        forward = (
            lattice.get_link(x_plus_mu, nu)
            @ dagger(lattice.get_link(x_plus_nu, mu))
            @ dagger(lattice.get_link(x, nu))
        )

        # Backward staple:
        # U_nu(x + mu - nu)^dagger U_mu(x - nu)^dagger? 
        #
        # For the local action with U_mu(x), the neighboring backward plaquette
        # contributes the path:
        # U_nu(x + mu - nu)^dagger U_mu(x - nu)^\dagger? 
        #
        # To maintain the local form ReTr(U_mu(x) * staple), the correct path is:
        # U_nu(x + mu - nu)^dagger U_mu(x - nu)^dagger? not dimensionally correct
        #
        # We instead use the standard backward staple:
        # U_nu(x + mu - nu)^dagger U_mu(x - nu) U_nu(x - nu)
        x_minus_nu = lattice.shift(x, nu, -1)
        x_plus_mu_minus_nu = lattice.shift(x_plus_mu, nu, -1)

        backward = (
            dagger(lattice.get_link(x_plus_mu_minus_nu, nu))
            @ lattice.get_link(x_minus_nu, mu)
            @ lattice.get_link(x_minus_nu, nu)
        )

        total += forward + backward

    return total


def local_link_action(lattice: Lattice, site: Site, mu: int, beta: float) -> float:
    """
    Compute the local Wilson action contribution involving one link.

    For SU(2), the link-dependent part is proportional to

        - (beta / 2) Re Tr(U_mu(x) * staple)

    Constants independent of the link cancel in Metropolis differences.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    u = lattice.get_link(site, mu)
    st = staple(lattice, site, mu)

    return -0.5 * beta * real_trace(u @ st)


def local_action_difference(
    lattice: Lattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
) -> float:
    """
    Compute the local action difference for replacing U_mu(site) by proposal.

    The staple is computed from the current neighboring links. The proposed
    link is not written into the lattice by this function.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    old_link = lattice.get_link(site, mu)
    st = staple(lattice, site, mu)

    old_local = -0.5 * beta * real_trace(old_link @ st)
    new_local = -0.5 * beta * real_trace(proposal @ st)

    return new_local - old_local
