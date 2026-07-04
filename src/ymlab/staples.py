"""
Staple construction and local Wilson-action contributions for SU(2).

For a positively oriented link U_mu(x), the local plaquette contribution can be
written in terms of a staple matrix V_mu(x):

    S_local = -(beta / 2) Re Tr[U_mu(x) V_mu(x)]

up to Wilson-action constants independent of the selected link.

The staple contains forward and backward contributions for every nu != mu.

The implementation is validated against full Wilson-action differences in
multiple lattice dimensions.
"""

from __future__ import annotations

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, real_trace


def staple(
    lattice: Lattice,
    site: Site,
    mu: int,
) -> np.ndarray:
    """
    Compute the SU(2) staple sum associated with U_mu(x).

    For every nu != mu, the forward contribution is

        U_nu(x + mu)
        U_mu(x + nu)^dagger
        U_nu(x)^dagger

    and the backward contribution is

        U_nu(x + mu - nu)^dagger
        U_mu(x - nu)^dagger
        U_nu(x - nu).

    With this convention, the link-dependent Wilson action is proportional to

        -Re Tr[U_mu(x) staple(x, mu)].
    """
    if mu < 0 or mu >= lattice.dim:
        raise ValueError("Invalid link direction.")

    total = np.zeros(
        (2, 2),
        dtype=complex,
    )

    for nu in range(lattice.dim):
        if nu == mu:
            continue

        x_plus_mu = lattice.shift(
            site,
            mu,
            1,
        )
        x_plus_nu = lattice.shift(
            site,
            nu,
            1,
        )

        forward = (
            lattice.get_link(
                x_plus_mu,
                nu,
            )
            @ dagger(
                lattice.get_link(
                    x_plus_nu,
                    mu,
                )
            )
            @ dagger(
                lattice.get_link(
                    site,
                    nu,
                )
            )
        )

        x_minus_nu = lattice.shift(
            site,
            nu,
            -1,
        )
        x_plus_mu_minus_nu = lattice.shift(
            x_minus_nu,
            mu,
            1,
        )

        backward = (
            dagger(
                lattice.get_link(
                    x_plus_mu_minus_nu,
                    nu,
                )
            )
            @ dagger(
                lattice.get_link(
                    x_minus_nu,
                    mu,
                )
            )
            @ lattice.get_link(
                x_minus_nu,
                nu,
            )
        )

        total += forward
        total += backward

    return total


def local_link_action(
    lattice: Lattice,
    site: Site,
    mu: int,
    beta: float,
    link_matrix: np.ndarray | None = None,
) -> float:
    """
    Compute the link-dependent local Wilson action.

    Wilson-action constants independent of the selected link are omitted.

    Therefore this function is intended for action differences.
    """
    if beta < 0.0:
        raise ValueError("beta must be nonnegative.")

    link = (
        lattice.get_link(site, mu)
        if link_matrix is None
        else np.asarray(
            link_matrix,
            dtype=complex,
        )
    )

    local_staple = staple(
        lattice=lattice,
        site=site,
        mu=mu,
    )

    return float(
        -0.5
        * beta
        * real_trace(
            link @ local_staple
        )
    )


def local_action_difference(
    lattice: Lattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
) -> float:
    """
    Compute S_local(proposal) - S_local(current link).

    The surrounding lattice is held fixed.
    """
    old_action = local_link_action(
        lattice=lattice,
        site=site,
        mu=mu,
        beta=beta,
    )

    new_action = local_link_action(
        lattice=lattice,
        site=site,
        mu=mu,
        beta=beta,
        link_matrix=proposal,
    )

    return float(
        new_action - old_action
    )
