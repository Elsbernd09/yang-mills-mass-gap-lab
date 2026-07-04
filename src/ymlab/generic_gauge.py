"""
Generic matrix-gauge finite-lattice observables and local Metropolis updates.

The functions in this module operate on GaugeLattice and use the supplied
MatrixGaugeGroup backend.

The Wilson plaquette density is normalized by the matrix dimension N:

    1 - (1/N) Re Tr(U_p).

The Wilson action is

    S
        =
        beta sum_p [
            1 - (1/N) Re Tr(U_p)
        ].

The reverse-oriented local staple V is defined so that

    U_mu(x) V

closes the plaquette contributions containing the selected link.

The link-dependent local action is

    S_local
        =
        -(beta / N) Re Tr[
            U_mu(x) V
        ].

This gives a generic local Metropolis update when the group backend supplies
small random near-identity proposals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.gauge_lattice import (
    GaugeLattice,
    Site,
)


@dataclass(frozen=True)
class GenericMetropolisSweepResult:
    """One generic matrix-group Metropolis sweep."""

    accepted_updates: int
    attempted_updates: int
    acceptance_rate: float


def generic_plaquette(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    nu: int,
) -> np.ndarray:
    """
    Return the oriented plaquette

        U_mu(x)
        U_nu(x + mu)
        U_mu(x + nu)^dagger
        U_nu(x)^dagger.
    """
    if mu == nu:
        raise ValueError(
            "Plaquette directions must be distinct."
        )

    lattice.validate_direction(
        mu
    )

    lattice.validate_direction(
        nu
    )

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

    return (
        lattice.get_link(
            site,
            mu,
        )
        @ lattice.get_link(
            x_plus_mu,
            nu,
        )
        @ lattice.group.dagger(
            lattice.get_link(
                x_plus_nu,
                mu,
            )
        )
        @ lattice.group.dagger(
            lattice.get_link(
                site,
                nu,
            )
        )
    )


def generic_normalized_real_trace(
    lattice: GaugeLattice,
    matrix: np.ndarray,
) -> float:
    """
    Return (1/N) Re Tr(matrix).
    """
    return float(
        lattice.group.trace_normalization
        * lattice.group.real_trace(
            matrix
        )
    )


def generic_plaquette_action_density(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    nu: int,
) -> float:
    """
    Return 1 - (1/N) Re Tr(U_p).
    """
    return float(
        1.0
        - generic_normalized_real_trace(
            lattice,
            generic_plaquette(
                lattice=lattice,
                site=site,
                mu=mu,
                nu=nu,
            ),
        )
    )


def generic_average_plaquette(
    lattice: GaugeLattice,
) -> float:
    """Return the average normalized real plaquette trace."""
    values = []

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            for nu in range(
                mu + 1,
                lattice.dim,
            ):
                values.append(
                    generic_normalized_real_trace(
                        lattice,
                        generic_plaquette(
                            lattice=lattice,
                            site=site,
                            mu=mu,
                            nu=nu,
                        ),
                    )
                )

    return float(
        np.mean(
            values
        )
    )


def generic_number_of_plaquettes(
    lattice: GaugeLattice,
) -> int:
    """Return the number of unoriented elementary plaquettes."""
    return (
        lattice.number_of_sites()
        * lattice.dim
        * (
            lattice.dim - 1
        )
        // 2
    )


def generic_wilson_action(
    lattice: GaugeLattice,
    beta: float,
) -> float:
    """
    Compute the generic Wilson plaquette action.
    """
    if beta < 0.0:
        raise ValueError(
            "beta must be nonnegative."
        )

    total = 0.0

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            for nu in range(
                mu + 1,
                lattice.dim,
            ):
                total += (
                    generic_plaquette_action_density(
                        lattice=lattice,
                        site=site,
                        mu=mu,
                        nu=nu,
                    )
                )

    return float(
        beta * total
    )


def generic_staple(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
) -> np.ndarray:
    """
    Return the reverse-oriented Wilson-action staple.

    Every staple path runs from x + mu back to x so that

        U_mu(x) V

    forms a closed loop.
    """
    lattice.validate_direction(
        mu
    )

    total = np.zeros(
        (
            lattice.group.dimension,
            lattice.group.dimension,
        ),
        dtype=complex,
    )

    for nu in range(
        lattice.dim
    ):
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
            @ lattice.group.dagger(
                lattice.get_link(
                    x_plus_nu,
                    mu,
                )
            )
            @ lattice.group.dagger(
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

        x_plus_mu_minus_nu = (
            lattice.shift(
                x_minus_nu,
                mu,
                1,
            )
        )

        backward = (
            lattice.group.dagger(
                lattice.get_link(
                    x_plus_mu_minus_nu,
                    nu,
                )
            )
            @ lattice.group.dagger(
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

        total += (
            forward
            + backward
        )

    return total


def generic_local_link_action(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    beta: float,
    link_matrix: np.ndarray | None = None,
) -> float:
    """
    Return the link-dependent local Wilson action.
    """
    if beta < 0.0:
        raise ValueError(
            "beta must be nonnegative."
        )

    link = (
        lattice.get_link(
            site,
            mu,
        )
        if link_matrix is None
        else np.asarray(
            link_matrix,
            dtype=complex,
        )
    )

    expected_shape = (
        lattice.group.dimension,
        lattice.group.dimension,
    )

    if link.shape != expected_shape:
        raise ValueError(
            "Link matrix shape does not match gauge-group dimension."
        )

    if not lattice.group.is_member(
        link
    ):
        raise ValueError(
            "Candidate link is not a gauge-group member."
        )

    value = (
        -beta
        * lattice.group.trace_normalization
        * lattice.group.real_trace(
            link
            @ generic_staple(
                lattice=lattice,
                site=site,
                mu=mu,
            )
        )
    )

    return float(
        value
    )


def generic_local_action_difference(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
) -> float:
    """
    Return S_local(proposal) - S_local(current).
    """
    old_action = (
        generic_local_link_action(
            lattice=lattice,
            site=site,
            mu=mu,
            beta=beta,
        )
    )

    new_action = (
        generic_local_link_action(
            lattice=lattice,
            site=site,
            mu=mu,
            beta=beta,
            link_matrix=proposal,
        )
    )

    return float(
        new_action
        - old_action
    )


def generic_metropolis_link_update(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    beta: float,
    epsilon: float,
) -> bool:
    """
    Attempt one local matrix-group Metropolis update.
    """
    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive."
        )

    old_link = lattice.get_link(
        site,
        mu,
    )

    perturbation = lattice.group.small_random(
        epsilon,
        lattice.rng,
    )

    proposal = (
        perturbation
        @ old_link
    )

    if not lattice.group.is_member(
        proposal
    ):
        raise ValueError(
            "Generic proposal is not a gauge-group member."
        )

    delta_action = (
        generic_local_action_difference(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=beta,
        )
    )

    accept = bool(
        delta_action <= 0.0
        or lattice.rng.random()
        < np.exp(
            -delta_action
        )
    )

    if accept:
        lattice.set_link(
            site,
            mu,
            proposal,
        )

    return accept


def generic_metropolis_sweep(
    lattice: GaugeLattice,
    beta: float,
    epsilon: float,
) -> GenericMetropolisSweepResult:
    """
    Perform one generic local Metropolis sweep.
    """
    accepted = 0
    attempted = 0

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            attempted += 1

            accepted += int(
                generic_metropolis_link_update(
                    lattice=lattice,
                    site=site,
                    mu=mu,
                    beta=beta,
                    epsilon=epsilon,
                )
            )

    acceptance_rate = (
        accepted / attempted
        if attempted > 0
        else 0.0
    )

    return GenericMetropolisSweepResult(
        accepted_updates=accepted,
        attempted_updates=attempted,
        acceptance_rate=float(
            acceptance_rate
        ),
    )


def generic_rectangular_wilson_loop(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    nu: int,
    width: int,
    height: int,
) -> float:
    """
    Compute one normalized rectangular Wilson loop.
    """
    if mu == nu:
        raise ValueError(
            "Wilson-loop directions must be distinct."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Wilson-loop dimensions must be positive."
        )

    current_site = lattice.validate_site(
        site
    )

    product_matrix = (
        lattice.group.identity()
    )

    for _ in range(
        width
    ):
        product_matrix = (
            product_matrix
            @ lattice.get_link(
                current_site,
                mu,
            )
        )

        current_site = lattice.shift(
            current_site,
            mu,
            1,
        )

    for _ in range(
        height
    ):
        product_matrix = (
            product_matrix
            @ lattice.get_link(
                current_site,
                nu,
            )
        )

        current_site = lattice.shift(
            current_site,
            nu,
            1,
        )

    for _ in range(
        width
    ):
        current_site = lattice.shift(
            current_site,
            mu,
            -1,
        )

        product_matrix = (
            product_matrix
            @ lattice.group.dagger(
                lattice.get_link(
                    current_site,
                    mu,
                )
            )
        )

    for _ in range(
        height
    ):
        current_site = lattice.shift(
            current_site,
            nu,
            -1,
        )

        product_matrix = (
            product_matrix
            @ lattice.group.dagger(
                lattice.get_link(
                    current_site,
                    nu,
                )
            )
        )

    return generic_normalized_real_trace(
        lattice,
        product_matrix,
    )
