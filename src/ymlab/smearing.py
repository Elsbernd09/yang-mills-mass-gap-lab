"""
Spatial-link smearing for finite SU(2) lattice gauge theory.

The spectroscopy pipeline uses a Euclidean time direction. Smearing is applied
only to links in spatial directions; temporal links are left unchanged.

For a spatial link U_mu(x), this implementation constructs oriented spatial
side paths from x to x + mu. The side-path sum transforms like the original
link under a local gauge transformation.

One smearing step forms

    X_mu(x) =
        (1 - alpha) U_mu(x)
        + alpha / N_staples * S_mu(x),

where S_mu(x) is the sum of spatial side paths and

    N_staples = 2 * (number of spatial directions - 1).

The resulting complex matrix is projected back to SU(2).

All links at a new smearing level are computed from the previous level, making
the update simultaneous rather than in-place.

This is an operator-improvement layer for finite-lattice spectroscopy. It does
not change the Monte Carlo probability distribution used to generate the
underlying gauge configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, is_su2, reunitarize


@dataclass(frozen=True)
class SmearingSummary:
    """Summary of one multi-step spatial smearing operation."""

    steps: int
    alpha: float
    time_direction: int
    maximum_link_membership_error: float


def _validate_time_direction(
    lattice: Lattice,
    time_direction: int,
) -> None:
    """Validate the selected Euclidean time direction."""
    if (
        time_direction < 0
        or time_direction >= lattice.dim
    ):
        raise ValueError(
            "Invalid time direction."
        )


def spatial_directions(
    lattice: Lattice,
    time_direction: int,
) -> list[int]:
    """
    Return all directions except the Euclidean time direction.
    """
    _validate_time_direction(
        lattice,
        time_direction,
    )

    return [
        direction
        for direction in range(lattice.dim)
        if direction != time_direction
    ]


def copy_lattice(
    lattice: Lattice,
) -> Lattice:
    """
    Return an independent copy of a lattice configuration.
    """
    copied = Lattice(
        shape=lattice.shape,
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for direction in range(lattice.dim):
            copied.set_link(
                site,
                direction,
                np.array(
                    lattice.get_link(
                        site,
                        direction,
                    ),
                    dtype=complex,
                    copy=True,
                ),
            )

    return copied


def spatial_side_path_sum(
    lattice: Lattice,
    site: Site,
    mu: int,
    time_direction: int,
) -> np.ndarray:
    """
    Compute spatial side paths associated with U_mu(x).

    Every path starts at x and ends at x + mu, so the sum transforms in the
    same gauge-covariant manner as U_mu(x).

    For each spatial nu != mu, the forward path is

        U_nu(x)
        U_mu(x + nu)
        U_nu(x + mu)^dagger,

    and the backward path is

        U_nu(x - nu)^dagger
        U_mu(x - nu)
        U_nu(x - nu + mu).
    """
    _validate_time_direction(
        lattice,
        time_direction,
    )

    if mu < 0 or mu >= lattice.dim:
        raise ValueError(
            "Invalid link direction."
        )

    if mu == time_direction:
        raise ValueError(
            "Spatial side-path sum is undefined "
            "for the selected temporal direction."
        )

    directions = spatial_directions(
        lattice,
        time_direction,
    )

    transverse_directions = [
        nu
        for nu in directions
        if nu != mu
    ]

    if len(transverse_directions) == 0:
        raise ValueError(
            "Spatial smearing requires at least "
            "two spatial directions."
        )

    total = np.zeros(
        (2, 2),
        dtype=complex,
    )

    for nu in transverse_directions:
        x_plus_nu = lattice.shift(
            site,
            nu,
            1,
        )

        x_plus_mu = lattice.shift(
            site,
            mu,
            1,
        )

        forward = (
            lattice.get_link(
                site,
                nu,
            )
            @ lattice.get_link(
                x_plus_nu,
                mu,
            )
            @ dagger(
                lattice.get_link(
                    x_plus_mu,
                    nu,
                )
            )
        )

        x_minus_nu = lattice.shift(
            site,
            nu,
            -1,
        )

        x_minus_nu_plus_mu = lattice.shift(
            x_minus_nu,
            mu,
            1,
        )

        backward = (
            dagger(
                lattice.get_link(
                    x_minus_nu,
                    nu,
                )
            )
            @ lattice.get_link(
                x_minus_nu,
                mu,
            )
            @ lattice.get_link(
                x_minus_nu_plus_mu,
                nu,
            )
        )

        total += forward
        total += backward

    return total


def smear_spatial_link(
    lattice: Lattice,
    site: Site,
    mu: int,
    time_direction: int,
    alpha: float,
) -> np.ndarray:
    """
    Compute one projected smeared spatial link.
    """
    _validate_time_direction(
        lattice,
        time_direction,
    )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must satisfy 0 <= alpha <= 1."
        )

    if mu == time_direction:
        raise ValueError(
            "Temporal links are not spatially smeared."
        )

    directions = spatial_directions(
        lattice,
        time_direction,
    )

    transverse_count = len(
        [
            nu
            for nu in directions
            if nu != mu
        ]
    )

    if transverse_count == 0:
        raise ValueError(
            "Spatial smearing requires at least "
            "two spatial directions."
        )

    number_of_side_paths = (
        2 * transverse_count
    )

    original_link = lattice.get_link(
        site,
        mu,
    )

    side_paths = spatial_side_path_sum(
        lattice=lattice,
        site=site,
        mu=mu,
        time_direction=time_direction,
    )

    combined = (
        (1.0 - alpha)
        * original_link
        + (
            alpha
            / number_of_side_paths
        )
        * side_paths
    )

    projected = reunitarize(
        combined
    )

    if not is_su2(
        projected,
        atol=1e-8,
    ):
        raise ValueError(
            "Projected smeared link is not in SU(2)."
        )

    return projected


def spatial_smearing_step(
    lattice: Lattice,
    time_direction: int,
    alpha: float,
) -> Lattice:
    """
    Perform one simultaneous spatial smearing step.

    Temporal links are copied exactly.
    """
    _validate_time_direction(
        lattice,
        time_direction,
    )

    if lattice.dim < 3:
        raise ValueError(
            "Spatial smearing requires at least "
            "three lattice dimensions."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must satisfy 0 <= alpha <= 1."
        )

    smeared = copy_lattice(
        lattice
    )

    for site in lattice.sites():
        for mu in range(lattice.dim):
            if mu == time_direction:
                continue

            smeared.set_link(
                site,
                mu,
                smear_spatial_link(
                    lattice=lattice,
                    site=site,
                    mu=mu,
                    time_direction=time_direction,
                    alpha=alpha,
                ),
            )

    return smeared


def smear_spatial_links(
    lattice: Lattice,
    time_direction: int,
    alpha: float,
    steps: int,
) -> Lattice:
    """
    Apply repeated spatial smearing steps.
    """
    _validate_time_direction(
        lattice,
        time_direction,
    )

    if steps < 0:
        raise ValueError(
            "steps must be nonnegative."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must satisfy 0 <= alpha <= 1."
        )

    current = copy_lattice(
        lattice
    )

    for _ in range(steps):
        current = spatial_smearing_step(
            lattice=current,
            time_direction=time_direction,
            alpha=alpha,
        )

    return current


def maximum_su2_link_error(
    lattice: Lattice,
) -> float:
    """
    Return the maximum SU(2) numerical constraint error over all links.
    """
    identity = np.eye(
        2,
        dtype=complex,
    )

    maximum = 0.0

    for site in lattice.sites():
        for direction in range(lattice.dim):
            link = lattice.get_link(
                site,
                direction,
            )

            unitarity_error = np.linalg.norm(
                dagger(link) @ link
                - identity,
                ord="fro",
            )

            determinant_error = abs(
                np.linalg.det(link)
                - 1.0
            )

            maximum = max(
                maximum,
                float(unitarity_error),
                float(determinant_error),
            )

    return maximum


def smear_with_summary(
    lattice: Lattice,
    time_direction: int,
    alpha: float,
    steps: int,
) -> tuple[Lattice, SmearingSummary]:
    """
    Smear a lattice and return numerical summary metadata.
    """
    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=time_direction,
        alpha=alpha,
        steps=steps,
    )

    summary = SmearingSummary(
        steps=steps,
        alpha=float(alpha),
        time_direction=time_direction,
        maximum_link_membership_error=(
            maximum_su2_link_error(
                smeared
            )
        ),
    )

    return smeared, summary
