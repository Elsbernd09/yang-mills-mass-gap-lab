"""
Gauge transformations for finite SU(2) lattice gauge theory.

A lattice gauge transformation assigns a group element G(x) to every site x.

A positively oriented link transforms as

    U_mu(x) -> G(x) U_mu(x) G(x + mu)^dagger.

Closed-loop trace observables should remain invariant under this transformation.

This module provides gauge-field generation, link transformation, full-lattice
transformation, and numerical comparison utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, is_su2, random_su2


GaugeField = dict[Site, np.ndarray]


@dataclass(frozen=True)
class GaugeInvarianceComparison:
    """Numerical before/after comparison for a scalar observable."""

    name: str
    before: float
    after: float
    absolute_difference: float
    relative_difference: float
    invariant: bool


def random_gauge_field(
    lattice: Lattice,
    seed: int | None = None,
) -> GaugeField:
    """
    Generate an independent random SU(2) gauge transformation G(x) at every site.
    """
    rng = np.random.default_rng(seed)

    return {
        site: random_su2(rng)
        for site in lattice.sites()
    }


def identity_gauge_field(lattice: Lattice) -> GaugeField:
    """
    Generate the identity gauge transformation at every site.
    """
    from ymlab.su2 import identity

    return {
        site: identity()
        for site in lattice.sites()
    }


def validate_gauge_field(
    lattice: Lattice,
    gauge_field: GaugeField,
    atol: float = 1e-10,
) -> bool:
    """
    Validate that the gauge field contains one SU(2) matrix for every lattice site.
    """
    lattice_sites = set(lattice.sites())

    if set(gauge_field.keys()) != lattice_sites:
        return False

    return all(
        is_su2(gauge_field[site], atol=atol)
        for site in lattice_sites
    )


def transform_link(
    lattice: Lattice,
    gauge_field: GaugeField,
    site: Site,
    direction: int,
) -> np.ndarray:
    """
    Transform one lattice link.

    The transformation law is

        U_mu(x) -> G(x) U_mu(x) G(x + mu)^dagger.
    """
    if not validate_gauge_field(lattice, gauge_field):
        raise ValueError("Invalid gauge field.")

    if direction < 0 or direction >= lattice.dim:
        raise ValueError("Invalid link direction.")

    next_site = lattice.shift(site, direction, 1)

    g_x = gauge_field[site]
    u = lattice.get_link(site, direction)
    g_next = gauge_field[next_site]

    transformed = g_x @ u @ dagger(g_next)

    if not is_su2(transformed, atol=1e-8):
        raise ValueError("Gauge-transformed link left SU(2) numerically.")

    return transformed


def gauge_transform_lattice(
    lattice: Lattice,
    gauge_field: GaugeField,
) -> Lattice:
    """
    Return a new lattice containing the gauge-transformed link configuration.

    The original lattice is not modified.
    """
    if not validate_gauge_field(lattice, gauge_field):
        raise ValueError("Invalid gauge field.")

    transformed_lattice = Lattice(
        shape=lattice.shape,
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for direction in range(lattice.dim):
            transformed_lattice.set_link(
                site,
                direction,
                transform_link(
                    lattice=lattice,
                    gauge_field=gauge_field,
                    site=site,
                    direction=direction,
                ),
            )

    return transformed_lattice


def max_link_membership_error(lattice: Lattice) -> float:
    """
    Return a simple maximum numerical deviation from SU(2) constraints.

    For each link U, this measures the maximum of:

    - ||U^dagger U - I||_F
    - |det(U) - 1|
    """
    from ymlab.su2 import identity

    maximum = 0.0

    for site in lattice.sites():
        for direction in range(lattice.dim):
            u = lattice.get_link(site, direction)

            unitarity_error = np.linalg.norm(
                dagger(u) @ u - identity(),
                ord="fro",
            )
            determinant_error = abs(np.linalg.det(u) - 1.0)

            maximum = max(
                maximum,
                float(unitarity_error),
                float(determinant_error),
            )

    return maximum


def compare_scalar_observable(
    name: str,
    before: float,
    after: float,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> GaugeInvarianceComparison:
    """
    Compare a scalar observable before and after a gauge transformation.
    """
    before = float(before)
    after = float(after)

    absolute_difference = abs(after - before)

    scale = max(abs(before), abs(after), np.finfo(float).eps)
    relative_difference = absolute_difference / scale

    invariant = bool(
        np.isclose(
            before,
            after,
            atol=atol,
            rtol=rtol,
        )
    )

    return GaugeInvarianceComparison(
        name=name,
        before=before,
        after=after,
        absolute_difference=float(absolute_difference),
        relative_difference=float(relative_difference),
        invariant=invariant,
    )
