"""
Generic local gauge transformations for GaugeLattice.

For a site-dependent gauge field G(x), a positively oriented link transforms as

    U_mu(x)
        ->
        G(x) U_mu(x) G(x + mu)^dagger.

The implementation operates through MatrixGaugeGroup and GaugeLattice, so the
same transformation geometry applies to SU(2) and SU(3).

Gauge-invariant scalar observables such as the Wilson action, normalized
plaquette traces, and closed Wilson loops should remain unchanged up to
floating-point error.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ymlab.gauge_lattice import (
    GaugeLattice,
    Site,
)


GenericGaugeField = Dict[
    Site,
    np.ndarray,
]


def random_generic_gauge_field(
    lattice: GaugeLattice,
    rng: np.random.Generator | None = None,
) -> GenericGaugeField:
    """
    Construct an independent random gauge-group matrix at every lattice site.
    """
    if rng is None:
        rng = lattice.rng

    field: GenericGaugeField = {}

    for site in lattice.sites():
        field[
            site
        ] = np.asarray(
            lattice.group.random(
                rng
            ),
            dtype=complex,
        )

    validate_generic_gauge_field(
        lattice=lattice,
        gauge_field=field,
    )

    return field


def identity_generic_gauge_field(
    lattice: GaugeLattice,
) -> GenericGaugeField:
    """
    Construct the identity local gauge field.
    """
    return {
        site: np.asarray(
            lattice.group.identity(),
            dtype=complex,
        )
        for site in lattice.sites()
    }


def validate_generic_gauge_field(
    lattice: GaugeLattice,
    gauge_field: GenericGaugeField,
) -> None:
    """
    Validate site coverage, matrix shape, and group membership.
    """
    expected_sites = set(
        lattice.sites()
    )

    actual_sites = set(
        gauge_field.keys()
    )

    if actual_sites != expected_sites:
        missing = (
            expected_sites
            - actual_sites
        )

        extra = (
            actual_sites
            - expected_sites
        )

        raise ValueError(
            "Gauge field site coverage does not match lattice: "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    expected_shape = (
        lattice.group.dimension,
        lattice.group.dimension,
    )

    for site, matrix in gauge_field.items():
        value = np.asarray(
            matrix,
            dtype=complex,
        )

        if value.shape != expected_shape:
            raise ValueError(
                f"Gauge matrix at site {site} has incorrect shape."
            )

        if not lattice.group.is_member(
            value
        ):
            raise ValueError(
                f"Gauge matrix at site {site} is not in "
                f"{lattice.group.name}."
            )


def generic_transform_link(
    lattice: GaugeLattice,
    gauge_field: GenericGaugeField,
    site: Site,
    mu: int,
) -> np.ndarray:
    """
    Transform one positive link under a local gauge transformation.
    """
    site = lattice.validate_site(
        site
    )

    mu = lattice.validate_direction(
        mu
    )

    endpoint = lattice.shift(
        site,
        mu,
        1,
    )

    return (
        gauge_field[
            site
        ]
        @ lattice.get_link(
            site,
            mu,
        )
        @ lattice.group.dagger(
            gauge_field[
                endpoint
            ]
        )
    )


def gauge_transform_generic_lattice(
    lattice: GaugeLattice,
    gauge_field: GenericGaugeField,
) -> GaugeLattice:
    """
    Return a gauge-transformed independent lattice configuration.
    """
    validate_generic_gauge_field(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    transformed = GaugeLattice(
        shape=lattice.shape,
        group=lattice.group,
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            transformed.set_link(
                site,
                mu,
                generic_transform_link(
                    lattice=lattice,
                    gauge_field=gauge_field,
                    site=site,
                    mu=mu,
                ),
            )

    return transformed


def maximum_generic_link_membership_error(
    lattice: GaugeLattice,
) -> float:
    """
    Return a matrix-constraint diagnostic over all links.

    The error combines unitarity and determinant-one deviations.
    """
    maximum_error = 0.0

    identity = np.eye(
        lattice.group.dimension,
        dtype=complex,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            matrix = lattice.get_link(
                site,
                mu,
            )

            unitarity_error = float(
                np.linalg.norm(
                    lattice.group.dagger(
                        matrix
                    )
                    @ matrix
                    - identity,
                    ord="fro",
                )
            )

            determinant_error = float(
                abs(
                    np.linalg.det(
                        matrix
                    )
                    - 1.0
                )
            )

            maximum_error = max(
                maximum_error,
                unitarity_error,
                determinant_error,
            )

    return float(
        maximum_error
    )
