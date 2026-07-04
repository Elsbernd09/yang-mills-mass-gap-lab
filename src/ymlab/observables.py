"""
Gauge-invariant observables for SU(2) lattice gauge theory.

This module implements Wilson loops, one of the central observables in
lattice gauge theory.

A rectangular Wilson loop is the trace of the ordered product of link variables
around a closed rectangle on the lattice.
"""

from __future__ import annotations

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.su2 import dagger, identity, real_trace


def rectangular_wilson_loop(
    lattice: Lattice,
    site: Site,
    mu: int,
    nu: int,
    width: int,
    height: int,
) -> float:
    """
    Compute the normalized SU(2) rectangular Wilson loop.

    The loop starts at `site`, moves:
    1. forward in direction mu for `width` steps
    2. forward in direction nu for `height` steps
    3. backward in direction mu for `width` steps
    4. backward in direction nu for `height` steps

    The observable is

        (1/2) Re Tr(product around loop)

    Parameters
    ----------
    lattice:
        Lattice configuration.
    site:
        Starting lattice site.
    mu:
        Horizontal direction.
    nu:
        Vertical direction.
    width:
        Number of links in the mu direction.
    height:
        Number of links in the nu direction.

    Returns
    -------
    float
        Normalized Wilson loop value.
    """
    if mu == nu:
        raise ValueError("Wilson loop directions must be different.")

    if width <= 0 or height <= 0:
        raise ValueError("Wilson loop width and height must be positive.")

    if mu < 0 or mu >= lattice.dim or nu < 0 or nu >= lattice.dim:
        raise ValueError("Invalid Wilson loop direction.")

    current = site
    product_matrix = identity()

    # Move forward in mu.
    for _ in range(width):
        product_matrix = product_matrix @ lattice.get_link(current, mu)
        current = lattice.shift(current, mu, 1)

    # Move forward in nu.
    for _ in range(height):
        product_matrix = product_matrix @ lattice.get_link(current, nu)
        current = lattice.shift(current, nu, 1)

    # Move backward in mu.
    for _ in range(width):
        previous = lattice.shift(current, mu, -1)
        product_matrix = product_matrix @ dagger(lattice.get_link(previous, mu))
        current = previous

    # Move backward in nu.
    for _ in range(height):
        previous = lattice.shift(current, nu, -1)
        product_matrix = product_matrix @ dagger(lattice.get_link(previous, nu))
        current = previous

    return 0.5 * real_trace(product_matrix)


def average_rectangular_wilson_loop(
    lattice: Lattice,
    mu: int,
    nu: int,
    width: int,
    height: int,
) -> float:
    """
    Average a rectangular Wilson loop over all lattice starting sites.
    """
    total = 0.0
    count = 0

    for site in lattice.sites():
        total += rectangular_wilson_loop(
            lattice=lattice,
            site=site,
            mu=mu,
            nu=nu,
            width=width,
            height=height,
        )
        count += 1

    return total / count


def wilson_loop_table(
    lattice: Lattice,
    mu: int,
    nu: int,
    max_width: int,
    max_height: int,
) -> list[dict[str, float]]:
    """
    Compute a table of average rectangular Wilson loops.

    Returns a list of dictionaries with:
    - width
    - height
    - area
    - perimeter
    - value
    """
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Maximum width and height must be positive.")

    rows: list[dict[str, float]] = []

    for width in range(1, max_width + 1):
        for height in range(1, max_height + 1):
            value = average_rectangular_wilson_loop(
                lattice=lattice,
                mu=mu,
                nu=nu,
                width=width,
                height=height,
            )

            rows.append(
                {
                    "width": float(width),
                    "height": float(height),
                    "area": float(width * height),
                    "perimeter": float(2 * (width + height)),
                    "value": float(value),
                }
            )

    return rows
