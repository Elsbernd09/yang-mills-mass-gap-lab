"""
Periodic hypercubic lattice for SU(2) lattice gauge theory.

A lattice site is represented by an integer coordinate tuple, for example:

    (x, y)

for a 2D lattice, or

    (x, y, z, t)

for a 4D lattice.

Each directed link from a site in a coordinate direction stores an SU(2) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np

from ymlab.su2 import identity, random_su2, is_su2


Site = tuple[int, ...]


@dataclass
class Lattice:
    """
    Periodic hypercubic lattice.

    Parameters
    ----------
    shape:
        Tuple giving the number of sites in each dimension.
        Example: (8, 8) for 2D, (4, 4, 4, 4) for 4D.
    cold_start:
        If True, initialize every link to the identity.
        If False, initialize every link randomly in SU(2).
    seed:
        Random seed for reproducible random starts.
    """

    shape: tuple[int, ...]
    cold_start: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if len(self.shape) < 2:
            raise ValueError("Lattice dimension must be at least 2.")

        if any(n <= 0 for n in self.shape):
            raise ValueError("All lattice dimensions must be positive.")

        self.dim = len(self.shape)
        self.rng = np.random.default_rng(self.seed)

        self.links: dict[tuple[Site, int], np.ndarray] = {}

        for site in self.sites():
            for mu in range(self.dim):
                if self.cold_start:
                    self.links[(site, mu)] = identity()
                else:
                    self.links[(site, mu)] = random_su2(self.rng)

    def sites(self) -> Iterator[Site]:
        """Iterate over all lattice sites."""
        return product(*[range(n) for n in self.shape])

    def shift(self, site: Site, direction: int, amount: int = 1) -> Site:
        """
        Shift a site periodically in a coordinate direction.

        Example:
        On a shape (4, 4) lattice, shifting (3, 1) by +1 in direction 0 gives (0, 1).
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError("Invalid lattice direction.")

        shifted = list(site)
        shifted[direction] = (shifted[direction] + amount) % self.shape[direction]
        return tuple(shifted)

    def get_link(self, site: Site, direction: int) -> np.ndarray:
        """Get the SU(2) link variable U_mu(x)."""
        return self.links[(site, direction)]

    def set_link(self, site: Site, direction: int, value: np.ndarray) -> None:
        """Set the SU(2) link variable U_mu(x)."""
        if not is_su2(value):
            raise ValueError("Link variable must be an SU(2) matrix.")

        self.links[(site, direction)] = value

    def number_of_sites(self) -> int:
        """Return the total number of lattice sites."""
        total = 1
        for n in self.shape:
            total *= n
        return total

    def number_of_links(self) -> int:
        """Return the total number of directed positive links."""
        return self.number_of_sites() * self.dim
