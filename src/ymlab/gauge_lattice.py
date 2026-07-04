"""
Generic matrix-gauge lattice container.

The original Lattice class is an SU(2)-specific implementation.

GaugeLattice provides a group-aware lattice whose link-matrix dimension and
membership rules are supplied by a MatrixGaugeGroup backend.

The class preserves:

1. periodic hypercubic geometry,
2. one matrix link per site and positive lattice direction,
3. cold and random starts,
4. explicit group-membership validation,
5. deterministic random-number generation from an optional seed.

The implementation favors mathematical transparency over maximum performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np

from ymlab.group_interface import MatrixGaugeGroup


Site = tuple[int, ...]


@dataclass
class GaugeLattice:
    """Periodic hypercubic lattice carrying a matrix gauge-group backend."""

    shape: tuple[int, ...]
    group: MatrixGaugeGroup
    cold_start: bool = True
    seed: int | None = None

    def __post_init__(
        self,
    ) -> None:
        self.shape = tuple(
            int(length)
            for length in self.shape
        )

        if len(self.shape) < 2:
            raise ValueError(
                "GaugeLattice requires at least two dimensions."
            )

        if any(
            length <= 0
            for length in self.shape
        ):
            raise ValueError(
                "All lattice extents must be positive."
            )

        if not isinstance(
            self.group,
            MatrixGaugeGroup,
        ):
            raise TypeError(
                "group must be a MatrixGaugeGroup."
            )

        self.dim = len(
            self.shape
        )

        self.rng = np.random.default_rng(
            self.seed
        )

        self.links: dict[
            tuple[Site, int],
            np.ndarray,
        ] = {}

        for site in self.sites():
            for direction in range(
                self.dim
            ):
                matrix = (
                    self.group.identity()
                    if self.cold_start
                    else self.group.random(
                        self.rng
                    )
                )

                self.set_link(
                    site,
                    direction,
                    matrix,
                )

    def sites(
        self,
    ) -> Iterator[Site]:
        """Iterate over every lattice site."""
        return product(
            *[
                range(length)
                for length in self.shape
            ]
        )

    def validate_site(
        self,
        site: Site,
    ) -> Site:
        """Validate and normalize a lattice site."""
        site = tuple(
            int(coordinate)
            for coordinate in site
        )

        if len(site) != self.dim:
            raise ValueError(
                "Site dimension does not match lattice dimension."
            )

        for coordinate, extent in zip(
            site,
            self.shape,
        ):
            if (
                coordinate < 0
                or coordinate >= extent
            ):
                raise ValueError(
                    "Site coordinate is outside the lattice."
                )

        return site

    def validate_direction(
        self,
        direction: int,
    ) -> int:
        """Validate one positive lattice direction."""
        direction = int(
            direction
        )

        if (
            direction < 0
            or direction >= self.dim
        ):
            raise ValueError(
                "Invalid lattice direction."
            )

        return direction

    def shift(
        self,
        site: Site,
        direction: int,
        amount: int,
    ) -> Site:
        """Shift a site periodically along one lattice direction."""
        site = self.validate_site(
            site
        )

        direction = self.validate_direction(
            direction
        )

        shifted = list(
            site
        )

        shifted[
            direction
        ] = (
            shifted[
                direction
            ]
            + int(
                amount
            )
        ) % self.shape[
            direction
        ]

        return tuple(
            shifted
        )

    def get_link(
        self,
        site: Site,
        direction: int,
    ) -> np.ndarray:
        """Return one stored gauge link."""
        site = self.validate_site(
            site
        )

        direction = self.validate_direction(
            direction
        )

        return self.links[
            (
                site,
                direction,
            )
        ]

    def set_link(
        self,
        site: Site,
        direction: int,
        value: np.ndarray,
    ) -> None:
        """Store one link after strict group-membership validation."""
        site = self.validate_site(
            site
        )

        direction = self.validate_direction(
            direction
        )

        value = np.asarray(
            value,
            dtype=complex,
        )

        expected_shape = (
            self.group.dimension,
            self.group.dimension,
        )

        if value.shape != expected_shape:
            raise ValueError(
                "Link matrix shape does not match gauge-group dimension."
            )

        if not self.group.is_member(
            value
        ):
            raise ValueError(
                f"Link variable must belong to {self.group.name}."
            )

        self.links[
            (
                site,
                direction,
            )
        ] = np.array(
            value,
            dtype=complex,
            copy=True,
        )

    def number_of_sites(
        self,
    ) -> int:
        """Return the number of lattice sites."""
        return int(
            np.prod(
                self.shape,
                dtype=int,
            )
        )

    def number_of_links(
        self,
    ) -> int:
        """Return the number of positively oriented lattice links."""
        return (
            self.number_of_sites()
            * self.dim
        )

    def copy(
        self,
    ) -> "GaugeLattice":
        """Return an independent copy of the gauge configuration."""
        copied = GaugeLattice(
            shape=self.shape,
            group=self.group,
            cold_start=True,
            seed=self.seed,
        )

        for site in self.sites():
            for direction in range(
                self.dim
            ):
                copied.set_link(
                    site,
                    direction,
                    self.get_link(
                        site,
                        direction,
                    ),
                )

        return copied
