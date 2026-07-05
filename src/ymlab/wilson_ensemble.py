"""
Configuration-level Wilson-loop ensembles and nonlinear Creutz-ratio
resampling.

The module measures a rectangular Wilson-loop basis on every configuration of
a Markov-chain ensemble.

A complete configuration measurement is stored as one row:

    configuration
        by
    Wilson-loop observable.

Circular moving-block bootstrap resamples complete configuration rows.

For each bootstrap replicate:

1. configuration blocks are sampled with replacement,
2. every Wilson loop is averaged over the resampled configurations,
3. Creutz ratios are reconstructed from the bootstrap loop means,
4. invalid nonlinear ratios remain invalid rather than being forced positive.

This preserves cross-loop correlations during nonlinear uncertainty
propagation.

The implementation provides finite-lattice string-tension-style diagnostics.
It does not establish a continuum string tension or the Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ymlab.creutz import (
    creutz_ratio_from_values,
)
from ymlab.gevp_bootstrap import (
    circular_block_bootstrap_indices,
)
from ymlab.lattice import Lattice
from ymlab.observables import (
    average_rectangular_wilson_loop,
)


LoopShape = tuple[int, int]


@dataclass(frozen=True)
class WilsonLoopBasis:
    """Ordered rectangular Wilson-loop measurement basis."""

    shapes: tuple[LoopShape, ...]

    def __post_init__(
        self,
    ) -> None:
        if len(
            self.shapes
        ) == 0:
            raise ValueError(
                "Wilson-loop basis cannot be empty."
            )

        normalized = []

        for width, height in self.shapes:
            width = int(
                width
            )

            height = int(
                height
            )

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Wilson-loop dimensions must be positive."
                )

            normalized.append(
                (
                    width,
                    height,
                )
            )

        if len(
            set(
                normalized
            )
        ) != len(
            normalized
        ):
            raise ValueError(
                "Wilson-loop basis contains duplicate shapes."
            )

        object.__setattr__(
            self,
            "shapes",
            tuple(
                normalized
            ),
        )

    @property
    def labels(
        self,
    ) -> tuple[str, ...]:
        """Return deterministic observable labels."""
        return tuple(
            f"W{width}{height}"
            for width, height in self.shapes
        )

    def index(
        self,
        shape: LoopShape,
    ) -> int:
        """Return the column index of one loop shape."""
        normalized = (
            int(
                shape[0]
            ),
            int(
                shape[1]
            ),
        )

        try:
            return self.shapes.index(
                normalized
            )
        except ValueError as error:
            raise ValueError(
                f"Loop shape {normalized} is not in the basis."
            ) from error


@dataclass(frozen=True)
class CreutzBootstrapPoint:
    """Bootstrap summary for one Creutz ratio."""

    width: int
    height: int
    central_value: float
    bootstrap_mean: float
    standard_error: float
    lower_95: float
    median: float
    upper_95: float
    finite_replicates: int
    requested_replicates: int
    valid_fraction: float


@dataclass(frozen=True)
class EnsembleCreutzBootstrapResult:
    """Complete loop and Creutz bootstrap output."""

    basis: WilsonLoopBasis
    central_loop_means: np.ndarray
    loop_mean_samples: np.ndarray
    creutz_shapes: tuple[LoopShape, ...]
    creutz_samples: np.ndarray
    creutz_points: tuple[
        CreutzBootstrapPoint,
        ...
    ]
    bootstrap_replicates: int
    block_size: int


def create_rectangular_loop_basis(
    maximum_width: int,
    maximum_height: int,
) -> WilsonLoopBasis:
    """Create all rectangular loops from 1x1 through the requested maxima."""
    if maximum_width < 1:
        raise ValueError(
            "maximum_width must be positive."
        )

    if maximum_height < 1:
        raise ValueError(
            "maximum_height must be positive."
        )

    shapes = tuple(
        (
            width,
            height,
        )
        for width in range(
            1,
            maximum_width + 1,
        )
        for height in range(
            1,
            maximum_height + 1,
        )
    )

    return WilsonLoopBasis(
        shapes=shapes
    )


def required_creutz_shapes(
    basis: WilsonLoopBasis,
) -> tuple[LoopShape, ...]:
    """
    Return every Creutz shape whose four required loops exist in the basis.
    """
    available = set(
        basis.shapes
    )

    shapes = []

    for width, height in basis.shapes:
        if width < 2 or height < 2:
            continue

        required = {
            (
                width,
                height,
            ),
            (
                width - 1,
                height - 1,
            ),
            (
                width,
                height - 1,
            ),
            (
                width - 1,
                height,
            ),
        }

        if required.issubset(
            available
        ):
            shapes.append(
                (
                    width,
                    height,
                )
            )

    return tuple(
        shapes
    )


def measure_wilson_loop_basis(
    lattice: Lattice,
    basis: WilsonLoopBasis,
    mu: int = 0,
    nu: int = 1,
) -> np.ndarray:
    """
    Measure every basis loop as a lattice-position averaged Wilson loop.
    """
    if mu == nu:
        raise ValueError(
            "Wilson-loop directions must be distinct."
        )

    if (
        mu < 0
        or mu >= lattice.dim
        or nu < 0
        or nu >= lattice.dim
    ):
        raise ValueError(
            "Invalid Wilson-loop direction."
        )

    values = []

    for width, height in basis.shapes:
        values.append(
            average_rectangular_wilson_loop(
                lattice=lattice,
                mu=mu,
                nu=nu,
                width=width,
                height=height,
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def validate_loop_ensemble(
    loop_ensemble: np.ndarray,
    basis: WilsonLoopBasis,
) -> np.ndarray:
    """
    Validate configuration-by-loop measurements.
    """
    ensemble = np.asarray(
        loop_ensemble,
        dtype=float,
    )

    if ensemble.ndim != 2:
        raise ValueError(
            "Expected loop ensemble shape "
            "(configurations, loop_observables)."
        )

    if ensemble.shape[0] < 2:
        raise ValueError(
            "Need at least two measured configurations."
        )

    if ensemble.shape[1] != len(
        basis.shapes
    ):
        raise ValueError(
            "Loop ensemble column count does not match basis."
        )

    if not np.all(
        np.isfinite(
            ensemble
        )
    ):
        raise ValueError(
            "Loop ensemble must contain finite values."
        )

    return ensemble


def creutz_ratios_from_loop_means(
    loop_means: np.ndarray,
    basis: WilsonLoopBasis,
    creutz_shapes: Iterable[
        LoopShape
    ] | None = None,
) -> np.ndarray:
    """
    Reconstruct Creutz ratios from one vector of mean Wilson-loop values.

    Invalid logarithmic ratios are represented by NaN.
    """
    loop_means = np.asarray(
        loop_means,
        dtype=float,
    )

    if loop_means.shape != (
        len(
            basis.shapes
        ),
    ):
        raise ValueError(
            "loop_means shape does not match Wilson-loop basis."
        )

    shapes = (
        required_creutz_shapes(
            basis
        )
        if creutz_shapes is None
        else tuple(
            (
                int(
                    width
                ),
                int(
                    height
                ),
            )
            for width, height in creutz_shapes
        )
    )

    values = []

    for width, height in shapes:
        try:
            w_rt = loop_means[
                basis.index(
                    (
                        width,
                        height,
                    )
                )
            ]

            w_rm1_tm1 = loop_means[
                basis.index(
                    (
                        width - 1,
                        height - 1,
                    )
                )
            ]

            w_r_tm1 = loop_means[
                basis.index(
                    (
                        width,
                        height - 1,
                    )
                )
            ]

            w_rm1_t = loop_means[
                basis.index(
                    (
                        width - 1,
                        height,
                    )
                )
            ]
        except ValueError as error:
            raise ValueError(
                "Creutz shape requires Wilson loops "
                "that are absent from the basis."
            ) from error

        result = creutz_ratio_from_values(
            w_rt=w_rt,
            w_r1_t1=w_rm1_tm1,
            w_rt1=w_r_tm1,
            w_r1_t=w_rm1_t,
            width=width,
            height=height,
        )

        values.append(
            result.value
            if result.valid
            else np.nan
        )

    return np.asarray(
        values,
        dtype=float,
    )


def block_bootstrap_creutz_ratios(
    loop_ensemble: np.ndarray,
    basis: WilsonLoopBasis,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    seed: int | None = None,
) -> EnsembleCreutzBootstrapResult:
    """
    Circular block-bootstrap complete configuration-level Wilson-loop rows.

    Creutz ratios are recomputed from the bootstrap loop means inside every
    replicate.
    """
    ensemble = validate_loop_ensemble(
        loop_ensemble=loop_ensemble,
        basis=basis,
    )

    if n_bootstrap <= 1:
        raise ValueError(
            "n_bootstrap must be greater than one."
        )

    if block_size < 1:
        raise ValueError(
            "block_size must be positive."
        )

    if block_size > ensemble.shape[0]:
        raise ValueError(
            "block_size cannot exceed the number of configurations."
        )

    creutz_shapes = required_creutz_shapes(
        basis
    )

    if len(
        creutz_shapes
    ) == 0:
        raise ValueError(
            "Wilson-loop basis supports no Creutz ratios."
        )

    central_loop_means = np.mean(
        ensemble,
        axis=0,
    )

    central_creutz = (
        creutz_ratios_from_loop_means(
            loop_means=central_loop_means,
            basis=basis,
            creutz_shapes=creutz_shapes,
        )
    )

    rng = np.random.default_rng(
        seed
    )

    loop_mean_samples = []
    creutz_samples = []

    for _ in range(
        n_bootstrap
    ):
        indices = circular_block_bootstrap_indices(
            number_of_configurations=(
                ensemble.shape[0]
            ),
            block_size=block_size,
            rng=rng,
        )

        resampled_means = np.mean(
            ensemble[
                indices
            ],
            axis=0,
        )

        loop_mean_samples.append(
            resampled_means
        )

        creutz_samples.append(
            creutz_ratios_from_loop_means(
                loop_means=resampled_means,
                basis=basis,
                creutz_shapes=creutz_shapes,
            )
        )

    loop_mean_samples = np.asarray(
        loop_mean_samples,
        dtype=float,
    )

    creutz_samples = np.asarray(
        creutz_samples,
        dtype=float,
    )

    points = []

    for index, (
        width,
        height,
    ) in enumerate(
        creutz_shapes
    ):
        samples = creutz_samples[
            :,
            index,
        ]

        finite = samples[
            np.isfinite(
                samples
            )
        ]

        finite_count = len(
            finite
        )

        valid_fraction = (
            finite_count
            / n_bootstrap
        )

        if finite_count == 0:
            mean = np.nan
            standard_error = np.nan
            lower = np.nan
            median = np.nan
            upper = np.nan
        else:
            mean = float(
                np.mean(
                    finite
                )
            )

            standard_error = (
                float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                )
                if finite_count > 1
                else np.nan
            )

            lower = float(
                np.quantile(
                    finite,
                    0.025,
                )
            )

            median = float(
                np.quantile(
                    finite,
                    0.5,
                )
            )

            upper = float(
                np.quantile(
                    finite,
                    0.975,
                )
            )

        points.append(
            CreutzBootstrapPoint(
                width=width,
                height=height,
                central_value=float(
                    central_creutz[
                        index
                    ]
                ),
                bootstrap_mean=mean,
                standard_error=standard_error,
                lower_95=lower,
                median=median,
                upper_95=upper,
                finite_replicates=finite_count,
                requested_replicates=n_bootstrap,
                valid_fraction=float(
                    valid_fraction
                ),
            )
        )

    return EnsembleCreutzBootstrapResult(
        basis=basis,
        central_loop_means=np.asarray(
            central_loop_means,
            dtype=float,
        ),
        loop_mean_samples=loop_mean_samples,
        creutz_shapes=creutz_shapes,
        creutz_samples=creutz_samples,
        creutz_points=tuple(
            points
        ),
        bootstrap_replicates=n_bootstrap,
        block_size=block_size,
    )


def square_creutz_plateau_values(
    result: EnsembleCreutzBootstrapResult,
    minimum_valid_fraction: float = 0.80,
) -> np.ndarray:
    """
    Return bootstrap-mean square Creutz ratios passing a validity threshold.

    This is a string-tension-style plateau diagnostic, not a continuum string
    tension estimator.
    """
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError(
            "minimum_valid_fraction must lie between zero and one."
        )

    values = [
        point.bootstrap_mean
        for point in result.creutz_points
        if (
            point.width == point.height
            and point.valid_fraction >= minimum_valid_fraction
            and np.isfinite(
                point.bootstrap_mean
            )
        )
    ]

    return np.asarray(
        values,
        dtype=float,
    )
