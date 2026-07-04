"""
Correlation-aware resampling utilities for lattice Monte Carlo data.

Markov-chain measurements are generally autocorrelated. Resampling individual
measurements as though they were independent can underestimate uncertainty.

This module implements:

1. Naive bootstrap mean estimation for comparison.
2. Circular moving-block bootstrap.
3. Delete-one jackknife.
4. Delete-one-block jackknife.
5. Block-size sensitivity analysis.

The purpose is numerical uncertainty analysis, not a proof of the Yang-Mills
mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResamplingEstimate:
    """Summary of a scalar resampling estimate."""

    method: str
    estimate: float
    standard_error: float
    lower_ci: float
    upper_ci: float
    resampled_statistics: np.ndarray


@dataclass(frozen=True)
class BlockSensitivityPoint:
    """Uncertainty estimate associated with one block size."""

    block_size: int
    number_of_blocks: int
    standard_error: float
    lower_ci: float
    upper_ci: float


def _validate_values(values: np.ndarray) -> np.ndarray:
    """Validate and return a finite one-dimensional floating-point array."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    if len(values) < 2:
        raise ValueError("Need at least two values.")

    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite.")

    return values


def _quantile_interval(
    samples: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Return a central quantile confidence interval."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1.0 - confidence_level

    return (
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def naive_bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> ResamplingEstimate:
    """
    Bootstrap individual measurements with replacement.

    This method is included mainly as a comparison baseline. For autocorrelated
    Markov-chain data, block-based methods are generally more appropriate.
    """
    values = _validate_values(values)

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    rng = np.random.default_rng(seed)
    n = len(values)

    statistics = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        statistics[i] = float(np.mean(sample))

    lower, upper = _quantile_interval(
        statistics,
        confidence_level,
    )

    return ResamplingEstimate(
        method="naive_bootstrap",
        estimate=float(np.mean(values)),
        standard_error=float(np.std(statistics, ddof=1)),
        lower_ci=lower,
        upper_ci=upper,
        resampled_statistics=statistics,
    )


def circular_block_sample(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one circular moving-block bootstrap sample.

    Blocks preserve short-range temporal ordering. A block may wrap around the
    end of the time series, giving a circular bootstrap.

    The concatenated result is truncated to the original sample length.
    """
    values = _validate_values(values)
    n = len(values)

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if block_size > n:
        raise ValueError("block_size cannot exceed the sample length.")

    number_of_blocks = int(np.ceil(n / block_size))
    blocks = []

    for _ in range(number_of_blocks):
        start = int(rng.integers(0, n))
        indices = (start + np.arange(block_size)) % n
        blocks.append(values[indices])

    return np.concatenate(blocks)[:n]


def moving_block_bootstrap_mean(
    values: np.ndarray,
    block_size: int,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> ResamplingEstimate:
    """
    Estimate the mean and uncertainty using a circular moving-block bootstrap.

    Contiguous blocks are resampled rather than individual measurements so that
    local autocorrelation structure is partially preserved.
    """
    values = _validate_values(values)

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if block_size <= 0 or block_size > len(values):
        raise ValueError(
            "block_size must satisfy 1 <= block_size <= len(values)."
        )

    rng = np.random.default_rng(seed)

    statistics = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = circular_block_sample(
            values=values,
            block_size=block_size,
            rng=rng,
        )
        statistics[i] = float(np.mean(sample))

    lower, upper = _quantile_interval(
        statistics,
        confidence_level,
    )

    return ResamplingEstimate(
        method=f"moving_block_bootstrap_b{block_size}",
        estimate=float(np.mean(values)),
        standard_error=float(np.std(statistics, ddof=1)),
        lower_ci=lower,
        upper_ci=upper,
        resampled_statistics=statistics,
    )


def delete_one_jackknife_mean(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> ResamplingEstimate:
    """
    Estimate mean uncertainty with an ordinary delete-one jackknife.
    """
    values = _validate_values(values)
    n = len(values)

    statistics = np.empty(n, dtype=float)

    for i in range(n):
        sample = np.delete(values, i)
        statistics[i] = float(np.mean(sample))

    jackknife_mean = float(np.mean(statistics))

    standard_error = float(
        np.sqrt(
            ((n - 1) / n)
            * np.sum((statistics - jackknife_mean) ** 2)
        )
    )

    # Normal approximation is used for the jackknife interval.
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    from scipy.stats import norm

    z_value = float(
        norm.ppf(0.5 + confidence_level / 2.0)
    )

    estimate = float(np.mean(values))
    lower = estimate - z_value * standard_error
    upper = estimate + z_value * standard_error

    return ResamplingEstimate(
        method="delete_one_jackknife",
        estimate=estimate,
        standard_error=standard_error,
        lower_ci=float(lower),
        upper_ci=float(upper),
        resampled_statistics=statistics,
    )


def delete_one_block_jackknife_mean(
    values: np.ndarray,
    block_size: int,
    confidence_level: float = 0.95,
) -> ResamplingEstimate:
    """
    Estimate uncertainty by deleting one non-overlapping block at a time.

    Only complete blocks are used. Any incomplete tail is discarded.
    """
    values = _validate_values(values)
    n = len(values)

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    number_of_blocks = n // block_size

    if number_of_blocks < 2:
        raise ValueError(
            "Need at least two complete blocks for block jackknife."
        )

    usable_length = number_of_blocks * block_size
    trimmed = values[:usable_length]

    statistics = np.empty(number_of_blocks, dtype=float)

    for block_index in range(number_of_blocks):
        start = block_index * block_size
        stop = start + block_size

        sample = np.concatenate(
            [
                trimmed[:start],
                trimmed[stop:],
            ]
        )

        statistics[block_index] = float(np.mean(sample))

    jackknife_mean = float(np.mean(statistics))

    standard_error = float(
        np.sqrt(
            ((number_of_blocks - 1) / number_of_blocks)
            * np.sum((statistics - jackknife_mean) ** 2)
        )
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    from scipy.stats import norm

    z_value = float(
        norm.ppf(0.5 + confidence_level / 2.0)
    )

    estimate = float(np.mean(trimmed))
    lower = estimate - z_value * standard_error
    upper = estimate + z_value * standard_error

    return ResamplingEstimate(
        method=f"delete_one_block_jackknife_b{block_size}",
        estimate=estimate,
        standard_error=standard_error,
        lower_ci=float(lower),
        upper_ci=float(upper),
        resampled_statistics=statistics,
    )


def block_size_sensitivity(
    values: np.ndarray,
    block_sizes: list[int],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> list[BlockSensitivityPoint]:
    """
    Run a moving-block bootstrap across several block sizes.

    This helps diagnose whether uncertainty estimates are sensitive to the
    correlation scale retained by the resampling procedure.
    """
    values = _validate_values(values)

    if len(block_sizes) == 0:
        raise ValueError("At least one block size is required.")

    points: list[BlockSensitivityPoint] = []

    for i, block_size in enumerate(block_sizes):
        if block_size <= 0 or block_size > len(values):
            raise ValueError(
                "Each block size must satisfy "
                "1 <= block_size <= len(values)."
            )

        result = moving_block_bootstrap_mean(
            values=values,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=None if seed is None else seed + i,
        )

        points.append(
            BlockSensitivityPoint(
                block_size=block_size,
                number_of_blocks=int(
                    np.ceil(len(values) / block_size)
                ),
                standard_error=result.standard_error,
                lower_ci=result.lower_ci,
                upper_ci=result.upper_ci,
            )
        )

    return points
