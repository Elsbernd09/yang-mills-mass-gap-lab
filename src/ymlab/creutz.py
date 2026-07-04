"""
Creutz ratios and string-tension-style diagnostics.

Wilson loops are central observables in lattice gauge theory. A rectangular
Wilson loop W(R,T) can decay approximately like

    W(R,T) ~ exp(-sigma R T - perimeter terms)

where sigma is a string-tension-like quantity.

Creutz ratios are designed to cancel leading perimeter contributions:

    chi(R,T) =
        -log( W(R,T) W(R-1,T-1) / (W(R,T-1) W(R-1,T)) )

For sufficiently large loops, chi(R,T) can be interpreted as a rough
finite-lattice string-tension-style diagnostic.

This is a numerical diagnostic, not a proof of confinement or the Yang-Mills
mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice
from ymlab.observables import average_rectangular_wilson_loop


@dataclass
class CreutzRatio:
    """Container for a single Creutz ratio measurement."""

    width: int
    height: int
    value: float
    valid: bool
    reason: str = ""


def creutz_ratio_from_values(
    w_rt: float,
    w_r1_t1: float,
    w_rt1: float,
    w_r1_t: float,
    width: int,
    height: int,
    eps: float = 1e-12,
) -> CreutzRatio:
    """
    Compute a Creutz ratio from four Wilson loop values.

    Parameters
    ----------
    w_rt:
        W(R,T)
    w_r1_t1:
        W(R-1,T-1)
    w_rt1:
        W(R,T-1)
    w_r1_t:
        W(R-1,T)
    width:
        R value.
    height:
        T value.
    eps:
        Small positivity threshold.

    Returns
    -------
    CreutzRatio
        Creutz ratio result with validity flag.
    """
    if width <= 1 or height <= 1:
        return CreutzRatio(
            width=width,
            height=height,
            value=float("nan"),
            valid=False,
            reason="Creutz ratios require width and height greater than 1.",
        )

    values = [w_rt, w_r1_t1, w_rt1, w_r1_t]

    if not all(np.isfinite(v) for v in values):
        return CreutzRatio(
            width=width,
            height=height,
            value=float("nan"),
            valid=False,
            reason="Wilson loop values must be finite.",
        )

    numerator = w_rt * w_r1_t1
    denominator = w_rt1 * w_r1_t

    if numerator <= eps or denominator <= eps:
        return CreutzRatio(
            width=width,
            height=height,
            value=float("nan"),
            valid=False,
            reason="Creutz ratio requires positive numerator and denominator.",
        )

    ratio = numerator / denominator

    if ratio <= eps:
        return CreutzRatio(
            width=width,
            height=height,
            value=float("nan"),
            valid=False,
            reason="Logarithm argument must be positive.",
        )

    return CreutzRatio(
        width=width,
        height=height,
        value=float(-np.log(ratio)),
        valid=True,
        reason="",
    )


def creutz_ratio(
    lattice: Lattice,
    mu: int,
    nu: int,
    width: int,
    height: int,
) -> CreutzRatio:
    """
    Compute a Creutz ratio from lattice Wilson loops.

    Uses averaged rectangular Wilson loops over all starting sites.
    """
    if width <= 1 or height <= 1:
        return CreutzRatio(
            width=width,
            height=height,
            value=float("nan"),
            valid=False,
            reason="Creutz ratios require width and height greater than 1.",
        )

    w_rt = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=mu,
        nu=nu,
        width=width,
        height=height,
    )

    w_r1_t1 = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=mu,
        nu=nu,
        width=width - 1,
        height=height - 1,
    )

    w_rt1 = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=mu,
        nu=nu,
        width=width,
        height=height - 1,
    )

    w_r1_t = average_rectangular_wilson_loop(
        lattice=lattice,
        mu=mu,
        nu=nu,
        width=width - 1,
        height=height,
    )

    return creutz_ratio_from_values(
        w_rt=w_rt,
        w_r1_t1=w_r1_t1,
        w_rt1=w_rt1,
        w_r1_t=w_r1_t,
        width=width,
        height=height,
    )


def creutz_ratio_table(
    lattice: Lattice,
    mu: int,
    nu: int,
    max_width: int,
    max_height: int,
) -> list[CreutzRatio]:
    """
    Compute Creutz ratios for 2 <= R <= max_width and 2 <= T <= max_height.
    """
    if max_width <= 1 or max_height <= 1:
        raise ValueError("max_width and max_height must both be greater than 1.")

    results: list[CreutzRatio] = []

    for width in range(2, max_width + 1):
        for height in range(2, max_height + 1):
            results.append(
                creutz_ratio(
                    lattice=lattice,
                    mu=mu,
                    nu=nu,
                    width=width,
                    height=height,
                )
            )

    return results


def valid_creutz_values(results: list[CreutzRatio]) -> np.ndarray:
    """
    Extract valid Creutz ratio values as a NumPy array.
    """
    return np.asarray([result.value for result in results if result.valid], dtype=float)


def estimate_string_tension(results: list[CreutzRatio]) -> float:
    """
    Estimate a string-tension-like quantity from valid Creutz ratios.

    This simple estimator returns the mean of valid Creutz ratios.
    """
    values = valid_creutz_values(results)

    if len(values) == 0:
        raise ValueError("No valid Creutz ratios available.")

    return float(np.mean(values))


def creutz_results_as_rows(results: list[CreutzRatio]) -> list[dict[str, float | int | bool | str]]:
    """
    Convert Creutz ratio results into table rows.
    """
    rows: list[dict[str, float | int | bool | str]] = []

    for result in results:
        rows.append(
            {
                "width": result.width,
                "height": result.height,
                "area": result.width * result.height,
                "value": result.value,
                "valid": result.valid,
                "reason": result.reason,
            }
        )

    return rows
