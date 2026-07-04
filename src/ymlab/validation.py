"""
Wilson-action consistency validation utilities.

The strongest local-update consistency check compares

    local staple action difference

against

    full Wilson action after proposal
    minus
    full Wilson action before proposal.

For a correctly oriented staple implementation, these quantities should agree
up to floating-point roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.staples import local_action_difference
from ymlab.su2 import is_su2
from ymlab.wilson_action import wilson_action


@dataclass(frozen=True)
class ActionDifferenceComparison:
    """Comparison of local and global Wilson-action differences."""

    local_difference: float
    global_difference: float
    absolute_error: float
    relative_error: float
    consistent: bool


def global_action_difference(
    lattice: Lattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
) -> float:
    """
    Compute the exact full Wilson-action difference for one proposed link.

    The original lattice link is restored before returning.
    """
    if not is_su2(
        proposal,
        atol=1e-8,
    ):
        raise ValueError(
            "proposal must be an SU(2) matrix."
        )

    old_link = np.array(
        lattice.get_link(
            site,
            mu,
        ),
        dtype=complex,
        copy=True,
    )

    action_before = wilson_action(
        lattice,
        beta=beta,
    )

    try:
        lattice.set_link(
            site,
            mu,
            proposal,
        )

        action_after = wilson_action(
            lattice,
            beta=beta,
        )

    finally:
        lattice.set_link(
            site,
            mu,
            old_link,
        )

    return float(
        action_after - action_before
    )


def compare_action_differences(
    lattice: Lattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> ActionDifferenceComparison:
    """
    Compare local staple and full global Wilson-action differences.
    """
    local_difference = local_action_difference(
        lattice=lattice,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=beta,
    )

    global_difference = global_action_difference(
        lattice=lattice,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=beta,
    )

    absolute_error = abs(
        local_difference
        - global_difference
    )

    scale = max(
        abs(local_difference),
        abs(global_difference),
        np.finfo(float).eps,
    )

    relative_error = (
        absolute_error / scale
    )

    consistent = bool(
        np.isclose(
            local_difference,
            global_difference,
            atol=atol,
            rtol=rtol,
        )
    )

    return ActionDifferenceComparison(
        local_difference=float(
            local_difference
        ),
        global_difference=float(
            global_difference
        ),
        absolute_error=float(
            absolute_error
        ),
        relative_error=float(
            relative_error
        ),
        consistent=consistent,
    )
