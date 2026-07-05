"""
Local-versus-global Wilson-action consistency validation for GaugeLattice.

For one link replacement U -> U', the local staple-based action difference
must equal the change obtained by recomputing the complete Wilson action:

    Delta S_local
        =
        Delta S_global.

This identity is tested for generic matrix-group lattices, with special focus
on SU(3).

A disagreement indicates a problem in at least one of:

1. plaquette orientation,
2. staple orientation,
3. trace normalization,
4. local-action normalization,
5. proposal handling.

The original lattice link is always restored after the global comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.gauge_lattice import (
    GaugeLattice,
    Site,
)
from ymlab.generic_gauge import (
    generic_local_action_difference,
    generic_wilson_action,
)


@dataclass(frozen=True)
class GenericActionDifferenceComparison:
    """Local-versus-global action-difference result."""

    local_difference: float
    global_difference: float
    absolute_error: float
    relative_error: float
    consistent: bool


def generic_global_action_difference(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
) -> float:
    """
    Compute a full Wilson-action difference and restore the original link.
    """
    proposal = np.asarray(
        proposal,
        dtype=complex,
    )

    if not lattice.group.is_member(
        proposal
    ):
        raise ValueError(
            "Proposal is not a member of the active gauge group."
        )

    old_link = np.array(
        lattice.get_link(
            site,
            mu,
        ),
        dtype=complex,
        copy=True,
    )

    action_before = generic_wilson_action(
        lattice=lattice,
        beta=beta,
    )

    try:
        lattice.set_link(
            site,
            mu,
            proposal,
        )

        action_after = generic_wilson_action(
            lattice=lattice,
            beta=beta,
        )
    finally:
        lattice.set_link(
            site,
            mu,
            old_link,
        )

    return float(
        action_after
        - action_before
    )


def compare_generic_action_differences(
    lattice: GaugeLattice,
    site: Site,
    mu: int,
    proposal: np.ndarray,
    beta: float,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> GenericActionDifferenceComparison:
    """
    Compare local and global Wilson-action differences.
    """
    local_difference = (
        generic_local_action_difference(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=beta,
        )
    )

    global_difference = (
        generic_global_action_difference(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=beta,
        )
    )

    absolute_error = abs(
        local_difference
        - global_difference
    )

    scale = max(
        abs(
            local_difference
        ),
        abs(
            global_difference
        ),
        np.finfo(float).eps,
    )

    relative_error = (
        absolute_error
        / scale
    )

    consistent = bool(
        np.isclose(
            local_difference,
            global_difference,
            atol=atol,
            rtol=rtol,
        )
    )

    return GenericActionDifferenceComparison(
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
