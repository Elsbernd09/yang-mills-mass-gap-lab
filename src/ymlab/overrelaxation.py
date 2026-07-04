"""
SU(2) microcanonical overrelaxation.

For one link U_mu(x), the Wilson-action staple V is oriented so that

    U_mu(x) V

forms the sum of closed plaquette products containing the link.

A sum of SU(2) matrices retains the SU(2) quaternionic matrix form and can be
written, when nonzero, as

    V = k V_tilde,

with k > 0 and V_tilde in SU(2).

The local Wilson-action contribution is proportional to

    Re Tr[U V]
        =
        k Re Tr[U V_tilde].

Let

    X = U V_tilde.

The microcanonical reflection

    X -> X^dagger

preserves Re Tr[X].

Solving for the reflected link gives

    U'
        =
        V_tilde^dagger
        U^dagger
        V_tilde^dagger.

The resulting update preserves the local Wilson action to floating-point
precision while moving the Markov chain through configuration space.

Overrelaxation alone is not used as the complete sampling algorithm. The
hybrid update combines stochastic Metropolis sweeps with deterministic
microcanonical sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ymlab.lattice import Lattice, Site
from ymlab.monte_carlo import metropolis_sweep
from ymlab.staples import (
    local_link_action,
    staple,
)
from ymlab.su2 import (
    dagger,
    is_su2,
    reunitarize,
)


@dataclass(frozen=True)
class OverrelaxationLinkResult:
    """Diagnostic result for one reflected link."""

    site: Site
    direction: int
    local_action_before: float
    local_action_after: float
    absolute_action_error: float
    staple_scale: float


@dataclass(frozen=True)
class OverrelaxationSweepResult:
    """Diagnostic result for one full overrelaxation sweep."""

    updated_links: int
    skipped_links: int
    maximum_local_action_error: float
    mean_local_action_error: float


@dataclass(frozen=True)
class HybridSweepResult:
    """One stochastic-plus-microcanonical hybrid update."""

    metropolis_acceptance_rate: float
    overrelaxation_sweeps: int
    overrelaxation_updated_links: int
    overrelaxation_skipped_links: int
    maximum_local_action_error: float


def normalized_su2_staple(
    lattice: Lattice,
    site: Site,
    mu: int,
    minimum_scale: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """
    Factor the Wilson-action staple as

        V = scale * V_tilde,

    where V_tilde is numerically in SU(2).

    For a matrix with SU(2) quaternionic form,

        V^dagger V = scale^2 I.

    Therefore

        scale^2
            =
            (1 / 2) Re Tr(V^dagger V).

    This Frobenius/quaternion norm is used directly rather than extracting the
    scale from a floating-point determinant.
    """
    if minimum_scale <= 0.0:
        raise ValueError(
            "minimum_scale must be positive."
        )

    value = staple(
        lattice=lattice,
        site=site,
        mu=mu,
    )

    gram = (
        dagger(value)
        @ value
    )

    scale_squared = float(
        0.5
        * np.trace(
            gram
        ).real
    )

    if (
        not np.isfinite(
            scale_squared
        )
        or scale_squared <= 0.0
    ):
        raise ValueError(
            "Staple norm is not positive and finite."
        )

    scale = float(
        np.sqrt(
            scale_squared
        )
    )

    if scale <= minimum_scale:
        raise ValueError(
            "Staple scale is too small for stable normalization."
        )

    normalized = (
        value / scale
    )

    if not is_su2(
        normalized,
        atol=1e-8,
    ):
        quaternion_error = float(
            np.linalg.norm(
                dagger(normalized)
                @ normalized
                - np.eye(
                    2,
                    dtype=complex,
                ),
                ord="fro",
            )
        )

        determinant_error = float(
            abs(
                np.linalg.det(
                    normalized
                )
                - 1.0
            )
        )

        raise ValueError(
            "Normalized staple is not numerically in SU(2): "
            f"unitarity_error={quaternion_error:.12e}, "
            f"determinant_error={determinant_error:.12e}."
        )

    return (
        np.asarray(
            normalized,
            dtype=complex,
        ),
        scale,
    )


def overrelaxation_proposal(
    lattice: Lattice,
    site: Site,
    mu: int,
    minimum_scale: float = 1e-14,
) -> tuple[np.ndarray, float]:
    """
    Construct the exact SU(2) microcanonical reflection proposal.
    """
    normalized_staple, scale = (
        normalized_su2_staple(
            lattice=lattice,
            site=site,
            mu=mu,
            minimum_scale=minimum_scale,
        )
    )

    old_link = lattice.get_link(
        site,
        mu,
    )

    normalized_staple_dagger = dagger(
        normalized_staple
    )

    proposal = (
        normalized_staple_dagger
        @ dagger(
            old_link
        )
        @ normalized_staple_dagger
    )

    # Floating-point matrix products can introduce tiny numerical drift away
    # from the exact SU(2) manifold. Project the reflected matrix back to SU(2)
    # before storing it in the lattice.
    proposal = reunitarize(
        proposal
    )

    if not is_su2(
        proposal,
        atol=1e-10,
    ):
        raise ValueError(
            "Reunitarized overrelaxation proposal "
            "is not numerically in SU(2)."
        )

    return (
        np.asarray(
            proposal,
            dtype=complex,
        ),
        scale,
    )


def overrelaxation_link_update(
    lattice: Lattice,
    site: Site,
    mu: int,
    beta: float,
    minimum_scale: float = 1e-14,
) -> OverrelaxationLinkResult:
    """
    Reflect one link while preserving its local Wilson-action contribution.
    """
    if beta < 0.0:
        raise ValueError(
            "beta must be nonnegative."
        )

    local_before = local_link_action(
        lattice=lattice,
        site=site,
        mu=mu,
        beta=beta,
    )

    proposal, scale = (
        overrelaxation_proposal(
            lattice=lattice,
            site=site,
            mu=mu,
            minimum_scale=minimum_scale,
        )
    )

    lattice.set_link(
        site,
        mu,
        proposal,
    )

    local_after = local_link_action(
        lattice=lattice,
        site=site,
        mu=mu,
        beta=beta,
    )

    absolute_error = abs(
        local_after
        - local_before
    )

    return OverrelaxationLinkResult(
        site=site,
        direction=int(
            mu
        ),
        local_action_before=float(
            local_before
        ),
        local_action_after=float(
            local_after
        ),
        absolute_action_error=float(
            absolute_error
        ),
        staple_scale=float(
            scale
        ),
    )


def overrelaxation_sweep(
    lattice: Lattice,
    beta: float,
    minimum_scale: float = 1e-14,
) -> OverrelaxationSweepResult:
    """
    Perform one sequential microcanonical sweep over every lattice link.

    A link is skipped only when its staple cannot be stably normalized.
    """
    if beta < 0.0:
        raise ValueError(
            "beta must be nonnegative."
        )

    errors = []

    updated_links = 0
    skipped_links = 0

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            try:
                result = (
                    overrelaxation_link_update(
                        lattice=lattice,
                        site=site,
                        mu=mu,
                        beta=beta,
                        minimum_scale=minimum_scale,
                    )
                )
            except ValueError as error:
                message = str(
                    error
                )

                if (
                    "Staple scale is too small"
                    in message
                ):
                    skipped_links += 1
                    continue

                raise

            updated_links += 1

            errors.append(
                result.absolute_action_error
            )

    if len(
        errors
    ) == 0:
        maximum_error = 0.0
        mean_error = 0.0
    else:
        maximum_error = float(
            np.max(
                errors
            )
        )

        mean_error = float(
            np.mean(
                errors
            )
        )

    return OverrelaxationSweepResult(
        updated_links=updated_links,
        skipped_links=skipped_links,
        maximum_local_action_error=(
            maximum_error
        ),
        mean_local_action_error=(
            mean_error
        ),
    )


def hybrid_metropolis_overrelaxation_sweep(
    lattice: Lattice,
    beta: float,
    epsilon: float,
    overrelaxation_sweeps: int = 1,
) -> HybridSweepResult:
    """
    Perform one Metropolis sweep followed by zero or more overrelaxation sweeps.
    """
    if overrelaxation_sweeps < 0:
        raise ValueError(
            "overrelaxation_sweeps must be nonnegative."
        )

    acceptance_rate = metropolis_sweep(
        lattice=lattice,
        beta=beta,
        epsilon=epsilon,
    )

    total_updated = 0
    total_skipped = 0
    maximum_error = 0.0

    for _ in range(
        overrelaxation_sweeps
    ):
        result = overrelaxation_sweep(
            lattice=lattice,
            beta=beta,
        )

        total_updated += (
            result.updated_links
        )

        total_skipped += (
            result.skipped_links
        )

        maximum_error = max(
            maximum_error,
            result.maximum_local_action_error,
        )

    return HybridSweepResult(
        metropolis_acceptance_rate=float(
            acceptance_rate
        ),
        overrelaxation_sweeps=int(
            overrelaxation_sweeps
        ),
        overrelaxation_updated_links=int(
            total_updated
        ),
        overrelaxation_skipped_links=int(
            total_skipped
        ),
        maximum_local_action_error=float(
            maximum_error
        ),
    )
