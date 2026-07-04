"""
Abstract group interface for future SU(N) lattice gauge theory support.

The current simulator is implemented primarily for SU(2). The repository also
contains a tested SU(3) matrix engine.

This module defines a lightweight interface describing what a gauge group
backend must provide for a future generic SU(N) lattice simulator.

It does not replace the current SU(2) lattice engine yet. It documents and
tests the architecture needed to evolve the project toward generic compact
matrix Lie groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ymlab import su2, su3


@dataclass(frozen=True)
class MatrixGaugeGroup:
    """
    Minimal interface for a compact matrix gauge group backend.

    Attributes
    ----------
    name:
        Human-readable group name, such as "SU(2)" or "SU(3)".
    dimension:
        Matrix dimension N for SU(N).
    trace_normalization:
        Normalization factor used in Wilson action observables.
        For SU(N), this is usually 1/N.
    identity:
        Function returning the identity matrix.
    random:
        Function returning a random group element.
    small_random:
        Function returning a small random group perturbation.
    dagger:
        Function returning Hermitian conjugate.
    is_member:
        Function checking group membership.
    real_trace:
        Function returning real trace.
    """

    name: str
    dimension: int
    trace_normalization: float
    identity: Callable[[], np.ndarray]
    random: Callable[..., np.ndarray]
    small_random: Callable[..., np.ndarray]
    dagger: Callable[[np.ndarray], np.ndarray]
    is_member: Callable[[np.ndarray], bool]
    real_trace: Callable[[np.ndarray], float]


def su2_group() -> MatrixGaugeGroup:
    """Return a MatrixGaugeGroup wrapper for SU(2)."""
    return MatrixGaugeGroup(
        name="SU(2)",
        dimension=2,
        trace_normalization=1.0 / 2.0,
        identity=su2.identity,
        random=su2.random_su2,
        small_random=su2.small_random_su2,
        dagger=su2.dagger,
        is_member=su2.is_su2,
        real_trace=su2.real_trace,
    )


def su3_group() -> MatrixGaugeGroup:
    """Return a MatrixGaugeGroup wrapper for SU(3)."""
    return MatrixGaugeGroup(
        name="SU(3)",
        dimension=3,
        trace_normalization=1.0 / 3.0,
        identity=su3.identity,
        random=su3.random_su3,
        small_random=su3.small_random_su3,
        dagger=su3.dagger,
        is_member=su3.is_su3,
        real_trace=su3.real_trace,
    )


def normalized_real_trace(group: MatrixGaugeGroup, matrix: np.ndarray) -> float:
    """
    Compute normalized real trace for a group element.

    For SU(N), this is

        (1/N) Re Tr(U).
    """
    return group.trace_normalization * group.real_trace(matrix)


def wilson_plaquette_density_from_matrix(
    group: MatrixGaugeGroup,
    plaquette_matrix: np.ndarray,
) -> float:
    """
    Compute generic Wilson plaquette density from a plaquette matrix.

    For SU(N):

        1 - (1/N) Re Tr(U_p)
    """
    return 1.0 - normalized_real_trace(group, plaquette_matrix)


def validate_group_backend(
    group: MatrixGaugeGroup,
    samples: int = 10,
    seed: int = 123,
) -> bool:
    """
    Validate that a group backend satisfies basic numerical expectations.

    This checks:
    - identity has correct shape,
    - identity is a group member,
    - random samples are group members,
    - small random samples are group members,
    - normalized trace of identity is 1,
    - Wilson density of identity plaquette is 0.
    """
    if samples <= 0:
        raise ValueError("samples must be positive.")

    rng = np.random.default_rng(seed)

    identity = group.identity()

    if identity.shape != (group.dimension, group.dimension):
        return False

    if not group.is_member(identity):
        return False

    if not np.isclose(normalized_real_trace(group, identity), 1.0):
        return False

    if not np.isclose(wilson_plaquette_density_from_matrix(group, identity), 0.0):
        return False

    for _ in range(samples):
        random_element = group.random(rng)
        small_element = group.small_random(rng=rng)

        if not group.is_member(random_element):
            return False

        if not group.is_member(small_element):
            return False

    return True


def available_groups() -> list[MatrixGaugeGroup]:
    """Return currently available gauge group backends."""
    return [su2_group(), su3_group()]
