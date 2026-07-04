"""
SU(2) matrix utilities.

SU(2) is the group of 2x2 complex unitary matrices with determinant 1.

A standard parameterization is

    U = a0 I + i(a1 sigma_1 + a2 sigma_2 + a3 sigma_3)

where (a0, a1, a2, a3) lies on the unit 3-sphere.
"""

from __future__ import annotations

import numpy as np


def identity() -> np.ndarray:
    """Return the 2x2 identity matrix in SU(2)."""
    return np.eye(2, dtype=np.complex128)


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian conjugate of a matrix."""
    return np.conjugate(matrix.T)


def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is unitary."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return np.allclose(dagger(matrix) @ matrix, identity(), atol=atol)


def determinant(matrix: np.ndarray) -> complex:
    """Return the determinant of a matrix."""
    return np.linalg.det(matrix)


def is_su2(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is numerically in SU(2)."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return (
        matrix.shape == (2, 2)
        and is_unitary(matrix, atol)
        and np.isclose(determinant(matrix), 1.0, atol=atol)
    )


def from_quaternion(a: np.ndarray) -> np.ndarray:
    """
    Convert a 4-vector into an SU(2) matrix.

    The vector is normalized automatically.
    """
    a = np.asarray(a, dtype=float)

    if a.shape != (4,):
        raise ValueError("Expected a 4-vector.")

    norm = np.linalg.norm(a)

    if norm == 0:
        raise ValueError("Cannot construct SU(2) matrix from zero vector.")

    a0, a1, a2, a3 = a / norm

    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


def random_su2(rng=None) -> np.ndarray:
    """
    Sample a random SU(2) matrix using a normalized Gaussian quaternion.
    """
    if rng is None:
        rng = np.random.default_rng()

    q = rng.normal(size=4)
    return from_quaternion(q)


def small_random_su2(epsilon: float = 0.1, rng=None) -> np.ndarray:
    """
    Generate a small random SU(2) perturbation close to the identity.

    This is useful for future Metropolis updates.
    """
    if rng is None:
        rng = np.random.default_rng()

    v = rng.normal(scale=epsilon, size=3)
    angle = np.linalg.norm(v)

    if angle == 0:
        return identity()

    axis = v / angle
    a0 = np.cos(angle)
    spatial = np.sin(angle) * axis

    return from_quaternion(np.array([a0, spatial[0], spatial[1], spatial[2]]))


def reunitarize(matrix: np.ndarray) -> np.ndarray:
    """
    Project a near-SU(2) matrix back toward SU(2) using singular value decomposition.
    """
    u, _, vh = np.linalg.svd(matrix)
    projected = u @ vh
    det = np.linalg.det(projected)
    projected = projected / np.sqrt(det)
    return projected.astype(np.complex128)


def real_trace(matrix: np.ndarray) -> float:
    """Return the real part of the trace."""
    return float(np.real(np.trace(matrix)))
