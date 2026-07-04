"""
SU(3) matrix utilities.

SU(3) is the group of 3x3 complex unitary matrices with determinant 1.

This module is an important extension point for the project because SU(3) is
the gauge group of quantum chromodynamics. The current lattice simulator is
implemented for SU(2), but this module provides a tested foundation for future
SU(3) lattice gauge theory support.
"""

from __future__ import annotations

import numpy as np


def identity() -> np.ndarray:
    """Return the 3x3 identity matrix in SU(3)."""
    return np.eye(3, dtype=np.complex128)


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian conjugate of a matrix."""
    return np.conjugate(matrix.T)


def determinant(matrix: np.ndarray) -> complex:
    """Return the determinant of a matrix."""
    return np.linalg.det(matrix)


def is_unitary(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is unitary."""
    matrix = np.asarray(matrix, dtype=np.complex128)

    if matrix.shape != (3, 3):
        return False

    return np.allclose(dagger(matrix) @ matrix, identity(), atol=atol)


def is_su3(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is numerically in SU(3)."""
    matrix = np.asarray(matrix, dtype=np.complex128)

    return (
        matrix.shape == (3, 3)
        and is_unitary(matrix, atol=atol)
        and np.isclose(determinant(matrix), 1.0, atol=atol)
    )


def project_to_su3(matrix: np.ndarray) -> np.ndarray:
    """
    Project a nonsingular complex 3x3 matrix toward SU(3).

    This uses QR decomposition to obtain a unitary matrix and then adjusts the
    determinant phase so that det(U) = 1.

    This is useful for creating random SU(3) matrices and for numerical
    reunitarization.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)

    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix.")

    q, r = np.linalg.qr(matrix)

    # Fix QR phase ambiguity so the distribution is stable.
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal, dtype=np.complex128)

    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])

    q = q @ np.diag(np.conjugate(phases))

    det_q = np.linalg.det(q)

    if np.isclose(abs(det_q), 0.0):
        raise ValueError("Cannot project singular matrix to SU(3).")

    q = q / det_q ** (1.0 / 3.0)

    # Numerical cleanup: apply determinant correction again if needed.
    det_q = np.linalg.det(q)
    q = q / det_q ** (1.0 / 3.0)

    return q.astype(np.complex128)


def random_su3(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a random SU(3) matrix.

    A complex Gaussian matrix is projected to SU(3) using QR decomposition.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    return project_to_su3(z)


def small_random_su3(
    epsilon: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a small random SU(3) perturbation close to identity.

    We build a small anti-Hermitian traceless matrix H and exponentiate it:

        U = exp(H)

    where H is anti-Hermitian and trace-free, so U is unitary with determinant 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    a = rng.normal(scale=epsilon, size=(3, 3)) + 1j * rng.normal(
        scale=epsilon,
        size=(3, 3),
    )

    anti_hermitian = a - dagger(a)
    trace_part = np.trace(anti_hermitian) / 3.0
    anti_hermitian = anti_hermitian - trace_part * identity()

    try:
        from scipy.linalg import expm
    except ImportError as error:
        raise ImportError("small_random_su3 requires scipy.") from error

    u = expm(anti_hermitian)

    return project_to_su3(u)


def reunitarize(matrix: np.ndarray) -> np.ndarray:
    """
    Reproject a near-SU(3) matrix back to SU(3).
    """
    return project_to_su3(matrix)


def real_trace(matrix: np.ndarray) -> float:
    """Return the real part of the trace."""
    return float(np.real(np.trace(matrix)))


def gell_mann_matrices() -> list[np.ndarray]:
    """
    Return the eight Gell-Mann matrices.

    These form a standard basis for the Lie algebra su(3) after multiplication
    by i and appropriate real linear combinations.
    """
    zero = 0.0 + 0.0j
    one = 1.0 + 0.0j
    i = 1.0j

    lambda_1 = np.array(
        [
            [zero, one, zero],
            [one, zero, zero],
            [zero, zero, zero],
        ],
        dtype=np.complex128,
    )

    lambda_2 = np.array(
        [
            [zero, -i, zero],
            [i, zero, zero],
            [zero, zero, zero],
        ],
        dtype=np.complex128,
    )

    lambda_3 = np.array(
        [
            [one, zero, zero],
            [zero, -one, zero],
            [zero, zero, zero],
        ],
        dtype=np.complex128,
    )

    lambda_4 = np.array(
        [
            [zero, zero, one],
            [zero, zero, zero],
            [one, zero, zero],
        ],
        dtype=np.complex128,
    )

    lambda_5 = np.array(
        [
            [zero, zero, -i],
            [zero, zero, zero],
            [i, zero, zero],
        ],
        dtype=np.complex128,
    )

    lambda_6 = np.array(
        [
            [zero, zero, zero],
            [zero, zero, one],
            [zero, one, zero],
        ],
        dtype=np.complex128,
    )

    lambda_7 = np.array(
        [
            [zero, zero, zero],
            [zero, zero, -i],
            [zero, i, zero],
        ],
        dtype=np.complex128,
    )

    lambda_8 = (1.0 / np.sqrt(3.0)) * np.array(
        [
            [one, zero, zero],
            [zero, one, zero],
            [zero, zero, -2.0 * one],
        ],
        dtype=np.complex128,
    )

    return [
        lambda_1,
        lambda_2,
        lambda_3,
        lambda_4,
        lambda_5,
        lambda_6,
        lambda_7,
        lambda_8,
    ]


def is_hermitian(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is Hermitian."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return np.allclose(matrix, dagger(matrix), atol=atol)


def is_traceless(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is traceless."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return np.isclose(np.trace(matrix), 0.0, atol=atol)
