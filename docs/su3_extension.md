# SU(3) Extension Foundation

## Purpose

The first version of this project focuses on SU(2) lattice gauge theory because
SU(2) is non-Abelian, mathematically rich, and computationally accessible.

This phase adds a tested SU(3) matrix module as a foundation for future SU(3)
lattice gauge theory support.

## Why SU(3) Matters

SU(3) is the gauge group of quantum chromodynamics, the theory of the strong
nuclear force. While the Clay Yang–Mills problem is stated for compact simple
gauge groups more generally, SU(3) is one of the most physically important
examples.

## SU(3)

The special unitary group SU(3) is

\[
SU(3)
=
\{ U \in \mathbb{C}^{3 \times 3} : U^\dagger U = I,\ \det(U)=1 \}.
\]

Its Lie algebra \(\mathfrak{su}(3)\) consists of traceless anti-Hermitian
matrices. Physicists often describe the related Hermitian basis using the eight
Gell-Mann matrices.

## Gell-Mann Matrices

The eight Gell-Mann matrices form a standard basis for traceless Hermitian
\(3 \times 3\) matrices. They satisfy the normalization relation

\[
\operatorname{Tr}(\lambda_i \lambda_j) = 2\delta_{ij}.
\]

The project includes tests verifying that the implemented Gell-Mann matrices
are Hermitian, traceless, and correctly normalized.

## Current Implementation

The SU(3) module includes:

- identity matrix,
- Hermitian conjugation,
- determinant and unitarity checks,
- numerical SU(3) membership checks,
- random SU(3) generation by QR projection,
- small random SU(3) perturbations by exponentiating traceless anti-Hermitian matrices,
- reunitarization/projection back to SU(3),
- Gell-Mann matrices.

## Limitation

The full lattice simulator is still implemented for SU(2). Extending the
lattice engine to generic SU(N), and then to SU(3), requires carefully
generalizing:

- link initialization,
- trace normalization,
- Wilson action normalization,
- local staple updates,
- Wilson loops,
- tests and validation benchmarks.

This SU(3) module is a roadmap milestone, not yet a complete SU(3) lattice
gauge simulator.
