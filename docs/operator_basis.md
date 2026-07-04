# Multi-Operator Scalar Basis and Correlator Matrices

## Purpose

Earlier spectroscopy phases analyzed one scalar gauge-invariant operator at a time.

A single operator may couple to several states and may have poor overlap with the lowest state of interest.

The Yang-Mills Mass Gap Laboratory therefore introduces a multi-operator scalar basis.

The first basis uses several spatial smearing levels measured on the same underlying gauge configurations.

## Operator Basis

The current scalar basis is constructed from:

- unsmeared spatial plaquettes,
- lightly smeared spatial plaquettes,
- moderately smeared spatial plaquettes,
- more heavily smeared spatial plaquettes.

Each basis element produces a Euclidean time-slice series:

O_i(t).

All operators remain gauge invariant because they are constructed from traces of closed plaquette loops after gauge-covariant spatial smearing.

## Correlator Matrix

For operators O_i and O_j, the project estimates

C_ij(tau)
=
<O_i(t) O_j(t + tau)>
-
<O_i><O_j>.

The expectation value is estimated over measured gauge configurations and periodic Euclidean time origins.

The resulting observable is a matrix for every Euclidean time lag.

## Cross-Correlation Structure

Diagonal entries measure the autocorrelation of one operator.

Off-diagonal entries measure cross-correlations between different operator constructions.

These off-diagonal components contain information about the overlap structure of the operator basis.

## Time-Reversal and Matrix Symmetry

Periodic cross-correlations satisfy the relation

C_ij(tau)
approximately corresponds to
C_ji(T - tau).

At zero lag, the correlator matrix is symmetric.

For later variational analysis, the project explicitly constructs the symmetric operator-space matrix

C_sym(t)
=
[C(t) + C(t)^T] / 2.

The magnitude of the original matrix asymmetry is recorded before symmetrization.

## Correlator Matrix Eigenvalues

The project computes ordinary eigenvalues of each symmetrized correlator matrix as a diagnostic.

These are not yet the principal correlators of a generalized eigenvalue problem.

Ordinary eigenvalues of C(t) alone do not correctly normalize the operator metric at a reference time.

They are included only as matrix-structure diagnostics.

## Why This Phase Matters

The project has advanced from

one operator
-> one correlator

to

operator basis
-> cross-correlations
-> matrix-valued Euclidean correlator.

This is the required mathematical structure for a variational spectroscopy method.

## Scientific Limitations

The present operator basis is generated only from different smearing depths of one scalar plaquette construction.

A professional spectroscopy program may include:

- different loop shapes,
- spatial blocking,
- symmetry-projected operators,
- multiple lattice irreducible representations,
- larger operator bases,
- carefully optimized smearing parameters.

The next phase introduces a regularized generalized eigenvalue problem using the correlator matrix.
