# Regularized Generalized Eigenvalue Problem

## Purpose

The multi-operator scalar basis produces a Euclidean correlator matrix

C_ij(t).

Analyzing each operator independently does not fully use the cross-correlation information in the basis.

The project therefore introduces a variational generalized eigenvalue problem.

The central equation is

C(t) v_n(t, t0)
=
lambda_n(t, t0) C(t0) v_n(t, t0).

The generalized eigenvalues lambda_n are called principal correlators.

## Variational Motivation

Suppose a collection of operators overlaps with several energy eigenstates.

Different linear combinations of the operators can have different state overlap.

The variational method searches operator-space directions described by generalized eigenvectors.

This can help separate dominant correlator modes more effectively than examining one raw operator at a time.

## Why Direct Matrix Inversion Is Avoided

A naive implementation could form

C(t0)^(-1) C(t).

This is numerically dangerous.

The operator basis contains several smearing levels of a closely related scalar observable.

These operators can be highly correlated.

As a result, C(t0) may be poorly conditioned or nearly singular.

The project therefore does not directly invert the reference correlator matrix.

## Reference Metric Eigendecomposition

The symmetrized reference matrix is diagonalized:

C(t0)
=
V D V^T.

The eigenvalues of D define numerical metric modes.

Only sufficiently positive eigenvalues are retained.

The cutoff is based on both:

- an absolute threshold,
- a threshold relative to the largest positive metric eigenvalue.

Modes below the cutoff are removed from the numerical variational space.

## Whitening Transformation

For retained eigenvectors V_r and retained positive eigenvalues D_r, define

W
=
V_r D_r^(-1/2).

Then

W^T C(t0) W
=
I

in the retained subspace.

The project explicitly measures the maximum numerical deviation from the identity.

## Reduced Symmetric Eigenproblem

For every Euclidean time lag, the project constructs

M(t)
=
W^T C(t) W.

The matrix M(t) is symmetrized numerically.

The ordinary symmetric eigenproblem

M(t) q_n
=
lambda_n(t, t0) q_n

is then solved.

The generalized operator-space vectors are

v_n
=
W q_n.

## Metric Orthonormality

At the reference time, the generalized vectors satisfy

v_m^T C(t0) v_n
approximately equals
delta_mn.

The project calculates the maximum metric-orthonormality error.

## Principal Correlators

The generalized eigenvalues are the principal correlators

lambda_n(t, t0).

At the reference time,

lambda_n(t0, t0)
=
1

for every retained state, up to numerical floating-point error.

This identity is tested using exact synthetic correlator systems.

## Synthetic Exact Validation

The test suite constructs matrix correlators of the form

C(t)
=
A exp(-E t) A^T,

where the energy values E are known exactly.

The regularized generalized eigenvalue solver is required to recover the corresponding principal correlators.

The project also validates:

- reference whitening,
- rank truncation,
- metric orthonormality,
- reference-time principal correlator normalization,
- exact log-ratio effective masses.

## State Tracking

Eigenvalues sorted independently at every Euclidean time can exchange order when noisy states cross.

The implementation therefore optionally tracks states by eigenvector overlap.

Between adjacent times, the absolute overlap matrix is constructed.

A one-to-one assignment maximizing total overlap is selected.

Eigenvector signs are also aligned between adjacent time slices.

This is a numerical state-tracking diagnostic.

It does not guarantee perfect physical state identification in a noisy or nearly degenerate spectrum.

## Variational Effective Masses

For positive adjacent principal correlators, the project computes

m_eff(t)
=
log[
lambda(t) / lambda(t + 1)
].

The project also calculates periodic arccosh effective masses when the principal correlator values support the estimator.

Invalid points are returned as NaN.

No absolute value is used to force negative or noisy principal correlators into a mass estimate.

## Scientific Limitations

The present GEVP analysis remains exploratory.

Important limitations include:

- a small operator basis,
- operators differentiated primarily by smearing depth,
- finite statistics,
- possible state-tracking ambiguity,
- no covariance-weighted fit of principal correlators,
- no continuum extrapolation,
- no infinite-volume extrapolation,
- no physical scale setting.

The GEVP produces finite-lattice variational spectroscopy diagnostics.

It does not prove the continuum Yang-Mills mass gap.
