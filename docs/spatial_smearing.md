# Spatial-Link Smearing and Operator Improvement

## Purpose

The scalar spectroscopy pipeline originally constructed its operator directly from unsmeared plaquettes.

Local operators built from raw gauge links can produce noisy finite-lattice correlators and may have limited overlap with low-lying states.

The project therefore introduces a spatial-link smearing layer for operator improvement.

The underlying Monte Carlo configuration is not replaced in the sampling algorithm.

Smearing is applied only when constructing spectroscopy observables.

## Euclidean Time Direction

One lattice direction is designated as Euclidean time.

The smearing algorithm updates only spatial links.

Temporal links remain exactly unchanged.

This preserves the explicit temporal structure used by the Euclidean correlator analysis.

## Gauge-Covariant Spatial Side Paths

For a spatial link U_mu(x), every smearing side path begins at x and ends at x + mu.

For a spatial direction nu different from mu, a forward path is

U_nu(x)
U_mu(x + nu)
U_nu(x + mu)^dagger.

A backward path is

U_nu(x - nu)^dagger
U_mu(x - nu)
U_nu(x - nu + mu).

These paths transform under local gauge transformations in the same endpoint-covariant manner as U_mu(x).

## Smearing Combination

One spatial smearing step constructs

X_mu(x)
=
(1 - alpha) U_mu(x)
+
alpha / N_side
times
S_mu(x),

where S_mu(x) is the sum of spatial side paths.

N_side is the number of side paths contributing to the selected spatial link.

The matrix X_mu(x) is generally not exactly inside SU(2).

It is therefore projected back to SU(2).

## Simultaneous Updating

Every smeared link at level n + 1 is computed from the lattice at level n.

The algorithm does not overwrite a link and then use that newly smeared link later in the same smearing sweep.

This makes one smearing step simultaneous.

## SU(2) Projection Validation

After smearing, the project measures numerical violations of:

U^dagger U = I

and

det(U) = 1.

Unit tests require projected links to remain in SU(2) to numerical tolerance.

## Gauge-Invariance Validation

The project performs the following comparison:

1. Begin with a nontrivial SU(2) lattice U.
2. Generate a random local gauge transformation.
3. Construct U^G.
4. Smear U.
5. Smear U^G.
6. Construct scalar glueball-style operator time series from both.
7. Verify the gauge-invariant operator values agree numerically.

This checks the gauge-covariant structure of the smearing pipeline through a gauge-invariant downstream observable.

## Smearing-Level Comparison

The spectroscopy experiment uses the same underlying measured gauge configurations at every smearing level.

Current comparison levels include:

- zero smearing steps,
- two smearing steps,
- five smearing steps,
- ten smearing steps.

For every level, the project measures:

- mean scalar operator,
- connected C(0),
- normalized Euclidean correlator,
- number of finite arccosh effective-mass points,
- heuristic plateau score,
- periodic cosh fit availability,
- fitted lattice mass when a valid fit exists,
- fit residual mean squared error,
- maximum SU(2) projection error.

## Why Same-Configuration Comparison Matters

Using independent Monte Carlo chains for each smearing level would mix operator changes with configuration-sampling differences.

The project instead measures every smearing level on the same configuration ensemble.

This creates a more direct comparison of the operator-improvement transformation.

## Scientific Limitations

Smearing does not prove a mass gap.

A visually smoother correlator does not by itself establish ground-state dominance.

The current project does not yet include:

- a large operator basis,
- lattice symmetry irreducible-representation projection,
- generalized eigenvalue analysis,
- covariance-aware correlated fits,
- continuum extrapolation,
- infinite-volume extrapolation.

The next technical phase will construct a multi-operator basis and an ensemble correlator matrix.
