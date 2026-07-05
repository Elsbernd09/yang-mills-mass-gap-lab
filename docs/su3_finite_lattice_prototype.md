# SU(3) Finite-Lattice Prototype

## Purpose

The generic GaugeLattice architecture provides a matrix-group-independent
lattice container and generic Wilson-action machinery.

This phase promotes SU(3) from an architecture smoke test to a directly
validated finite-lattice simulation prototype.

The prototype focuses on structural correctness.

## SU(3) Link Variables

The active MatrixGaugeGroup backend is SU(3).

Every lattice link is a three by three complex matrix satisfying numerical
unitarity and determinant one.

GaugeLattice validates group membership before storing every link.

## Wilson Normalization

The SU(3) backend supplies a multiplicative trace normalization of one third.

The normalized plaquette observable is

one third Re Tr(U_p).

The Wilson plaquette density is

one minus one third Re Tr(U_p).

## Local Wilson Action

For a selected link U and reverse-oriented staple V, the link-dependent action
is

minus beta divided by three
times
Re Tr(U V).

The generic implementation obtains the factor one third from the active group
backend.

## Local-Versus-Global Action Validation

For a proposed SU(3) link replacement, the project calculates the action
difference in two independent ways.

The local calculation uses the staple.

The global calculation recomputes the complete Wilson action before and after
temporarily replacing the selected link.

The proposal link is then restored.

The required identity is

Delta S local
equals
Delta S global

up to floating-point error.

This test probes plaquette orientation, staple orientation, trace
normalization, and local-action normalization simultaneously.

## Local Gauge Transformations

A site-dependent SU(3) gauge field G(x) transforms a positive link as

U_mu(x)
to
G(x) U_mu(x) G(x plus mu) dagger.

The generic transformation engine constructs a new independent GaugeLattice.

## Gauge-Invariant Observables

The project directly validates invariance of:

- the complete Wilson action,
- the average normalized plaquette,
- closed rectangular Wilson loops.

The transformed links are also checked for continued SU(3) membership.

## Structural Validation Audit

The dedicated SU(3) structural audit generates a nontrivial finite
configuration and performs many random local-action comparisons.

It then applies a random local SU(3) gauge transformation.

The audit reports:

- local/global action consistency count,
- maximum action-difference error,
- mean action-difference error,
- Wilson-action gauge error,
- average-plaquette gauge error,
- closed-loop gauge error,
- maximum transformed-link membership error.

The audit exits with failure if the required structural checks fail.

## SU(3) Metropolis Evolution

The current SU(3) update uses a small random near-identity SU(3) proposal.

The proposal is left-multiplied onto the active link.

A local Wilson-action difference determines the Metropolis acceptance
probability.

Accepted links pass the GaugeLattice membership validator before storage.

## Independent Short Chains

The exploratory ensemble experiment runs several chains with independent
random seeds.

For each chain, it reports:

- thermalization acceptance,
- measurement acceptance,
- mean average plaquette,
- plaquette standard deviation,
- mean Wilson action per plaquette,
- integrated plaquette autocorrelation time,
- plaquette effective sample size,
- selected rectangular Wilson loops,
- maximum final link-membership error.

## Why This Is Still a Prototype

The update algorithm is a small-step Metropolis method.

The project does not yet implement an SU(3) Cabibbo-Marinari heatbath or a
production overrelaxation algorithm.

The lattices and chains are small.

No scale setting is performed.

No continuum limit is taken.

No infinite-volume limit is taken.

Equal bare couplings across different gauge groups would not constitute a
matched physical comparison.

## Scientific Scope

This phase demonstrates a structurally validated finite-lattice SU(3)
simulation path inside the generic GaugeLattice architecture.

It validates local action geometry and local gauge invariance numerically.

It does not prove Yang-Mills existence or the continuum mass gap.
