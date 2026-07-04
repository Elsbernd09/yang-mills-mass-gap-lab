# Generic GaugeLattice Architecture

## Purpose

The original lattice implementation is specialized for SU(2).

That implementation remains preserved because it is already covered by a
large regression and validation suite.

The project now introduces a second group-aware lattice architecture called
GaugeLattice.

GaugeLattice obtains matrix dimension and group operations from a
MatrixGaugeGroup backend.

## Group Backend

A MatrixGaugeGroup defines:

- group name,
- matrix dimension,
- trace normalization,
- identity construction,
- random group sampling,
- near-identity random proposals,
- Hermitian conjugation,
- group-membership validation,
- real trace.

Current backends include SU(2) and SU(3).

## GaugeLattice

GaugeLattice stores one matrix-valued link for every lattice site and positive
lattice direction.

The link matrix shape is determined by the gauge-group dimension.

For SU(2), links are 2 by 2 matrices.

For SU(3), links are 3 by 3 matrices.

The lattice itself does not hard-code either matrix dimension.

## Strict Group Membership

Every link stored through set_link is validated using the active group backend.

A matrix with the wrong shape is rejected.

A matrix outside the selected group is rejected.

This preserves a strong lattice-level invariant.

## Generic Plaquette

The generic plaquette is

U_mu(x)
U_nu(x plus mu)
U_mu(x plus nu) dagger
U_nu(x) dagger.

The same path geometry applies independently of the matrix dimension.

## Trace Normalization

The MatrixGaugeGroup backend stores the multiplicative trace-normalization
factor.

For SU(N), normalized Wilson observables use

one divided by N
times
Re Tr.

Therefore the generic implementation evaluates

trace_normalization
times
Re Tr(matrix).

For SU(2), the backend factor is one half.

For SU(3), the backend factor is one third.

This convention was directly regression-tested against the established SU(2)
implementation.

## Generic Wilson Action

The generic Wilson action is

beta
times
the sum over plaquettes of

one
minus
one over N
Re Tr(U_p).

The same implementation can therefore operate on SU(2) and SU(3) link fields.

## Generic Wilson-Action Staple

The reverse-oriented local staple is constructed so that

U_mu(x) V

closes the plaquette paths containing the selected link.

The path orientation is the same one previously validated by the local-versus-
global Wilson-action consistency audit.

The staple matrix dimension is obtained from the active gauge group.

## Generic Local Action

The link-dependent local action is

minus beta times one over N
times
Re Tr(U V).

This produces a generic local action difference for matrix gauge groups.

## Generic Metropolis Proposal

The active group backend generates a small random near-identity group element.

The proposal is left-multiplied onto the current link.

The local Wilson-action difference determines the Metropolis acceptance
probability.

The accepted proposal is stored only after group-membership validation.

## Generic Wilson Loops

Rectangular Wilson loops are constructed from generic forward and reverse link
transport.

The final real trace is multiplied by the active matrix-group trace
normalization.

## SU(2) Regression Equivalence

The new architecture is validated against the established SU(2) backend on the
same link configuration.

The comparison includes:

- plaquette matrices,
- average plaquette,
- Wilson action,
- reverse-oriented staples,
- local action differences,
- rectangular Wilson loops.

The old and generic SU(2) implementations are required to agree numerically.

The dedicated normalization repair audit verifies:

- cold normalized trace equals one,
- cold average plaquette equals one,
- cold Wilson action equals zero,
- old and generic SU(2) average plaquettes agree,
- old and generic SU(2) Wilson actions agree,
- old and generic SU(2) local action differences agree,
- old and generic SU(2) Wilson loops agree.

## SU(3) Architecture Smoke Test

The generic lattice is also instantiated with the SU(3) backend.

The project validates:

- cold-start identity links,
- zero cold Wilson action,
- unit cold average plaquette,
- 3 by 3 link shape,
- local Metropolis evolution,
- preservation of SU(3) link membership.

## Important Scientific Distinction

This phase creates a real generic finite-lattice architecture and demonstrates
an SU(3) simulation prototype path.

It does not yet establish a production-quality SU(3) ensemble study.

The current SU(3) update uses small random Metropolis proposals.

The next phase focuses specifically on the SU(3) lattice prototype,
local-versus-global action validation, gauge invariance, Wilson loops, and
finite-lattice ensemble diagnostics.

## Scientific Limitations

The generic implementation is pure Python and NumPy.

The architecture prioritizes transparency and regression validation.

It is not intended to compete with optimized production lattice gauge theory
software.

The generic SU(3) prototype does not imply physical comparison with SU(2) at
equal bare beta.

No continuum or infinite-volume claim is made.

The generic GaugeLattice is infrastructure for controlled finite-lattice
experiments.
