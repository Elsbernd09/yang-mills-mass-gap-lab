# SU(2) Microcanonical Overrelaxation

## Purpose

The original sampling engine uses local stochastic Metropolis updates.

Metropolis updates sample the target finite-lattice Wilson distribution, but
successive configurations can remain strongly autocorrelated.

The project therefore introduces SU(2) microcanonical overrelaxation as a
deterministic companion update.

## Local Wilson-Action Geometry

For a link U and its reverse-oriented Wilson-action staple V, the local action
is proportional to

Re Tr(U V).

The SU(2) staple sum retains quaternionic matrix structure.

For a nonzero staple,

V = k V_tilde,

where k is positive and V_tilde is in SU(2).

The local action is therefore proportional to

k Re Tr(U V_tilde).

## Staple Normalization

The numerical implementation extracts the SU(2) quaternion norm using

k squared
=
one half Re Tr(V dagger V).

For an exact SU(2)-quaternion-form matrix,

V dagger V
=
k squared I.

This approach avoids relying on a floating-point determinant to extract the
staple scale.

The normalized staple is explicitly validated as an SU(2) matrix.

## Microcanonical Reflection

Define

X = U V_tilde.

The real trace of an SU(2) matrix is unchanged by Hermitian conjugation:

Re Tr(X dagger)
=
Re Tr(X).

The algorithm reflects

X
to
X dagger.

Solving for the updated link gives

U prime
=
V_tilde dagger
U dagger
V_tilde dagger.

Floating-point matrix multiplication can introduce tiny deviations from the
exact SU(2) manifold.

The reflected proposal is therefore reunitarized before it is stored.

The lattice continues to enforce strict SU(2) membership for every stored link.

## Global Action Preservation

Changing one link affects only plaquettes containing that link.

The staple contains exactly those local Wilson-action contributions.

Therefore preservation of the staple-based local action implies preservation
of the full Wilson action for one link update, up to numerical error.

The project validates this directly by recomputing the complete Wilson action
before and after:

- individual reflected links,
- complete overrelaxation sweeps,
- repeated sweeps in two, three, and four dimensions.

## Why Overrelaxation Is Not Used Alone

The overrelaxation transformation is deterministic and microcanonical.

It moves through a surface of constant Wilson action.

The complete sampling engine therefore retains stochastic Metropolis sweeps.

A hybrid schedule is

Metropolis
followed by
zero or more overrelaxation sweeps.

The Metropolis step provides stochastic action-changing evolution.

The overrelaxation steps provide deterministic microcanonical motion.

## Hybrid Schedules

The efficiency experiment compares:

- Metropolis only,
- Metropolis plus one overrelaxation sweep,
- Metropolis plus two overrelaxation sweeps,
- Metropolis plus four overrelaxation sweeps.

Every schedule begins from a copy of the same thermalized configuration.

## Observable-Dependent Autocorrelation

Sampling efficiency is evaluated for two observables:

- average plaquette,
- scalar glueball-style composite operator.

One update schedule need not decorrelate every observable equally.

The project therefore reports integrated autocorrelation time independently
for each observable.

## Effective Sample Size

Given integrated autocorrelation time, the diagnostic pipeline estimates an
effective sample size.

A lower autocorrelation time can increase effective sample size for a fixed
number of measured updates.

## Effective Sample Size Per Second

Overrelaxation sweeps consume computational time.

Therefore the schedule with the lowest autocorrelation time is not
automatically the most efficient.

The experiment computes

effective sample size divided by wall-clock seconds.

This ESS-per-second metric compares statistical decorrelation with runtime
cost.

## Numerical Action Audit

The project records:

- local action error for reflected links,
- maximum local error in a sweep,
- full Wilson action before a sweep,
- full Wilson action after a sweep,
- absolute global action error,
- relative global action error.

A dedicated multidimensional audit runs overrelaxation on nontrivial 2D, 3D,
and 4D SU(2) configurations.

## Scientific Limitations

The current implementation is pure Python and NumPy.

Absolute runtime comparisons therefore describe this implementation rather
than a highly optimized lattice gauge theory code.

The autocorrelation estimator uses the project's current positive-sequence
integrated-autocorrelation diagnostic.

A production study would investigate longer chains, multiple independent
chains, more robust autocorrelation windows, and hardware-optimized kernels.

The efficiency comparison is finite-lattice numerical methodology.

It does not prove the continuum Yang-Mills mass gap.
