# Gauge-Invariant Scalar Glueball-Style Correlators

## Purpose

The Yang-Mills Mass Gap Laboratory originally estimated mass-gap-style behavior from a simple plaquette time-series autocorrelation.

That approach was useful as a numerical diagnostic, but it did not yet implement a configuration-ensemble Euclidean correlator of a gauge-invariant composite operator.

This phase introduces a more physically meaningful finite-lattice spectroscopy pipeline.

The workflow is:

gauge configurations
-> scalar gauge-invariant operator O(t)
-> ensemble Euclidean correlator
-> connected correlator
-> periodic mass-analysis pipeline.

## Scalar Glueball-Style Operator

The project constructs a simple scalar operator from purely spatial plaquette action densities.

Choose one lattice direction as Euclidean time.

At fixed Euclidean time t, the operator averages plaquette action density over all plaquette planes that do not contain the time direction.

Conceptually,

O(t)
=
spatial average of
[1 - (1/2) Re Tr(U_p)].

Because plaquette traces are gauge invariant, this operator is gauge invariant.

## Why This Is Called Glueball-Style

In pure non-Abelian gauge theory, glueball spectroscopy studies gauge-invariant composite operators constructed from closed gauge-field loops.

A full glueball calculation generally requires:

- operator bases,
- lattice rotational symmetry classification,
- parity and charge-conjugation channels,
- smearing or blocking,
- variational analysis,
- large ensembles,
- continuum and infinite-volume studies.

The current operator is therefore described carefully as a scalar glueball-style operator.

It is inspired by the scalar 0++ channel but is not presented as a full professional glueball spectroscopy calculation.

## Euclidean Time-Slice Operator

For every Euclidean time slice t, the project computes O(t).

One measured gauge configuration therefore produces the vector:

O(0), O(1), ..., O(T-1).

The experiment repeats this across many gauge configurations drawn from a post-thermalization Markov chain.

The resulting dataset has shape:

number of configurations x temporal extent.

## Raw Correlator

For one configuration, the periodic correlation is

C_raw(tau)
=
(1/T) sum_t O(t) O(t + tau),

where time indexing is periodic.

The project then averages this quantity across configurations.

## Connected Correlator

The connected Euclidean correlator is

C(tau)
=
<O(t) O(t + tau)> - <O>^2.

Subtracting the disconnected contribution is important because the scalar operator can have a nonzero expectation value.

## Periodic Symmetry

On a periodic Euclidean lattice, the correlator should satisfy the symmetry

C(tau) = C(T - tau)

up to finite-sample and floating-point effects.

The experiment explicitly computes the maximum numerical symmetry error.

## Gauge Invariance

The scalar operator is built from plaquette traces.

The project's unit tests apply a random local gauge transformation to a nontrivial lattice configuration and verify that the full scalar operator time series is numerically unchanged.

## Why This Improves the Mass-Gap Pipeline

The earlier pipeline was:

single lattice
-> plaquette time series
-> autocorrelation
-> effective mass.

The new pipeline is:

many thermalized gauge configurations
-> gauge-invariant O(t)
-> ensemble <O(t) O(0)>
-> connected Euclidean correlator
-> periodic/cosh mass analysis.

This is a materially stronger spectroscopy architecture.

## Scientific Limitation

The current implementation remains exploratory.

It does not establish:

- a continuum Yang-Mills theory,
- an infinite-volume limit,
- a rigorous Hilbert space,
- a spectral theorem for the simulation,
- a physical glueball mass,
- or the Clay Millennium mass gap.

The finite lattice is small, the operator basis contains only a simple scalar plaquette operator, and the update algorithm remains Metropolis-based.

The next analysis phase will add periodic cosh fits and effective mass plateau diagnostics.
