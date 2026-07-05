# Ensemble Wilson Loops and Block-Bootstrap Creutz Ratios

## Purpose

The original Creutz-ratio implementation established the finite-lattice
observable and validated its mathematical definition on an exact synthetic
area law.

The earlier exploratory experiment relied too heavily on loop measurements
from a final configuration.

This phase replaces that workflow with post-thermalization configuration-level
Wilson-loop ensembles and correlation-aware nonlinear resampling.

## Wilson-Loop Measurement Basis

A WilsonLoopBasis defines an ordered collection of rectangular loop shapes.

For example, a maximum size of four by four produces all loops

W(R,T)

for

R from one through four

and

T from one through four.

Every measured configuration produces one complete vector of Wilson-loop
observables.

## Configuration-Level Measurement Matrix

The measured data have shape

configurations
by
Wilson-loop observables.

One row therefore contains all loop measurements obtained from the same gauge
configuration.

The cross-loop dependence within that row is preserved during resampling.

## Why Joint Resampling Matters

A Creutz ratio depends on four Wilson-loop means:

W(R,T),

W(R minus one,T minus one),

W(R,T minus one),

and

W(R minus one,T).

These loop observables are measured on the same gauge configurations and can
be strongly correlated.

Treating their uncertainties as independent would discard this statistical
structure.

The project therefore resamples complete configuration rows.

## Circular Block Bootstrap

Successive Markov-chain configurations may remain autocorrelated.

The implementation uses circular moving-block bootstrap indices.

Consecutive configuration blocks are sampled with replacement.

The final sampled index sequence is truncated to the original number of
configurations.

The experiment uses the measured W(1,1) integrated autocorrelation time to
construct a heuristic block size

ceiling of two times tau integrated.

This is a practical diagnostic choice rather than a universal optimal block
selection theorem.

## Nonlinear Creutz Reconstruction

For every bootstrap replicate:

1. complete configuration rows are block resampled,
2. every Wilson loop is averaged over the resampled configurations,
3. each supported Creutz ratio is reconstructed from the bootstrap loop means.

The nonlinear logarithmic ratio is therefore evaluated inside the bootstrap.

## Creutz Ratio

The implemented observable is

chi(R,T)

equals minus the logarithm of

W(R,T) times W(R minus one,T minus one)

divided by

W(R,T minus one) times W(R minus one,T).

For an exact synthetic area law

W(R,T)
equals
exp minus sigma R T,

the Creutz ratio is exactly sigma.

The test suite verifies this property.

## Invalid Bootstrap Replicates

The logarithm requires a positive finite ratio.

A bootstrap replicate that does not support the logarithm produces NaN for
that Creutz point.

The implementation does not apply absolute values.

It does not force negative loop combinations to become positive.

For every Creutz point, the project reports:

- finite bootstrap replicate count,
- requested bootstrap replicate count,
- valid bootstrap fraction.

## Bootstrap Summaries

For every supported Creutz shape, the project reports:

- central ratio from full-ensemble loop means,
- bootstrap mean,
- bootstrap standard error,
- bootstrap median,
- lower 95 percent percentile bound,
- upper 95 percent percentile bound,
- finite replicate count,
- valid fraction.

## Square Creutz Diagnostics

Square values with R equal to T are extracted when their bootstrap valid
fraction exceeds a selected threshold.

Their variation with loop size is used as a string-tension-style plateau
diagnostic.

The arithmetic mean across valid square ratios may be displayed as a compact
summary.

It is not automatically interpreted as a continuum string tension.

## Wilson-Loop Noise Growth

Large Wilson loops can become statistically difficult because their expected
values may become small while relative uncertainty grows.

The experiment reports each loop mean, bootstrap standard error, and relative
standard error.

This makes loss of signal directly visible.

## Scientific Limitations

The current experiment uses small finite SU(2) lattices.

The block-size rule is heuristic.

The integrated-autocorrelation estimator uses the current positive-sequence
diagnostic.

Large-loop signal degradation can make nonlinear ratio distributions highly
non-Gaussian.

The analysis does not include multilevel variance reduction.

The analysis does not perform continuum extrapolation.

The analysis does not perform infinite-volume extrapolation.

The square Creutz diagnostic is not claimed to be a physical continuum string
tension measurement.

This phase provides correlation-aware finite-lattice Wilson-loop and
Creutz-ratio uncertainty analysis.

It does not prove the Yang-Mills mass gap.
