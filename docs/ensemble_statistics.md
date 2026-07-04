# Ensemble Averaging and Bootstrap Uncertainty

## Purpose

A single Monte Carlo trajectory is not enough for credible numerical physics.
Lattice gauge theory studies typically use many configurations, often generated
from long Markov chains, and report observables with uncertainty estimates.

This project now includes an ensemble layer for running multiple independent
SU(2) lattice simulations and estimating uncertainty with bootstrap resampling.

## Independent Chains

An independent chain is a Markov chain initialized with its own random seed.
For each chain, the project records:

- final Wilson action,
- final average plaquette,
- mean Metropolis acceptance rate.

These summaries are then treated as an ensemble of measurements.

## Bootstrap Mean

Given measurements

\[
x_1, x_2, \dots, x_n,
\]

the bootstrap estimates uncertainty by repeatedly resampling these values with
replacement and computing the mean of each resample.

This produces a bootstrap distribution of the mean.

The project reports:

\[
\bar{x} \pm \mathrm{SE},
\]

where \(\mathrm{SE}\) is the standard deviation of bootstrap sample means.

It also reports an approximate confidence interval from bootstrap quantiles.

## Why This Improves Scientific Credibility

Earlier experiments showed one simulation trajectory. The ensemble layer gives a
more honest numerical picture by asking:

- How much do results change across random seeds?
- What is the uncertainty of the measured average?
- Are the results stable across independent chains?

This is closer to how computational physics results are reported.

## Current Limitation

The current ensemble method is intentionally simple. More advanced studies
should include:

- thermalization/burn-in removal,
- autocorrelation time estimation,
- blocked bootstrap or jackknife analysis,
- longer chains,
- larger lattices,
- finite-size scaling,
- systematic scans over \(\beta\),
- independent configuration storage.

## Scientific Limitation

Bootstrap uncertainty on finite simulations does not prove the Yang–Mills mass
gap. It improves numerical honesty within the finite lattice model.
