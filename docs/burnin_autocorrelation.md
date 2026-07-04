# Burn-In and Autocorrelation Diagnostics

## Purpose

Monte Carlo simulations generate correlated samples. In lattice gauge theory,
one should not blindly trust every point produced by a Markov chain.

This project now includes basic diagnostics for:

- burn-in removal,
- autocorrelation functions,
- integrated autocorrelation time,
- effective sample size.

## Burn-In

The early part of a Markov chain is often affected by the initial condition.
For example, a cold-start lattice begins with every link equal to the identity.
This is an artificial starting point.

The first sweeps are discarded as burn-in:

\[
N_{\text{burn-in}}
\]

Only measurements after this period are used for observable summaries.

## Autocorrelation

Markov chain samples are not independent. The autocorrelation function measures
how strongly a time series is correlated with itself at later lags:

\[
\rho(t)
=
\frac{
\langle (X_s-\bar{X})(X_{s+t}-\bar{X}) \rangle
}{
\langle (X_s-\bar{X})^2 \rangle
}.
\]

If \(\rho(t)\) decays slowly, the simulation has fewer independent samples than
the raw number of measurements suggests.

## Integrated Autocorrelation Time

The integrated autocorrelation time is estimated as

\[
\tau_{\text{int}}
=
\frac{1}{2}
+
\sum_{t=1}^{T}
\rho(t).
\]

A larger value means stronger correlation between samples.

## Effective Sample Size

The effective sample size is roughly

\[
N_{\text{eff}}
=
\frac{N}{2\tau_{\text{int}}}.
\]

This estimates how many independent samples the correlated chain is worth.

## Why This Matters

Adding these diagnostics improves scientific honesty. The project no longer
only reports raw traces and bootstrap uncertainties; it also checks whether
the measurements are correlated.

## Current Limitation

The current autocorrelation diagnostics are simple. More serious studies should
use longer chains, binning/blocking analysis, jackknife errors, and careful
thermalization checks.
