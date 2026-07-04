# Periodic Euclidean Spectroscopy and Effective Mass Analysis

## Purpose

The Yang-Mills Mass Gap Laboratory now includes a finite-lattice spectroscopy analysis layer for gauge-invariant scalar glueball-style correlators.

The goal is to move beyond a simple exponential autocorrelation estimator and analyze the periodic Euclidean geometry of the lattice explicitly.

The spectroscopy pipeline is:

gauge configurations
-> scalar gauge-invariant O(t)
-> connected Euclidean correlator C(t)
-> periodic cosh fitting
-> arccosh effective mass
-> plateau diagnostics
-> configuration-level bootstrap uncertainty.

## Periodic Euclidean Correlator

For a periodic lattice with temporal extent T, a single-state contribution has the form

C(t)
approximately equals
A [exp(-m t) + exp(-m (T - t))].

The second exponential represents propagation around the periodic temporal boundary.

The model is symmetric under

t -> T - t.

## Periodic Cosh Interpretation

The periodic two-exponential form can be rewritten as a cosh centered at T/2.

Therefore a cosh-like model is more appropriate than a single forward exponential when the lattice has periodic Euclidean time.

The project directly evaluates the two-exponential periodic form.

## Arccosh Effective Mass

For a single periodic state,

C(t - 1) + C(t + 1)
=
2 cosh(m) C(t).

Therefore

m_eff(t)
=
arccosh(
[C(t - 1) + C(t + 1)]
/
[2 C(t)]
).

The implementation reports NaN when the estimator is numerically invalid.

It does not take an absolute value or otherwise force an invalid correlator into a positive mass.

## Effective-Mass Plateaus

In an ideal spectroscopy calculation, an effective-mass region may become approximately constant when a dominant state controls the correlator.

The project scans contiguous finite positive effective-mass windows.

Each candidate window records:

- starting lag,
- stopping lag,
- number of points,
- mean mass,
- standard deviation,
- relative spread.

A heuristic stability score favors flatter and longer windows.

This is a diagnostic only.

The lowest-scoring window is not automatically a proof that the asymptotic ground state has been isolated.

## Periodic Cosh Fit

The connected correlator is fitted over candidate positive time windows.

The fit model is

C(t)
=
A [exp(-m t) + exp(-m (T - t))].

The fitted parameters are:

- amplitude A,
- lattice mass m.

Positive parameter bounds are imposed.

Candidate windows that contain nonpositive correlator values are rejected.

## Fit-Window Selection Limitation

The current experiment compares candidate fit windows using mean squared residuals and window length.

This is not yet a covariance-aware correlated chi-squared analysis.

A more advanced spectroscopy calculation should estimate the correlator covariance matrix and perform correlated fits with regularization where necessary.

## Configuration-Level Bootstrap

The project bootstraps entire measured configurations.

Each configuration contributes a complete Euclidean operator time series.

A bootstrap replicate therefore resamples rows of the configuration-by-time operator ensemble.

For every replicate:

1. configurations are sampled with replacement,
2. a new connected correlator is constructed,
3. the selected periodic cosh window is fitted,
4. successful positive mass estimates are recorded.

The resulting mass distribution provides:

- bootstrap mean,
- bootstrap standard error,
- percentile confidence interval,
- successful fit count.

## Why Configuration-Level Resampling Matters

Resampling individual correlator time points would destroy the internal Euclidean structure of one measured configuration.

The project instead resamples complete configuration measurements.

This preserves the time-slice vector associated with each configuration.

## Why This Improves the Project

The original mass-gap-style estimator used a simple logarithmic ratio of an exploratory autocorrelation.

The current pipeline now includes:

- gauge-invariant scalar composite operators,
- multi-configuration measurement ensembles,
- connected Euclidean correlators,
- periodic boundary geometry,
- cosh-style correlator fitting,
- arccosh effective masses,
- plateau-window diagnostics,
- configuration-level bootstrap mass distributions.

This is a substantially stronger finite-lattice spectroscopy framework.

## Scientific Limitations

The extracted quantity is a finite-lattice scalar glueball-style mass diagnostic in lattice units.

It is not presented as:

- a continuum physical glueball mass,
- a precision Yang-Mills spectrum calculation,
- evidence sufficient to establish a rigorous spectral gap,
- or a solution to the Clay Millennium Problem.

Important remaining limitations include:

- small lattice volumes,
- one simple scalar operator,
- no smearing,
- no variational operator basis,
- no generalized eigenvalue problem,
- Metropolis sampling,
- limited scale setting,
- no controlled continuum extrapolation,
- no controlled infinite-volume extrapolation,
- no correlated covariance-matrix fit.

The project records these limitations explicitly rather than hiding them.
