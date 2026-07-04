# Correlated Fits of Matched GEVP Principal Correlators

## Purpose

The generalized eigenvalue pipeline produces matched variational principal correlators

lambda_n(t, t0).

The full GEVP bootstrap produces statistical samples of these principal correlators.

The project therefore has enough information to estimate the Euclidean time covariance of each matched variational state.

This phase combines variational spectroscopy with covariance-aware correlated fitting.

## Principal-Correlator Normalization

For a retained GEVP state,

lambda_n(t0, t0)
=
1

up to numerical floating-point error.

A periodic single-state model can therefore be normalized at the reference time.

The model used in this phase is

lambda(t)
=
[
exp(-m t)
+
exp(-m (T - t))
]
divided by
[
exp(-m t0)
+
exp(-m (T - t0))
].

The model contains one free positive parameter:

m.

No free amplitude parameter is required.

## Why Removing the Amplitude Matters

An ordinary periodic cosh fit uses both an amplitude and a mass.

The GEVP normalization already fixes the principal correlator at the reference time.

Introducing an independent free amplitude would ignore that exact variational normalization condition.

The normalized principal model therefore uses one fitted parameter.

## Bootstrap Time-Time Covariance

For each matched variational state, the bootstrap principal samples form an array

bootstrap replicate
by
Euclidean time.

The project estimates the covariance matrix of this array.

The covariance captures statistical dependence between different Euclidean time lags of the same matched principal state.

## Covariance Regularization

The principal-correlator covariance is restricted to a candidate fit window.

Diagonal shrinkage is applied.

The resulting matrix is diagonalized and small covariance eigenmodes can be removed.

The fit objective is evaluated using the regularized covariance pseudoinverse.

## Correlated Principal-State Chi-Squared

For a principal correlator lambda and normalized model f(m), the objective is

chi squared
=
[lambda - f(m)] transpose
Sigma inverse
[lambda - f(m)].

The mass is constrained to remain positive.

## One-Parameter Curvature Uncertainty

The derivative of the normalized periodic model with respect to mass is calculated analytically.

The local information value is

J transpose
Sigma inverse
J.

For a one-parameter model, the inverse of this value provides a local linearized mass variance when the information is positive.

This produces the local curvature mass uncertainty.

## Effective Degrees of Freedom

The regularized covariance pseudoinverse may retain fewer modes than the number of Euclidean time points in the fit window.

The effective residual dimension is therefore taken as the retained covariance rank.

One fitted mass parameter is subtracted.

The effective degrees of freedom are

retained covariance rank minus one.

## Fit-Window Scan

Candidate fit windows are scanned within the first half of the periodic temporal extent.

Only windows containing positive principal-correlator values are accepted.

Successful fits are ranked by:

1. closeness of reduced chi-squared to one,
2. longer fit-window length,
3. smaller local mass uncertainty.

This ranking is heuristic.

It does not prove ground-state saturation.

## Fixed-Window Bootstrap Refitting

Once the central fit selects a window, each matched bootstrap principal-correlator replicate is refit in the same window.

The covariance matrix and fit window remain fixed.

This produces a bootstrap mass distribution.

The project reports:

- accepted bootstrap fits,
- rejected bootstrap fits,
- bootstrap mean mass,
- bootstrap standard error,
- bootstrap median mass,
- bootstrap 95 percent percentile interval.

## Why the Window Is Fixed During Bootstrap Refitting

Selecting a new best fit window independently inside every bootstrap replicate would combine parameter uncertainty with a complicated data-dependent model-selection process.

The current implementation instead conditions on the central selected window.

This makes the bootstrap distribution easier to interpret.

It does not quantify fit-window-selection uncertainty.

## State Matching Comes Before Fitting

Bootstrap state labels are not trusted by raw eigenvalue order.

The full GEVP bootstrap first matches every accepted bootstrap variational state to the central state using metric-normalized operator-space overlaps.

Only after this matching is principal-state covariance estimated and mass fitting performed.

## Rejected Fits Are Reported

A matched variational state can fail to produce a supported mass estimate.

Reasons include:

- negative principal-correlator values,
- insufficient positive fit-window points,
- unstable covariance geometry,
- optimization failure.

The experiment explicitly reports when no valid correlated principal-state fit is available.

No positive value is forced using absolute values.

No mass estimate is fabricated.

## Scientific Limitations

The principal-state model remains a single-state periodic model.

Excited-state contamination may remain.

The operator basis is small and generated primarily by smearing depth.

The central-window-conditioned bootstrap does not include window-selection uncertainty.

The selected covariance shrinkage strength is fixed rather than inferred.

The project does not yet perform:

- multi-exponential principal-state fits,
- Bayesian prior analysis,
- continuum extrapolation,
- infinite-volume extrapolation,
- physical scale setting.

This phase provides covariance-aware finite-lattice variational mass diagnostics.

It does not prove the continuum Yang-Mills mass gap.
