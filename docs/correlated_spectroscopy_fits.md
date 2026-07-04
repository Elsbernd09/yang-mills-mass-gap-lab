# Covariance-Aware Correlated Spectroscopy Fits

## Purpose

Euclidean correlator values at different time separations are not statistically independent.

The values C(1), C(2), C(3), and later time points are estimated from the same gauge configurations.

A spectroscopy fit that minimizes an unweighted sum of squared residuals ignores this covariance structure.

The Yang-Mills Mass Gap Laboratory therefore includes covariance-aware correlated chi-squared fitting.

## Correlated Chi-Squared

Let C be the measured correlator vector and let f(theta) be a model correlator.

The correlated objective is

chi squared
=
[C - f(theta)] transpose
Sigma inverse
[C - f(theta)],

where Sigma is the covariance matrix of the estimated correlator.

Off-diagonal covariance entries describe statistical coupling between different Euclidean time points.

## Configuration Bootstrap

The project estimates correlator covariance by resampling complete gauge configurations.

Each measured configuration contributes a full Euclidean operator time series.

For one bootstrap replicate:

1. configuration indices are sampled with replacement,
2. the complete operator time-series rows are resampled,
3. a new connected Euclidean correlator is constructed,
4. the resulting correlator vector is stored.

Repeating this process produces an ensemble of bootstrap correlator vectors.

## Correlator Covariance

Given bootstrap correlator vectors C_b(t), the project estimates the sample covariance matrix across Euclidean time.

The corresponding correlation matrix is also calculated.

This makes the time-time dependence visible directly.

## Why Covariance Inversion Is Difficult

A finite ensemble can produce a covariance matrix with very small eigenvalues.

Strongly correlated Euclidean time points can make the covariance matrix poorly conditioned.

Blindly applying an ordinary matrix inverse can amplify statistical noise.

The project therefore uses regularization.

## Diagonal Shrinkage

The covariance matrix can be shrunk toward its diagonal:

Sigma_shrunk
=
(1 - alpha) Sigma
+
alpha diagonal(Sigma).

At alpha equal to zero, the original covariance is retained.

At alpha equal to one, all estimated cross-covariances are removed.

Intermediate values reduce the numerical influence of uncertain off-diagonal covariance estimates without completely discarding them.

## Eigenvalue-Truncated Pseudoinverse

The symmetrized covariance matrix is diagonalized.

Only sufficiently positive eigenmodes are retained.

The threshold combines:

- an absolute eigenvalue cutoff,
- a cutoff relative to the largest positive covariance eigenvalue.

The covariance pseudoinverse is constructed only in the retained numerical subspace.

The project records:

- retained covariance rank,
- covariance eigenvalue cutoff,
- retained condition number,
- covariance-pseudoinverse projection error.

## Periodic Cosh Model

The fitted single-state periodic model is

C(t)
=
A [
exp(-m t)
+
exp(-m (T - t))
].

The amplitude A and lattice mass m are constrained to remain positive by optimizing their logarithms.

## Parameter Covariance

After the correlated fit, the project evaluates the model Jacobian with respect to amplitude and mass.

A local information matrix is constructed:

J transpose
Sigma inverse
J.

Its pseudoinverse provides an approximate parameter covariance matrix.

The reported amplitude and mass errors are derived from its diagonal.

These are local linearized fit uncertainties.

They are not a substitute for a complete resampling analysis of every fitting choice.

## Effective Degrees of Freedom

When covariance eigenmodes are truncated, the numerical dimension of the fitted residual space is the retained covariance rank.

The project defines an effective degrees-of-freedom diagnostic as

retained covariance rank
minus
number of fitted parameters.

For the two-parameter periodic cosh model, two parameters are subtracted.

The reduced chi-squared diagnostic is then

chi squared
divided by
effective degrees of freedom

when the effective degrees of freedom is positive.

## Fit-Window Scan

The experiment scans positive correlator fit windows.

Correlated candidate fits are ranked by the absolute distance of reduced chi-squared from one, with longer windows preferred as a secondary criterion.

This is a heuristic model-selection diagnostic.

It does not remove the need for human scrutiny of:

- fit-window stability,
- excited-state contamination,
- covariance uncertainty,
- operator dependence,
- volume dependence.

## Uncorrelated Versus Correlated Fits

The experiment reports both:

- an ordinary unweighted periodic cosh fit,
- a covariance-aware correlated periodic cosh fit.

The purpose is not to assume that one must always return a dramatically different mass.

The purpose is to measure whether accounting for time-time covariance changes:

- the selected fit window,
- fitted mass,
- estimated uncertainty,
- fit quality diagnostics.

## Scientific Limitations

The covariance matrix is estimated from a finite configuration ensemble.

Bootstrap replicates derived from one Markov chain do not automatically eliminate all autocorrelation concerns.

The current experiment also does not yet combine:

- GEVP principal-correlator bootstrap,
- covariance-aware principal-correlator fits,
- multi-exponential Bayesian models,
- continuum extrapolation,
- infinite-volume extrapolation.

The covariance-aware fit improves finite-lattice numerical methodology.

It does not prove the continuum Yang-Mills mass gap.
