# Beta Scans and Finite-Size Scaling

## Purpose

A serious lattice gauge theory project should not report observables from only
one lattice size and one coupling parameter. It should study how observables
change as simulation parameters vary.

This project now includes scans over:

- the lattice coupling parameter \(\beta\),
- the lattice size \(L\).

## Beta in SU(2) Lattice Gauge Theory

For SU(2), the lattice coupling is often written as

\[
\beta = \frac{4}{g^2},
\]

where \(g\) is the gauge coupling.

Larger \(\beta\) corresponds to weaker coupling. Smaller \(\beta\) corresponds
to stronger coupling.

## Measured Observables

For each beta/size setting, the project measures:

- final Wilson action,
- final average plaquette,
- mean Metropolis acceptance rate.

Each point is estimated from multiple independent Markov chains, and bootstrap
uncertainty is reported.

## Finite-Size Scaling

Finite-size scaling asks how observables change as the lattice size changes.

For example, if \(L\) is increased from \(4\) to \(6\) to \(8\), one can compare:

\[
\langle P \rangle_L
\]

where \(\langle P \rangle_L\) is the average plaquette on an \(L \times L\)
lattice.

A true continuum or infinite-volume study would require much larger lattices
and careful scaling limits. This project implements the framework needed to
begin that kind of analysis in miniature.

## Scientific Limitation

These beta and finite-size scans are finite numerical experiments. They do not
prove the Yang–Mills mass gap, continuum existence, or confinement. They are
designed to make the computational exploration more systematic and honest.

## Future Improvements

Future versions should include:

- larger lattices,
- longer thermalization,
- separate burn-in and measurement phases,
- autocorrelation time estimation,
- blocked bootstrap uncertainty,
- scans in three and four dimensions,
- comparison to known lattice gauge theory benchmarks.
