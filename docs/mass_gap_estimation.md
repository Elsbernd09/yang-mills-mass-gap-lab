# Numerical Mass-Gap-Style Estimation

## Purpose

The Yang–Mills mass gap problem asks for a rigorous construction of quantum Yang–Mills theory in four-dimensional spacetime and a proof that the theory has a positive mass gap.

This project does not prove the mass gap. Instead, it implements finite-lattice numerical diagnostics inspired by the mass gap idea.

## Correlation Decay

In Euclidean quantum field theory, particle masses are related to the decay of correlation functions. A typical heuristic form is

\[
C(t) \sim A e^{-mt},
\]

where:

- \(C(t)\) is a correlation function,
- \(A\) is an amplitude,
- \(m\) is a mass scale,
- \(t\) is Euclidean time separation.

If \(m > 0\), the correlation decays exponentially.

## Effective Mass

Given a correlation function, an effective mass can be estimated by

\[
m_{\text{eff}}(t)
=
\log
\left(
\frac{C(t)}{C(t+1)}
\right).
\]

If the correlation is close to a single exponential decay, the effective mass may flatten into a plateau.

## Plaquette-Based Observable

This project currently constructs a simple plaquette-based time-slice observable. For each time index, it averages plaquette action density over that slice.

The connected autocorrelation is then computed as

\[
C(\tau)
=
\langle
(O(t)-\bar{O})(O(t+\tau)-\bar{O})
\rangle_t.
\]

## Scientific Limitation

The current implementation is a small finite-lattice diagnostic. It does not establish:

- continuum existence,
- reflection positivity,
- infinite-volume limits,
- rigorous Hilbert space construction,
- spectral gap proof,
- or the Clay Millennium Yang–Mills result.

It is a computational exploration of mass-gap-style behavior, not a theorem.

## Future Improvements

Future versions should add:

- larger lattices,
- 3D and 4D simulations,
- improved local Metropolis updates using staples,
- heatbath algorithms,
- better glueball operators,
- ensemble averaging across independent Markov chains,
- autocorrelation time analysis,
- jackknife or bootstrap uncertainty estimates,
- finite-size scaling studies.
