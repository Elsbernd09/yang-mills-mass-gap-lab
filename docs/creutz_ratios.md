# Creutz Ratios and String-Tension-Style Diagnostics

## Purpose

Wilson loops are central observables in lattice gauge theory. They are often
used to study confinement-style behavior.

This project now includes Creutz ratios, which are constructed from rectangular
Wilson loops and are commonly used as finite-lattice diagnostics for
string-tension-like behavior.

## Wilson Loop Decay

A rectangular Wilson loop may have the rough form

\[
W(R,T)
\sim
\exp(-\sigma R T - \mu(2R+2T)),
\]

where:

- \(R T\) is the loop area,
- \(2R+2T\) is the loop perimeter,
- \(\sigma\) is a string-tension-like quantity,
- \(\mu\) is a perimeter contribution.

## Creutz Ratio

The Creutz ratio is defined by

\[
\chi(R,T)
=
-\log
\left(
\frac{
W(R,T)W(R-1,T-1)
}{
W(R,T-1)W(R-1,T)
}
\right).
\]

This combination approximately cancels perimeter terms. In idealized
area-law behavior,

\[
\chi(R,T) \approx \sigma.
\]

## Why This Matters

Adding Creutz ratios gives the project a second numerical route toward
confinement-style diagnostics:

1. correlation decay and effective mass estimation,
2. Wilson loop decay and Creutz-ratio string-tension estimation.

This makes the project more closely aligned with standard lattice gauge theory
observables.

## Scientific Limitation

The Creutz ratio measurements in this repository are finite-lattice numerical
diagnostics. They do not prove confinement, continuum Yang–Mills existence, or
the Yang–Mills mass gap.

The current simulations are small and exploratory. Serious lattice studies
would require larger lattices, longer Markov chains, thermalization checks,
autocorrelation analysis, improved update algorithms, and careful finite-size
scaling.
