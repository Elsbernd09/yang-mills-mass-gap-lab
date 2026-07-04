# Dimension-General SU(2) Lattice Gauge Theory

## Purpose

The Yang–Mills Millennium Problem concerns quantum Yang–Mills theory in
four-dimensional spacetime. Many educational lattice projects begin in two
dimensions because the computations are cheaper and easier to visualize.

This project now includes dimension-general SU(2) lattice diagnostics across
2D, 3D, and 4D finite periodic hypercubic lattices.

## Hypercubic Lattices

A \(d\)-dimensional periodic hypercubic lattice has sites

\[
x = (x_1, x_2, \dots, x_d).
\]

Each site has \(d\) positive directed links, one in each coordinate direction.

The number of coordinate plaquette planes per site is

\[
\binom{d}{2}.
\]

Therefore:

\[
\binom{2}{2} = 1,
\quad
\binom{3}{2} = 3,
\quad
\binom{4}{2} = 6.
\]

This means the number of plaquette interactions grows rapidly with dimension.

## Why 4D Matters

The Clay problem asks for a rigorous construction of four-dimensional quantum
Yang–Mills theory and a proof of a positive mass gap. This project does not
solve that problem, but adding finite 4D lattice experiments makes the
computational framework more aligned with the actual mathematical setting.

## Measured Quantities

The dimension comparison experiment measures:

- number of sites,
- number of links,
- number of plaquettes,
- Wilson action density per site,
- Wilson action density per plaquette,
- average plaquette,
- Metropolis acceptance rate,
- runtime.

## Scientific Limitation

The 4D simulations in this repository are intentionally small. They are finite
lattice experiments, not continuum quantum field theory constructions. Serious
4D lattice gauge theory requires much larger lattices, optimized algorithms,
careful statistical analysis, and controlled limits.

## Importance for the Project

This phase shows that the codebase is not merely a two-dimensional toy. The
same mathematical architecture works in higher dimensions, including the
finite-lattice analogue of the four-dimensional setting relevant to Yang–Mills.
