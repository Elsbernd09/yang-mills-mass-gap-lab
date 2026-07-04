# Lattice Gauge Theory Notes

## Why Lattice Gauge Theory?

The Yang–Mills Millennium Problem concerns the rigorous construction of quantum Yang–Mills theory in four-dimensional continuum spacetime. This is beyond the scope of a simple computational project.

However, lattice gauge theory provides a powerful finite-dimensional approximation. Instead of working directly with continuous spacetime, we replace spacetime with a discrete grid. Gauge fields are assigned to links between neighboring lattice sites, and curvature is represented by plaquette variables around elementary squares.

## Sites, Links, and Plaquettes

A lattice site is a coordinate such as

\[
x = (x_1, x_2, \dots, x_d).
\]

A link variable is an SU(2) matrix

\[
U_\mu(x),
\]

which lives on the directed edge from \(x\) to \(x+\hat{\mu}\).

A plaquette is the ordered product

\[
U_{\mu\nu}(x)
=
U_\mu(x)
U_\nu(x+\hat{\mu})
U_\mu(x+\hat{\nu})^\dagger
U_\nu(x)^\dagger.
\]

This is the lattice analogue of curvature.

## Wilson Action

For SU(2), the Wilson action is

\[
S_W(U)
=
\beta
\sum_p
\left(
1 - \frac{1}{2}\operatorname{Re}\operatorname{Tr}(U_p)
\right).
\]

A cold-start lattice, where every link is the identity matrix, has zero Wilson action because every plaquette is also the identity.

A random-start lattice generally has positive Wilson action.

## Scientific Interpretation

The Wilson action is central to lattice Yang–Mills theory. It allows finite-dimensional gauge field configurations to be weighted probabilistically in Monte Carlo simulations. Later phases of this project will use Metropolis-Hastings sampling to generate ensembles of gauge configurations and measure gauge-invariant observables such as Wilson loops and correlation functions.

## Limitation

This lattice model is a numerical and finite-dimensional approximation. It does not prove the existence of continuum quantum Yang–Mills theory and does not prove the mass gap.
