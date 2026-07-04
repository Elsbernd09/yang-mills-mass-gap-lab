# Gauge Transformations and Numerical Gauge Invariance

## Purpose

Gauge symmetry is a defining mathematical structure of Yang-Mills theory.

The Yang-Mills Mass Gap Laboratory therefore includes an explicit lattice gauge-transformation engine and numerical gauge-invariance validation.

This phase is not merely an implementation feature. It tests whether the core lattice observables respect the symmetry structure they are intended to model.

## Local Gauge Transformations

A lattice gauge transformation assigns an SU(2) matrix G(x) to every lattice site x.

A positively oriented link transforms as

U_mu(x) -> G(x) U_mu(x) G(x + mu)^dagger.

The transformation depends independently on the lattice site, which is why the symmetry is local rather than merely global.

## Plaquette Transformation

The oriented plaquette is

U_mu,nu(x)
=
U_mu(x)
U_nu(x + mu)
U_mu(x + nu)^dagger
U_nu(x)^dagger.

After applying the local gauge transformation, the internal gauge factors cancel and the plaquette transforms by conjugation:

U_mu,nu(x)
->
G(x) U_mu,nu(x) G(x)^dagger.

Because trace is invariant under conjugation,

Tr(G U G^-1) = Tr(U),

the normalized real trace of the plaquette is gauge invariant.

## Wilson Action Invariance

For SU(2), the Wilson action is

S_W(U)
=
beta sum_p [1 - (1/2) Re Tr(U_p)].

Since each plaquette trace is gauge invariant,

S_W(U^G) = S_W(U).

The project validates this equality numerically up to floating-point tolerance.

## Wilson Loop Invariance

For a closed loop C, define the ordered link product

U(C) = product over links in C.

Under a local gauge transformation, the internal factors cancel around the closed path, leaving

U(C)
->
G(x_0) U(C) G(x_0)^dagger,

where x_0 is the base point.

Therefore

Tr(U(C))

is gauge invariant.

The project validates both individual rectangular Wilson loops and lattice-averaged Wilson loops.

## Numerical Validation Procedure

The gauge-invariance experiment performs the following steps:

1. Initialize a periodic SU(2) lattice.
2. Generate a nontrivial configuration with local staple-based Metropolis updates.
3. Measure gauge-invariant observables.
4. Generate an independent random SU(2) matrix G(x) at every lattice site.
5. Transform every lattice link.
6. Remeasure the observables.
7. Compare before and after values.
8. Report absolute and relative numerical differences.
9. Verify transformed links remain numerically inside SU(2).

## Validated Observables

The experiment currently compares:

- Wilson action
- action per plaquette
- average plaquette
- average plaquette action density
- individual 1 x 1 Wilson loop
- individual 2 x 2 Wilson loop
- average 1 x 1 Wilson loop
- average 2 x 2 Wilson loop

## Why This Matters

A finite-lattice gauge theory implementation should preserve the gauge-invariant quantities encoded by its mathematical definitions.

Explicit gauge-transformation testing is therefore a stronger structural validation than merely checking that simulations run without errors.

The project now checks not only numerical execution but also a central symmetry principle of Yang-Mills theory.

## Scientific Limitation

Numerical gauge invariance of a finite lattice implementation does not prove the existence of continuum quantum Yang-Mills theory and does not prove a positive mass gap.

It validates an essential structural property of the discretized computational framework.
