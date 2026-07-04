# Validation Suite

## Purpose

The Yang–Mills Mass Gap Laboratory includes a validation suite to check the
internal consistency of the finite-lattice computational framework.

The validation suite does not prove the Yang–Mills mass gap. Instead, it checks
that the implemented mathematical and numerical objects behave as expected.

## Checks Included

The validation suite currently checks:

1. SU(2) identity and random matrix generation.
2. Cold-start plaquette behavior.
3. Cold-start Wilson action.
4. Cold-start Wilson loop values.
5. SU(2) preservation under Metropolis sweeps.
6. Creutz ratio recovery on exact synthetic area-law data.
7. Dimension-dependent lattice count formulas.
8. SU(3) identity and random matrix generation.
9. Gell-Mann matrix Hermitian, traceless, and normalization properties.

## Cold-Start Validation

A cold-start SU(2) lattice assigns the identity matrix to every link:

\[
U_\mu(x) = I.
\]

Therefore every plaquette is also the identity:

\[
U_{\mu\nu}(x) = I.
\]

This implies:

\[
\frac{1}{2}\operatorname{Re}\operatorname{Tr}(U_{\mu\nu}) = 1,
\]

and the Wilson action is zero.

## Wilson Loop Validation

On a cold-start lattice, every rectangular Wilson loop is the trace of a product
of identity matrices. Therefore:

\[
W(R,T) = 1.
\]

The validation suite checks this for several loop sizes.

## Creutz Ratio Validation

For synthetic Wilson loop data obeying an exact area law,

\[
W(R,T) = e^{-\sigma RT},
\]

the Creutz ratio should recover:

\[
\chi(R,T) = \sigma.
\]

The validation suite verifies this identity numerically.

## Dimension Count Validation

For a \(d\)-dimensional hypercubic lattice with \(V\) sites:

- number of positive directed links is \(Vd\),
- number of positive plaquettes is \(V\binom{d}{2}\).

The validation suite checks this in 2D, 3D, and 4D.

## SU(3) Validation

The SU(3) extension is validated by checking:

- identity is in SU(3),
- random generated matrices are in SU(3),
- Gell-Mann matrices are Hermitian,
- Gell-Mann matrices are traceless,
- \(\operatorname{Tr}(\lambda_i\lambda_j)=2\delta_{ij}\).

## Importance

This validation layer makes the repository more trustworthy. It shows that the
project is not only running simulations but also checking mathematical
consistency across its core objects.
