# Wilson Loops and Confinement-Style Observables

## Purpose

Wilson loops are central gauge-invariant observables in lattice gauge theory. They measure the ordered product of link variables around a closed loop.

For a rectangular loop \(C\), the Wilson loop is

\[
W(C)
=
\frac{1}{2}
\operatorname{Re}
\operatorname{Tr}
\left(
\prod_{\ell \in C} U_\ell
\right).
\]

Because the trace of the closed loop product is gauge invariant, Wilson loops are physically meaningful observables.

## Rectangular Wilson Loops

For a rectangle of width \(R\) and height \(T\), the Wilson loop is often written as

\[
W(R,T).
\]

In lattice gauge theory, one studies how \(W(R,T)\) decays as the loop size grows.

## Area-Law Heuristic

A confinement-style area law has the rough form

\[
W(R,T)
\sim
e^{-\sigma R T},
\]

where \(\sigma\) is a string-tension-like parameter.

Taking logarithms gives

\[
\log W(R,T)
\sim
-\sigma R T.
\]

This project includes exploratory Wilson loop measurements and area-decay plots. These are numerical diagnostics, not rigorous proofs.

## Relationship to Yang–Mills

The Yang–Mills mass gap problem is deeply connected to non-Abelian quantum gauge theory. Wilson loops are one way physicists and mathematicians study confinement behavior in lattice approximations.

## Limitation

Finite lattice Wilson loop measurements do not prove the Clay Millennium Problem. They provide numerical and conceptual evidence within a discretized model.
