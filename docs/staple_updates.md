# Local Staple-Based Metropolis Updates

## Purpose

Early versions of this project computed the full Wilson action before and after
every proposed link update. That approach is mathematically transparent but
computationally inefficient.

In lattice gauge theory, a single link affects only the plaquettes that touch
that link. Therefore, the action difference for a proposed update can be
computed locally using the staple.

## The Staple

For a link \(U_\mu(x)\), the staple is the sum of the neighboring paths that
complete the plaquettes containing that link.

For each direction \(\nu \ne \mu\), there is a forward staple and a backward
staple. The total staple is the sum over all such neighboring paths.

## Local Wilson Contribution

For SU(2), the link-dependent part of the Wilson action can be written as

\[
S_{\text{local}}
=
-\frac{\beta}{2}
\operatorname{Re}
\operatorname{Tr}
\left(
U_\mu(x) V_\mu(x)
\right),
\]

where \(V_\mu(x)\) is the staple.

For a proposal \(U_\mu(x)' = R U_\mu(x)\), the local action difference is

\[
\Delta S
=
-\frac{\beta}{2}
\operatorname{Re}
\operatorname{Tr}
\left(
U_\mu(x)' V_\mu(x)
\right)
+
\frac{\beta}{2}
\operatorname{Re}
\operatorname{Tr}
\left(
U_\mu(x) V_\mu(x)
\right).
\]

The Metropolis acceptance probability is

\[
\min(1, e^{-\Delta S}).
\]

## Why This Matters

This upgrade makes the simulation closer to real lattice gauge theory code.
Instead of repeatedly recomputing the entire action, the sampler uses the local
geometry of plaquettes around a link.

## Limitation

This is still a simple Metropolis sampler. More advanced lattice gauge theory
codes often use heatbath updates, overrelaxation, improved actions, and careful
autocorrelation analysis.
