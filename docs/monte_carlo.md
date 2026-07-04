# Metropolis-Hastings Sampling for SU(2) Lattice Gauge Theory

## Purpose

The Wilson action assigns an energy-like value to each lattice gauge field configuration. To study lattice Yang–Mills numerically, we need to sample configurations according to the Boltzmann-type weight

\[
P(U) \propto e^{-S_W(U)}.
\]

The Metropolis-Hastings algorithm provides a simple way to generate approximate samples from this distribution.

## Single-Link Proposal

For a selected link variable \(U_\mu(x)\), the sampler proposes

\[
U_\mu(x)' = R U_\mu(x),
\]

where \(R\) is a small random SU(2) matrix close to the identity.

The update is accepted with probability

\[
\min(1, e^{-\Delta S}),
\]

where

\[
\Delta S = S_W(U') - S_W(U).
\]

If the proposal is rejected, the old link variable is restored.

## Sweep

A sweep attempts one update for every positive directed link in the lattice.

For a lattice with \(V\) sites and dimension \(d\), the number of link updates per sweep is

\[
Vd.
\]

## Diagnostics

This project records:

- Wilson action
- acceptance rate
- average plaquette

These diagnostics are used to check whether the simulation evolves smoothly and whether the proposal size is reasonable.

## Current Limitation

This implementation recomputes the full Wilson action after every proposed link update. That is simple and transparent, but inefficient. A future version should compute the local action difference using staples, which is the standard lattice gauge theory optimization.

## Scientific Limitation

This Monte Carlo simulation provides numerical exploration of finite lattice SU(2) Yang–Mills theory. It does not constitute a proof of continuum Yang–Mills existence or a proof of a positive mass gap.
