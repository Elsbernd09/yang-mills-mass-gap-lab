# Performance Benchmarking

## Purpose

The Yang-Mills Mass Gap Laboratory includes performance benchmarks to make the computational cost of finite-lattice simulations transparent.

This matters because lattice gauge theory computations scale quickly with lattice volume and dimension.

## Measured Quantities

The performance benchmark records:

- lattice shape,
- dimension,
- number of sites,
- number of links,
- number of plaquettes,
- total runtime,
- average sweep time,
- link updates per second,
- plaquettes per second,
- mean Metropolis acceptance rate.

## Why Runtime Scaling Matters

A d-dimensional hypercubic lattice with V sites has:

- Vd positive directed links,
- V * C(d, 2) positive plaquettes.

As dimension increases, each site participates in more plaquette planes. For example:

- 2D has C(2, 2) = 1 plaquette plane per site,
- 3D has C(3, 2) = 3 plaquette planes per site,
- 4D has C(4, 2) = 6 plaquette planes per site.

This means higher-dimensional simulations become more expensive even when the number of sites is similar.

## Current Limitation

The current implementation is written for clarity, not maximum speed. It uses pure Python and NumPy, with dictionary-based lattice storage.

Future performance upgrades could include:

- local staple optimization refinements,
- array-based lattice storage,
- vectorized plaquette computations,
- Numba acceleration,
- C++ or Rust kernels,
- parallel independent chains,
- GPU acceleration,
- optimized SU(3) kernels.

## Scientific Limitation

Performance benchmarking does not prove Yang-Mills theory or the mass gap. It makes the computational framework more transparent and helps identify realistic scaling limits.
