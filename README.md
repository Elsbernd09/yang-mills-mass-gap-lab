# Yang–Mills Mass Gap Laboratory

A research-grade computational and mathematical laboratory for studying non-Abelian Yang–Mills theory, lattice gauge theory, Wilson loops, confinement, and numerical mass-gap behavior.

## Purpose

The Yang–Mills existence and mass gap problem is one of the Clay Mathematics Institute Millennium Prize Problems. It asks for a rigorous construction of quantum Yang–Mills theory on four-dimensional spacetime for a compact simple gauge group, together with a proof that the theory has a positive mass gap.

This repository does **not** claim to solve the Yang–Mills Millennium Problem.

Instead, it builds a serious computational and expository framework for studying the mathematical structures surrounding the problem:

- Lie groups and Lie algebras
- Non-Abelian gauge symmetry
- Connections and curvature
- Classical Yang–Mills equations
- Lattice gauge theory
- Wilson action dynamics
- Wilson loop observables
- Confinement behavior
- Correlation decay
- Numerical mass-gap estimation

## Core Mathematical Objects

The classical Yang–Mills curvature of a connection \(A\) is

\[
F_A = dA + A \wedge A.
\]

The classical Yang–Mills equation is

\[
D_A^* F_A = 0.
\]

On a lattice, the Wilson action is

\[
S_W(U) = \beta \sum_p \left(1 - \frac{1}{N}\operatorname{Re}\operatorname{Tr}(U_p)\right),
\]

where \(U_p\) is the plaquette variable around an elementary square.

A mass-gap-like signal may be numerically estimated from exponential correlation decay:

\[
C(t) \sim e^{-mt},
\]

where \(m > 0\) is interpreted as an effective mass scale.

## Project Structure

```text
yang-mills-mass-gap-lab/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── docs/
├── notebooks/
├── src/
│   └── ymlab/
│       └── su2.py
├── experiments/
├── results/
└── tests/
    └── test_su2.py
Current Scope
The first implementation focuses on compact SU(2) lattice gauge theory. SU(2) is mathematically rich, non-Abelian, and computationally more accessible than SU(3), making it a strong starting point for a rigorous educational and experimental framework.
Scientific Honesty
This repository is a computational exploration and educational research framework. It does not provide a proof of Yang–Mills existence or of the mass gap. Numerical evidence from lattice models is not equivalent to a rigorous continuum quantum field theory construction.
Long-Term Vision
The long-term goal is to create a clean, reproducible, mathematically transparent platform for exploring the geometry and physics of Yang–Mills theory through computation, exposition, and numerical experimentation.


## Advanced Sampling Diagnostics

- Exact SU(2) microcanonical overrelaxation reflections
- Quaternion-norm staple normalization
- Strict SU(2) projection and lattice membership preservation
- Local and global Wilson-action preservation audits
- Hybrid Metropolis-overrelaxation update schedules
- Observable-dependent integrated autocorrelation comparisons
- Effective sample size per second sampling-efficiency benchmarks


## Generic Gauge-Group Architecture

- Group-aware GaugeLattice architecture
- Matrix dimension supplied by the active gauge-group backend
- Strict backend-driven link membership validation
- Generic plaquettes and normalized Wilson observables
- Generic Wilson action using backend-provided multiplicative 1/N trace normalization
- Generic reverse-oriented local staples
- Generic local Metropolis action differences
- Generic rectangular Wilson loops
- Numerical SU(2) old-versus-generic backend equivalence audit
- Working SU(3) GaugeLattice and Metropolis prototype path


## SU(3) Finite-Lattice Prototype

- Generic 3x3 SU(3) GaugeLattice evolution
- SU(3) local-versus-global Wilson-action difference validation
- Generic site-local gauge transformation engine
- Numerical SU(3) Wilson-action gauge-invariance validation
- Numerical SU(3) average-plaquette gauge-invariance validation
- Closed SU(3) rectangular Wilson-loop gauge-invariance tests
- Independent short-chain SU(3) finite-lattice diagnostics
- Plaquette autocorrelation and effective-sample-size analysis
- Explicit SU(3) link-membership error monitoring


## Ensemble Wilson-Loop Uncertainty

- Configuration-level rectangular Wilson-loop measurement matrices
- Complete post-thermalization Wilson-loop ensembles
- Circular moving-block bootstrap of complete configuration rows
- Joint nonlinear Creutz-ratio uncertainty propagation
- Bootstrap Creutz means, standard errors, medians, and 95% intervals
- Explicit invalid-replicate and valid-fraction diagnostics
- Autocorrelation-informed heuristic block-size selection
- Square Creutz string-tension-style plateau diagnostics
- Wilson-loop relative-noise growth analysis


## Reproducible Experiment Runner

- Central validated experiment registry
- `ymlab list` command for registered numerical experiments
- `ymlab run` command for deterministic configured runs
- Canonical JSON configuration hashing with SHA-256
- Independent timestamped run directories
- JSON run manifests with package and platform versions
- Best-effort Git commit and dirty-tree provenance
- Runtime duration and execution status metadata
- Registered experiment output-file tracking
- Failure manifests that preserve exception metadata
- Configuration-level SU(2) and SU(3) plaquette-chain runners


## Final Controlled Research Question

**How does SU(2) microcanonical overrelaxation frequency affect effective sample size per wall-clock second for the average plaquette and a scalar glueball-style composite observable across finite bare Wilson couplings?**

The controlled campaign:

- compares Metropolis, M+1OR, M+2OR, and M+4OR schedules,
- studies beta 1.5, 2.0, and 2.5,
- uses independent deterministic seeds,
- begins paired schedules from shared thermalized configurations,
- measures plaquette and scalar autocorrelation independently,
- calculates observable-specific effective sample size,
- incorporates measured wall-clock runtime,
- reports ESS per second and paired efficiency ratios,
- allows the winning schedule to vary by beta and observable,
- records the complete campaign in a reproducible run manifest.

The study is a finite-lattice computational-methodology experiment. It does not claim continuum or infinite-volume Yang-Mills results.
