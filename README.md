# Yang–Mills Mass Gap Laboratory

A validated computational laboratory for finite-lattice non-Abelian gauge theory, Wilson-action simulation, gauge-invariant observables, spectroscopy pipelines, uncertainty analysis, and reproducible numerical experiments.

> **Scientific scope:** This repository does **not** claim to solve the Clay Mathematics Institute Yang–Mills existence and mass-gap problem. It studies finite-lattice numerical gauge theory and computational methodology.

## Final Research Result

The laboratory culminates in a controlled SU(2) update-efficiency study:

> **How does microcanonical overrelaxation frequency affect effective sample size per wall-clock second for the average plaquette and a scalar glueball-style composite observable across finite bare Wilson couplings?**

The clean release campaign compared four schedules:

- `M` — one Metropolis sweep
- `M+1OR` — Metropolis plus one overrelaxation sweep
- `M+2OR` — Metropolis plus two overrelaxation sweeps
- `M+4OR` — Metropolis plus four overrelaxation sweeps

The campaign studied:

- beta = 1.5, 2.0, and 2.5
- three independent deterministic seeds
- a periodic 5 x 4 x 4 SU(2) lattice
- 180 measured hybrid updates per schedule
- 36 measured campaign chains

### Main finding

Plain Metropolis produced the highest measured mean **plaquette ESS/sec** at all three tested couplings.

For the scalar composite observable:

- `M` was most efficient at beta = 1.5
- `M` was most efficient at beta = 2.0
- `M+1OR` was most efficient at beta = 2.5

At beta = 2.5, the clean release campaign measured a paired scalar ESS/sec ratio of **1.250488** for `M+1OR` relative to Metropolis, corresponding to an approximately **25.05% measured advantage**.

The central empirical conclusion is:

> **Autocorrelation reduction and computational efficiency are distinct quantities, and the computational value of microcanonical overrelaxation can depend on both the measured observable and the finite bare-coupling regime.**

The result is finite-lattice, implementation-dependent, and descriptive. It is not a continuum mass-gap result.

## Research Paper

The final manuscript is available here:

[`docs/final_research_paper.md`](docs/final_research_paper.md)

**Title:** Observable-Dependent Computational Efficiency of Microcanonical Overrelaxation in Finite-Lattice SU(2) Yang–Mills Simulation

**Subtitle:** A Paired Effective-Sample-Size-per-Second Study Across Bare Wilson Couplings

## Laboratory Architecture

The project builds the numerical pipeline layer by layer:

    SU(2) / SU(3) matrix groups
            |
            v
    periodic hypercubic gauge lattices
            |
            v
    plaquettes and Wilson action
            |
            v
    local staple geometry
            |
            v
    Metropolis evolution
            |
            v
    gauge transformations and invariance
            |
            v
    Wilson loops and Creutz ratios
            |
            v
    autocorrelation and effective sample size
            |
            v
    correlation-aware resampling
            |
            v
    scalar glueball-style operators
            |
            v
    APE-style spatial smearing
            |
            v
    multi-operator Euclidean correlator matrices
            |
            v
    regularized generalized eigenvalue problem
            |
            v
    covariance-aware periodic spectroscopy
            |
            v
    GEVP bootstrap state matching
            |
            v
    microcanonical overrelaxation
            |
            v
    generic matrix-gauge lattice architecture
            |
            v
    validated SU(3) finite-lattice prototype
            |
            v
    ensemble Wilson-loop block bootstrap
            |
            v
    reproducible experiment manifests and CLI
            |
            v
    controlled overrelaxation-efficiency campaign

## Implemented Components

### Gauge-group foundations

- SU(2) identity, conjugation, real trace, random sampling, and reunitarization
- Quaternion-based SU(2) construction
- Near-identity SU(2) proposal generation
- SU(3) identity, projection, random sampling, and reunitarization
- Eight Gell–Mann matrices
- Hermiticity, tracelessness, and normalization checks
- Generic `MatrixGaugeGroup` backend

### Lattice gauge infrastructure

- Periodic hypercubic lattices
- Arbitrary finite lattice dimension
- Positive-direction link storage
- Strict link group-membership validation
- Generic matrix-dimension-aware `GaugeLattice`
- SU(2)-specific and generic gauge backends

### Wilson action and local geometry

- Oriented plaquette construction
- Normalized plaquette observables
- Wilson plaquette action density
- Full Wilson action
- Reverse-oriented link staples
- Local link action
- Local action differences
- Local-versus-global Wilson-action difference audits in 2D, 3D, and 4D

### Monte Carlo and update algorithms

- Local SU(2) Metropolis updates
- Generic matrix-group Metropolis evolution
- Burn-in handling
- Measurement intervals
- SU(2) microcanonical overrelaxation
- Hybrid Metropolis-overrelaxation schedules
- Numerical microcanonical action-preservation auditing

### Gauge symmetry validation

Local gauge transformations use

U_mu(x) -> G(x) U_mu(x) G(x+mu)^dagger.

The laboratory validates gauge invariance of:

- Wilson action
- average plaquette
- closed rectangular Wilson loops
- scalar glueball-style time-slice operators
- smeared scalar observables

The generic SU(3) prototype includes independent local gauge-transformation checks.

### Wilson loops and confinement-style diagnostics

- Rectangular Wilson loops
- Lattice-position-averaged Wilson loops
- Wilson-loop measurement tables
- Creutz ratios
- Exact synthetic area-law recovery
- Configuration-level Wilson-loop ensembles
- Circular moving-block bootstrap
- Joint nonlinear Creutz-ratio uncertainty propagation
- Bootstrap standard errors and percentile intervals
- Invalid-replicate and valid-fraction reporting
- Square Creutz string-tension-style diagnostics

### Correlation and uncertainty analysis

- Connected autocorrelation functions
- Normalized autocorrelation functions
- Integrated autocorrelation time
- Effective sample size
- Burn-in removal
- IID bootstrap baseline
- Circular moving-block bootstrap
- Delete-one jackknife
- Delete-one-block jackknife
- Block-size sensitivity analysis

### Scalar composite and spectroscopy pipeline

- Spatial-plane scalar glueball-style operator
- Euclidean time-slice observables
- Periodic configuration correlations
- Ensemble connected correlators
- Normalized connected correlators
- APE-style spatial link smearing
- Gauge-covariant spatial staples
- Multi-smearing operator bases
- Operator correlation matrices
- Regularized generalized eigenvalue analysis
- Principal correlators
- Arccosh effective mass
- Periodic cosh models
- Plateau scans
- Covariance-aware correlated fits
- Shrinkage and regularized inverse covariance
- GEVP bootstrap state matching
- Correlated principal-state mass fitting

### SU(3) finite-lattice prototype

The generic architecture supports a validated finite SU(3) simulation path with:

- 3 x 3 link matrices
- normalized one-third real-trace Wilson observables
- generic Wilson action
- local SU(3) Metropolis proposals
- local-versus-global action-difference validation
- random local SU(3) gauge transformations
- Wilson-action gauge-invariance validation
- average-plaquette gauge-invariance validation
- closed-loop gauge-invariance validation
- short-chain autocorrelation and ESS diagnostics

This is a finite-lattice prototype, not production lattice QCD software.

## Validation

The master validation suite contains **26 registered structural and synthetic checks**.

Coverage includes:

1. SU(2) identity and random-element validation
2. cold-start plaquette validation
3. cold-start Wilson-action validation
4. cold-start Wilson-loop validation
5. Metropolis SU(2) preservation
6. exact synthetic area-law Creutz recovery
7. lattice dimension counts
8. SU(3) identity and random-element validation
9. Gell–Mann basis validation
10. local gauge invariance
11. correlation-aware resampling
12. scalar glueball-style correlators
13. periodic spectroscopy
14. local/global Wilson-action consistency
15. spatial smearing
16. operator correlation matrices
17. generalized eigenvalue recovery
18. correlated spectroscopy fits
19. GEVP bootstrap state matching
20. correlated principal-state fitting
21. microcanonical overrelaxation
22. generic `GaugeLattice`
23. SU(3) structural lattice validation
24. ensemble Creutz block bootstrap
25. reproducibility manifests
26. controlled research-campaign machinery

Run the master validation suite with:

    python experiments/validation_suite.py

Expected final state:

    Validation checks passed: 26/26
    All validation checks passed.

Run the complete unit-test suite with:

    pytest

## Reproducible Experiment CLI

Install the project in editable mode:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

List registered experiments:

    ymlab list

Run a compact SU(2) plaquette experiment:

    ymlab run su2-plaquette --shape 4 4 --beta 2.0 --epsilon 0.18 --thermalization 10 --measurements 25 --seed 2026

Run a compact SU(3) plaquette experiment:

    ymlab run su3-plaquette --shape 3 3 3 --beta 5.5 --epsilon 0.05 --thermalization 3 --measurements 5 --seed 2032

Registered runs create independent directories under:

    results/runs/

A run can contain:

    manifest.json
    measurements.csv
    summary.json

The manifest records:

- experiment name
- validated configuration
- SHA-256 configuration hash
- UTC timestamp
- Python version
- NumPy version
- SciPy version
- Matplotlib version
- operating system
- machine architecture
- Git commit, when available
- Git dirty or clean state, when available
- runtime
- execution status
- registered output files

## Final Controlled Campaign

Run:

    python experiments/final_overrelaxation_campaign.py

The campaign writes:

    campaign_records.csv
    campaign_summary.csv
    campaign_conclusion.json
    manifest.json
    figures/

The final paper generator selects the newest completed campaign whose manifest records:

    git.available = true
    git.dirty = false
    status = completed

Generate the manuscript with:

    python scripts/generate_final_paper.py

## Selected Documentation

- [`docs/lattice_gauge_theory.md`](docs/lattice_gauge_theory.md)
- [`docs/gauge_invariance.md`](docs/gauge_invariance.md)
- [`docs/correlated_uncertainty.md`](docs/correlated_uncertainty.md)
- [`docs/glueball_correlators.md`](docs/glueball_correlators.md)
- [`docs/spectroscopy.md`](docs/spectroscopy.md)
- [`docs/smearing.md`](docs/smearing.md)
- [`docs/generic_gauge_lattice.md`](docs/generic_gauge_lattice.md)
- [`docs/su3_finite_lattice_prototype.md`](docs/su3_finite_lattice_prototype.md)
- [`docs/ensemble_wilson_creutz.md`](docs/ensemble_wilson_creutz.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/final_research_campaign.md`](docs/final_research_campaign.md)
- [`docs/final_research_paper.md`](docs/final_research_paper.md)

## Repository Structure

    yang-mills-mass-gap-lab/
    |-- docs/
    |-- experiments/
    |-- notebooks/
    |-- results/
    |-- scripts/
    |-- src/
    |   `-- ymlab/
    |-- tests/
    |-- LICENSE
    |-- README.md
    |-- pyproject.toml
    `-- requirements.txt

## Scientific Limitations

The laboratory currently uses small finite lattices and is primarily implemented in Python and NumPy.

The final campaign:

- uses a 5 x 4 x 4 three-dimensional SU(2) lattice
- studies three bare Wilson couplings
- uses three deterministic independent seeds
- measures 180 hybrid updates per schedule
- uses a positive-sequence autocorrelation diagnostic
- reports descriptive finite-sample schedule comparisons
- measures machine- and implementation-dependent wall-clock runtime

The project does not perform:

- continuum extrapolation
- infinite-volume extrapolation
- physical scale setting
- production high-statistics glueball spectroscopy
- a four-dimensional constructive Yang–Mills proof
- a proof of a positive continuum mass gap

## Author

**Michael Elsbernd**

Sacred Heart Cathedral Preparatory  
San Francisco, California

## License

MIT License.
