# Reproducible Experiment Infrastructure

## Purpose

The project contains a growing collection of numerical lattice-gauge
experiments.

Historically, many exploratory experiments were configured by editing constants
inside individual Python scripts.

That workflow is useful during development but makes large controlled campaigns
harder to reproduce.

The project therefore introduces a validated experiment registry, command-line
interface, and run-manifest system.

## Registered Experiments

The command-line interface exposes experiments through a central registry.

The initial registry includes compact SU(2) and SU(3) plaquette-chain
experiments.

The registry is intentionally incremental.

Historical research scripts remain available.

Future controlled campaigns can add registered runners without requiring a
destructive rewrite of the validated exploratory code.

## Validated Configuration

PlaquetteExperimentConfig records:

- lattice shape,
- beta,
- proposal epsilon,
- thermalization sweeps,
- measurement sweeps,
- deterministic random seed.

Configuration values are validated before an experiment begins.

Invalid dimensions, nonpositive proposal scales, negative beta values, and
insufficient measurement lengths are rejected.

## Deterministic Configuration Hash

A configuration dictionary is converted to canonical sorted JSON.

The canonical representation is hashed with SHA-256.

A fixed prefix of the digest becomes the configuration hash.

Equivalent dictionaries with different key insertion order therefore produce
the same configuration hash.

Changing a configuration parameter changes the hash.

The hash identifies configuration content.

It is not a cryptographic statement about scientific validity.

## Run Directory

Every registered experiment receives an independent run directory.

The directory name contains:

- experiment name,
- configuration hash,
- UTC creation timestamp.

Repeated executions of the same configuration therefore remain separate while
retaining a common configuration identity.

## Manifest Metadata

Every run writes manifest.json.

The manifest records:

- experiment name,
- configuration,
- configuration hash,
- UTC creation timestamp,
- Python version,
- NumPy version,
- SciPy version,
- Matplotlib version,
- operating-system name,
- operating-system release,
- machine architecture,
- Git commit when available,
- Git dirty or clean state when available,
- runtime duration,
- execution status,
- generated output files,
- optional notes.

## Git Metadata

Git metadata is collected on a best-effort basis.

When the experiment executes inside a Git repository, the current commit is
recorded.

The manifest also records whether Git reports uncommitted working-tree
changes.

A dirty working tree is not hidden.

This allows later readers to distinguish a run performed from a committed
snapshot from one performed during active development.

## Failure Manifests

The manifest is written before the experiment begins.

If a registered experiment raises an exception, the manifest is finalized with
failed status and the exception type and message are recorded in the notes.

This makes failed computational attempts visible rather than silently
discarding their execution state.

## Output Registration

Generated files can be registered with the manifest.

Paths under the run directory are stored relative to the run directory.

The compact plaquette experiments register:

- measurements.csv,
- summary.json.

The manifest itself is always stored as manifest.json.

## Command-Line Interface

The package exposes a ymlab command.

The list command displays registered experiments.

The run command accepts:

- experiment name,
- lattice shape,
- beta,
- proposal epsilon,
- thermalization count,
- measurement count,
- random seed,
- optional output root.

Example:

ymlab run su2-plaquette
with shape 4 4
beta 2.0
epsilon 0.18
thermalization 10
measurements 25
seed 2026.

## Measurement Output

The compact plaquette runners write measurements.csv.

Each measurement row records:

- sweep index,
- average plaquette,
- Wilson action per plaquette,
- Metropolis acceptance rate.

## Summary Output

The summary JSON records compact chain statistics including:

- gauge group,
- lattice shape,
- beta,
- epsilon,
- mean thermal acceptance,
- mean measurement acceptance,
- mean average plaquette,
- mean action per plaquette,
- plaquette integrated autocorrelation time,
- plaquette effective sample size.

## Scope

This infrastructure improves computational provenance and repeatability.

It does not make a short chain scientifically sufficient.

It does not validate thermalization automatically.

It does not guarantee that a selected lattice volume or bare coupling is
physically appropriate.

Reproducibility infrastructure records what was done.

Scientific methodology must still determine whether the run answers a valid
research question.
