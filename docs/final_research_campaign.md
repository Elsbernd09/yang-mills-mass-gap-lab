# Final Controlled Research Campaign

## Research Question

How does SU(2) microcanonical overrelaxation frequency affect effective sample
size per wall-clock second for the average plaquette and a scalar
glueball-style composite observable across finite bare Wilson couplings?

## Motivation

Microcanonical overrelaxation can reduce Markov-chain autocorrelation by moving
through constant-action surfaces.

Additional overrelaxation sweeps also consume computational time.

Therefore lower integrated autocorrelation time does not automatically imply a
more computationally efficient simulation.

The central quantity in this campaign is effective sample size per wall-clock
second.

## Update Schedules

The campaign compares four update schedules.

M denotes one complete local Metropolis sweep.

OR denotes one complete SU(2) microcanonical overrelaxation sweep.

The schedules are:

M,

M plus one OR,

M plus two OR,

M plus four OR.

## Coupling Regimes

The default campaign studies three finite bare Wilson couplings:

beta equals 1.5,

beta equals 2.0,

beta equals 2.5.

These values are treated as computational regimes of the finite lattice.

No physical scale matching or continuum interpretation is implied.

## Fixed Lattice Geometry

The default campaign uses a small three-dimensional periodic lattice.

The lattice shape is five by four by four.

Keeping the geometry fixed isolates the update-schedule comparison from a
simultaneous volume change.

## Paired Starting Configurations

For each beta and random seed, one Metropolis chain is thermalized.

The resulting configuration is copied.

Every update schedule for that beta-seed pair begins from the same thermalized
link field.

This creates a paired computational design.

The paired design does not eliminate Monte Carlo variability.

It reduces one source of baseline starting-configuration variation in
within-pair schedule comparisons.

## Independent Seeds

The default campaign uses three deterministic random seeds.

Every seed creates an independently thermalized starting chain.

Results are retained at the beta-seed-schedule level.

Aggregate schedule summaries are calculated across seeds.

Three seeds are not treated as sufficient for a formal asymptotic significance
claim.

The campaign reports descriptive finite-sample behavior.

## Observables

Two observables are measured after every hybrid update.

The first is the average normalized plaquette.

The second is the mean of the scalar glueball-style Euclidean time-slice
operator previously developed in the spectroscopy pipeline.

The two observables probe different functions of the gauge field.

An update schedule can therefore decorrelate them differently.

## Integrated Autocorrelation Time

The project estimates integrated autocorrelation time using its current
positive-sequence estimator.

The estimator is applied separately to every observable time series.

## Effective Sample Size

Effective sample size is calculated from the measured chain length and the
estimated integrated autocorrelation time.

This attempts to express the number of effectively independent observations
represented by an autocorrelated measured chain.

## Wall-Clock Runtime

Every measured schedule chain is timed using a high-resolution process clock.

The campaign records the complete measured hybrid-update runtime.

Absolute timings describe the current pure Python and NumPy implementation on
the executing machine.

They should not be interpreted as universal lattice-gauge performance numbers.

## Effective Sample Size Per Second

For every observable and schedule, the campaign calculates

effective sample size divided by measured runtime in seconds.

This combines statistical decorrelation and computational cost.

## Paired Efficiency Ratios

For every beta-seed pair, each schedule's ESS per second is divided by the ESS
per second of the Metropolis-only chain from the same paired condition.

A ratio greater than one indicates greater measured efficiency than the paired
Metropolis baseline.

A ratio below one indicates lower measured efficiency.

The ratio is descriptive.

## Winner Selection

At each beta, the campaign identifies the schedule with the largest mean ESS
per second across seeds.

Plaquette and scalar winners are selected independently.

The implementation does not require the same schedule to win for both
observables.

No preferred conclusion is encoded in advance.

## Reproducibility

The complete campaign executes inside the project's reproducibility
infrastructure.

A run manifest records:

- full campaign configuration,
- configuration hash,
- UTC timestamp,
- Python and package versions,
- platform metadata,
- Git commit when available,
- Git dirty-tree state,
- runtime,
- output files,
- completion status.

## Campaign Outputs

The campaign writes configuration-level records to campaign_records.csv.

Each row represents one beta-seed-schedule chain.

Aggregated beta-schedule summaries are written to campaign_summary.csv.

Machine-readable winner and interpretation metadata are written to
campaign_conclusion.json.

Figures show plaquette and scalar ESS per second across beta and paired
plaquette efficiency ratios relative to Metropolis.

## Interpretation Rule

A schedule is described as more computationally efficient only where its
measured mean ESS per second exceeds the paired Metropolis baseline.

A lower autocorrelation time alone is not sufficient.

The campaign does not infer formal statistical significance from three seeds.

## Scientific Scope

This is a finite-lattice computational-methodology study.

It studies observable-dependent sampling efficiency inside the project's SU(2)
Wilson-lattice implementation.

It does not perform continuum extrapolation.

It does not perform infinite-volume extrapolation.

It does not set a physical scale.

It does not claim a new Yang-Mills mass-gap result.

The campaign is intended to provide one narrow, reproducible, empirically
answered research question for the Yang-Mills Mass Gap Laboratory.
