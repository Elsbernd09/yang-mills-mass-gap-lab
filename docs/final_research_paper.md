# Observable-Dependent Computational Efficiency of Microcanonical Overrelaxation in Finite-Lattice SU(2) Yang–Mills Simulation

## A Paired Effective-Sample-Size-per-Second Study Across Bare Wilson Couplings

**Michael Elsbernd**

Sacred Heart Cathedral Preparatory, San Francisco, California

Yang–Mills Mass Gap Laboratory

Repository: https://github.com/Elsbernd09/yang-mills-mass-gap-lab

---

## Abstract

Microcanonical overrelaxation is designed to move a lattice gauge configuration along a constant-action surface and can reduce Markov-chain autocorrelation when combined with stochastic updates. Reduction of integrated autocorrelation time, however, does not by itself imply greater computational efficiency because additional overrelaxation sweeps incur wall-clock cost.

This study asks a narrow computational-methodology question: **How does SU(2) microcanonical overrelaxation frequency affect effective sample size per wall-clock second for the average plaquette and a scalar glueball-style composite observable across finite bare Wilson couplings?**

A three-dimensional periodic SU(2) Wilson lattice of shape 5 x 4 x 4 was studied at bare couplings beta = 1.5, 2.0, and 2.5. Four update schedules were compared: one Metropolis sweep alone and one Metropolis sweep followed by one, two, or four microcanonical overrelaxation sweeps. Three independent deterministic seeds were used. Within each beta-seed pair, every schedule began from a copy of the same Metropolis-thermalized gauge configuration. Each measured schedule contained 180 hybrid updates.

Integrated autocorrelation time and effective sample size were estimated separately for the average plaquette and scalar composite observable. Efficiency was reported as effective sample size divided by measured wall-clock runtime.

Across 36 measured chains, plain Metropolis produced the highest measured mean plaquette ESS/s at all three tested couplings. For the scalar observable, plain Metropolis was the most efficient tested schedule at beta = 1.5 and beta = 2.0. At beta = 2.5, one overrelaxation sweep per Metropolis sweep was the most efficient tested scalar schedule. Its measured mean ESS/s ratio to the paired Metropolis baseline was 1.250488, an approximately 25.05% advantage.

Several hybrid schedules reduced estimated integrated autocorrelation time without improving ESS/s. Autocorrelation reduction and runtime-normalized computational efficiency were therefore empirically non-equivalent in the tested implementation.

The results support an observable- and finite-regime-dependent view of update-schedule efficiency. They do not establish a universal advantage for overrelaxation, a continuum result, an infinite-volume result, or a solution of the Yang–Mills mass-gap problem.

**Keywords:** Yang–Mills theory; lattice gauge theory; SU(2); Wilson action; microcanonical overrelaxation; Metropolis algorithm; autocorrelation; effective sample size; computational efficiency.

---

## 1. Introduction

Non-Abelian Yang–Mills theory is central to modern mathematical physics. Its local gauge symmetry and nonlinear field interactions produce profound nonperturbative questions. The rigorous construction of four-dimensional quantum Yang–Mills theory together with proof of a positive mass gap remains a Clay Mathematics Institute Millennium Prize Problem.

The present study does not claim to solve that problem.

Its purpose is considerably narrower: to examine how the computational value of a deterministic microcanonical update depends on the measured observable and finite bare-coupling regime.

The Wilson lattice formulation replaces continuum gauge fields with group-valued link variables. In SU(2), every positively oriented lattice link carries a matrix

U_mu(x) in SU(2).

The elementary plaquette is

U_mu_nu(x)
=
U_mu(x)
U_nu(x + mu)
U_mu(x + nu) dagger
U_nu(x) dagger.

The Wilson action used by the present laboratory is

S[U]
=
beta sum over p of
[
1 - one half ReTr(U_p)
].

Local Metropolis updates provide a stochastic mechanism for evolving finite-lattice gauge configurations. A near-identity SU(2) perturbation is proposed for one link. The local Wilson-action difference is calculated using the staple. The proposal is then accepted according to the Metropolis rule.

Successive Markov-chain configurations are correlated. Consequently, a chain of N measured configurations generally contains less statistical information than N independent samples.

Microcanonical overrelaxation provides a complementary deterministic transition. The implemented SU(2) reflection preserves the Wilson action to numerical precision while moving the gauge field through configuration space.

A common algorithmic intuition is that lower integrated autocorrelation time implies a better update schedule. That criterion is incomplete when computational time is a limited resource.

An update can reduce autocorrelation while being sufficiently expensive that fewer effectively independent samples are produced per second.

The central efficiency quantity in the present study is therefore

ESS/sec
=
effective sample size
divided by
measured wall-clock time.

Four update schedules are compared:

M,

M + 1 OR,

M + 2 OR,

M + 4 OR,

where M denotes one complete Metropolis sweep and OR denotes one complete microcanonical overrelaxation sweep.

Two observables are evaluated independently.

The first is the average normalized plaquette.

The second is a scalar glueball-style composite observable derived from spatial plaquette action densities.

The research question is:

> **How does SU(2) microcanonical overrelaxation frequency affect effective sample size per wall-clock second for the average plaquette and a scalar glueball-style composite observable across finite bare Wilson couplings?**

No preferred schedule is selected in advance.

---

## 2. Computational Laboratory

The controlled campaign is embedded in a broader numerical Yang–Mills laboratory.

The project implements and validates:

- SU(2) group primitives;
- SU(3) algebraic primitives;
- periodic hypercubic lattices;
- Wilson plaquettes and Wilson action;
- local staple-based action differences;
- local Metropolis evolution;
- rectangular Wilson loops;
- Creutz-ratio diagnostics;
- connected Euclidean correlators;
- effective-mass estimators;
- bootstrap and correlation-aware block resampling;
- spatial APE-style smearing;
- scalar glueball-style operators;
- multi-operator correlation matrices;
- generalized eigenvalue analysis;
- covariance-aware periodic spectroscopy;
- GEVP bootstrap state matching;
- SU(2) microcanonical overrelaxation;
- a generic matrix-gauge lattice backend;
- an SU(3) finite-lattice prototype;
- ensemble Wilson-loop and Creutz-ratio uncertainty propagation;
- reproducible command-line experiment execution;
- configuration hashing and run manifests.

The central design principle is independent numerical validation.

Where practical, local algorithms are compared against global recomputation.

Local staple action differences are tested against full Wilson-action differences after a temporary link replacement.

Overrelaxation is checked through local and global action preservation.

Gauge transformations are tested against gauge-invariant observables.

The generic SU(2) backend is regression-compared against the established SU(2)-specific implementation.

The SU(3) prototype separately validates local-versus-global action consistency and local gauge invariance.

At the time of the final campaign, the master validation suite contained 26 registered structural and synthetic checks.

The broader unit-test suite was green before the clean release campaign was interpreted.

These tests establish internal numerical consistency and synthetic recovery targets. They do not constitute a mathematical proof of continuum Yang–Mills theory.

---

## 3. SU(2) Wilson-Lattice Formulation

### 3.1 Link variables

At each site x and positive direction mu, the gauge field is represented by

U_mu(x) in SU(2).

The lattice storage layer validates numerical group membership before storing a link.

### 3.2 Plaquettes

The oriented plaquette is constructed from four links forming a closed elementary loop.

The normalized real plaquette trace is

P_mu_nu(x)
=
one half ReTr(U_mu_nu(x)).

The average plaquette is obtained by averaging over all sites and unordered positive direction pairs.

### 3.3 Wilson action

The finite-lattice Wilson action is

S[U]
=
beta
times the sum over plaquettes of

1 - one half ReTr(U_p).

For a cold identity configuration, every plaquette is the identity, the average normalized plaquette is one, and the Wilson action is zero.

These limits are directly tested.

### 3.4 Local action and staple geometry

For one selected link U_mu(x), the implementation constructs a reverse-oriented staple V so that

U_mu(x) V

closes all attached elementary plaquette paths contributing to the selected link.

The link-dependent Wilson action is

S_local
=
- beta over 2
times
ReTr(U_mu(x) V).

For a proposed link U prime, the local action difference is

Delta S_local
=
S_local[U prime]
-
S_local[U].

The laboratory independently computes a global action difference by evaluating the complete Wilson action before and after a temporary link replacement.

Random proposals in multiple lattice dimensions are required to satisfy local-global action consistency to numerical precision.

---

## 4. Metropolis Evolution

A Metropolis proposal has the form

U prime
=
R(epsilon) U,

where R(epsilon) is a small random near-identity SU(2) matrix.

The proposal scale in the final campaign was epsilon = 0.18.

A proposal is accepted automatically when Delta S is nonpositive.

For positive Delta S, it is accepted with probability exp(-Delta S).

One Metropolis sweep attempts a local update of every positively oriented link.

Across the 36 final campaign records, the mean measured Metropolis acceptance rate was 0.78527456.

Every hybrid schedule retained one stochastic Metropolis sweep.

---

## 5. Microcanonical Overrelaxation

### 5.1 SU(2) staple normalization

The SU(2) Wilson-action staple retains quaternionic matrix structure.

For a nonzero staple V,

V
=
k times V_tilde,

where V_tilde belongs numerically to SU(2).

The scale is evaluated from

k squared
=
one half ReTr(V dagger V).

Thus,

k
=
square root of
one half ReTr(V dagger V).

This avoids relying on a floating-point determinant to normalize the staple.

### 5.2 Reflection update

Let

X
=
U V_tilde.

The overrelaxation reflection maps X to X dagger.

Solving for the reflected link yields the implemented microcanonical proposal.

The resulting matrix is numerically reunitarized before storage to remove floating-point drift.

### 5.3 Numerical action preservation

The overrelaxation pipeline was validated before the campaign.

The clean final campaign recorded a maximum microcanonical action error of

1.598721155460e-14.

This is at floating-point scale for the tested computation.

The microcanonical update was not used as a standalone sampler.

Every measured schedule contained a stochastic Metropolis sweep.

---

## 6. Observables

### 6.1 Average plaquette

The first measured observable was the average normalized plaquette.

It is local, gauge invariant, and directly related to Wilson-action density.

### 6.2 Scalar glueball-style composite observable

The second measurement was derived from the laboratory's scalar composite operator pipeline.

Purely spatial plaquette action densities are averaged at fixed Euclidean time.

The campaign takes the mean of the resulting time-slice observable series to produce one scalar configuration measurement per hybrid update.

The term **glueball-style** is intentional.

The campaign observable is not presented as a complete production 0++ glueball spectroscopy basis.

The broader laboratory contains smearing, multi-operator correlation matrices, generalized eigenvalue analysis, and periodic spectroscopy, but the controlled campaign uses the compact scalar composite as an observable-specific Markov-chain diagnostic.

---

## 7. Autocorrelation, Effective Sample Size, and Runtime

For a measured scalar time series X_t, the laboratory constructs the normalized connected autocorrelation function.

Integrated autocorrelation time is estimated using the project's positive-sequence truncation diagnostic.

The estimator is applied identically to every schedule and observable.

From the measured chain length and estimated autocorrelation time, an effective sample size is calculated.

The campaign's central efficiency measure is

E_X
=
N_eff,X
divided by
t_wall.

This quantity is calculated separately for the plaquette and scalar observable.

Within each beta-seed pair, schedule efficiency is also normalized to the paired Metropolis-only baseline:

rho_X(schedule)
=
E_X(schedule)
divided by
E_X(M).

A paired ratio greater than one indicates higher measured ESS/sec than Metropolis.

A ratio below one indicates lower measured ESS/sec.

A reduction in integrated autocorrelation time alone is not sufficient to declare a schedule more efficient.

---

## 8. Controlled Experimental Design

The final campaign used a three-dimensional periodic lattice with shape

5 x 4 x 4.

The tested bare Wilson couplings were

beta = 1.5,

beta = 2.0,

beta = 2.5.

The independent deterministic seeds were

2026,

2027,

2028.

The schedules were

M,

M + 1 OR,

M + 2 OR,

M + 4 OR.

For each beta-seed pair, one lattice was thermalized with 60 Metropolis sweeps.

The resulting gauge configuration was copied.

Every schedule for the same beta-seed pair began from the same thermalized link field.

Every measured schedule then generated 180 hybrid updates.

The complete controlled design contained

3 beta values
times
3 seeds
times
4 schedules
=
36 measured chains.

The paired starting-field design reduces one source of baseline configuration variation in within-pair comparisons.

It does not eliminate Monte Carlo variation.

Three seeds are treated descriptively and are not used to claim asymptotic statistical significance.

---

## 9. Reproducibility and Provenance

The final release campaign was executed from Git commit

`6b8ea687a07ece302bfa2c6e6d3a5c31ff01b458`

with the Git working tree recorded as clean.

The canonical campaign configuration hash was

`379d46d5354be739`.

The complete campaign runtime was

353.857940 seconds.

The active Python version was `3.9.6`.

The active NumPy version was `2.0.2`.

The campaign produced 36 chain-level records and 12 beta-schedule aggregate summaries.

The manifest also records package versions, platform metadata, Git provenance, run status, runtime, and registered output files.

The final manuscript generator selects the newest completed clean-Git campaign rather than manually choosing a result directory.

---

## 10. Results

### 10.1 Aggregate numerical results

| Beta | Schedule | Plaquette tau_int | Plaquette ESS/s | Plaquette ratio | Scalar tau_int | Scalar ESS/s | Scalar ratio | Mean runtime (s) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1.500 | M | 4.771099 | 8.679435 | 1.000000 | 4.106735 | 9.618905 | 1.000000 | 2.339922 |
| 1.500 | M+1OR | 7.670215 | 2.437833 | 0.332788 | 3.087340 | 6.160767 | 0.683342 | 6.559804 |
| 1.500 | M+2OR | 5.199253 | 1.795774 | 0.233805 | 2.373123 | 3.632836 | 0.385413 | 10.764512 |
| 1.500 | M+4OR | 4.900811 | 0.992301 | 0.124289 | 2.220409 | 2.260309 | 0.234043 | 19.162525 |
| 2.000 | M | 7.667430 | 7.877528 | 1.000000 | 4.482488 | 9.133984 | 1.000000 | 2.305541 |
| 2.000 | M+1OR | 6.867032 | 2.072950 | 0.388259 | 2.045167 | 8.667350 | 0.921294 | 6.513833 |
| 2.000 | M+2OR | 7.629955 | 1.200956 | 0.283991 | 2.797711 | 3.363365 | 0.389716 | 10.688571 |
| 2.000 | M+4OR | 5.926702 | 0.810291 | 0.174774 | 2.416103 | 1.955422 | 0.223250 | 19.079604 |
| 2.500 | M | 8.024171 | 5.084677 | 1.000000 | 6.903737 | 6.088727 | 1.000000 | 2.258725 |
| 2.500 | M+1OR | 9.012770 | 2.055141 | 0.454478 | 2.229499 | 6.921965 | 1.250488 | 6.502951 |
| 2.500 | M+2OR | 8.212713 | 1.151956 | 0.220736 | 4.490669 | 2.110711 | 0.354390 | 10.677586 |
| 2.500 | M+4OR | 7.621565 | 0.678630 | 0.134421 | 4.277133 | 1.352094 | 0.210779 | 18.998472 |

### 10.2 Winning schedules by coupling and observable

- Beta = 1.5: plaquette winner = M (paired ESS/s ratio 1.000000); scalar winner = M (paired ESS/s ratio 1.000000).
- Beta = 2.0: plaquette winner = M (paired ESS/s ratio 1.000000); scalar winner = M (paired ESS/s ratio 1.000000).
- Beta = 2.5: plaquette winner = M (paired ESS/s ratio 1.000000); scalar winner = M+1OR (paired ESS/s ratio 1.250488).

Plain Metropolis was the plaquette winner at all three tested couplings.

For the scalar observable, Metropolis won at beta = 1.5 and beta = 2.0.

At beta = 2.5, M+1OR produced the highest measured mean scalar ESS/sec.

Its clean-release paired efficiency ratio was

1.250488.

This corresponds to an approximately

25.05%

measured advantage relative to the paired Metropolis baseline.

### 10.3 Plaquette efficiency

No hybrid schedule produced a mean paired plaquette ESS/s ratio above one in any tested beta regime.

This does not imply that the overrelaxation update failed to preserve its microcanonical property.

The numerical action-preservation audit remained at floating-point scale.

Rather, the measured decorrelation benefit did not offset the additional runtime cost for the plaquette observable.

### 10.4 Autocorrelation reduction and computational efficiency were not equivalent

Several hybrid conditions lowered estimated integrated autocorrelation time while simultaneously reducing ESS/sec.

The experiment therefore directly demonstrates, for the tested finite-lattice implementation, that

lower tau_int

does not imply

higher ESS/sec.

An update can improve decorrelation per measured hybrid update and still produce fewer effectively independent measurements per unit of wall-clock time.

### 10.5 Observable dependence

At beta = 2.5, the plaquette and scalar observable selected different winning schedules.

The plaquette continued to favor M.

The scalar observable favored M+1OR.

Therefore, within the tested campaign, schedule efficiency could not be described as a single observable-independent property of the transition algorithm.

A more precise conceptual description is:

efficiency
=
function of

transition kernel,

observable,

bare coupling,

implementation,

hardware.

The campaign does not identify a universal optimal update schedule.

### 10.6 Increasing overrelaxation frequency

The results do not support a monotonic rule that more overrelaxation produces greater runtime-normalized efficiency.

The four-overrelaxation schedule carried the largest update cost.

Although hybrid schedules could reduce estimated autocorrelation for particular observables, added microcanonical sweeps often produced diminishing or negative ESS/sec returns.

---

## 11. Discussion

### 11.1 Principal empirical result

The central finding is conditional rather than universal.

Plain Metropolis produced the highest measured mean plaquette ESS/sec at beta = 1.5, 2.0, and 2.5.

For the scalar composite observable, Metropolis also won at beta = 1.5 and beta = 2.0.

At beta = 2.5, M+1OR produced the greatest measured mean scalar ESS/sec.

The clean release efficiency ratio was 1.250488, corresponding to an approximately 25.05% measured advantage over the paired Metropolis baseline.

The correct conclusion is not that overrelaxation is always better.

It is that the computational benefit of a microcanonical update can depend materially on the measured observable and finite bare-coupling regime.

### 11.2 Why observables can favor different update schedules

A Markov transition acts on the full gauge configuration.

An observable maps that high-dimensional configuration to a scalar measurement.

Different observables can overlap differently with slowly evolving structures in the Markov chain.

The plaquette is local and closely connected to the action density.

The scalar composite combines purely spatial plaquette information across time slices.

A transition can therefore decorrelate one observable more effectively than another.

The current campaign does not perform a spectral decomposition of the Markov transition operator.

The winner mismatch nevertheless motivates observable-specific reporting of sampling efficiency.

### 11.3 Why ESS/sec matters

Integrated autocorrelation time is measured in chain-update units.

It does not account for computational cost.

For algorithm selection, a schedule should be evaluated against the resource actually consumed.

In this campaign the resource was wall-clock time.

An update schedule that reduces tau_int but multiplies runtime can lose ESS/sec.

This phenomenon appeared repeatedly in the measured results.

### 11.4 Implementation dependence

The absolute timings and efficiency rankings reported here are properties of the tested Python and NumPy implementation on the executing machine.

The project is designed for mathematical and computational transparency rather than production lattice-gauge performance.

A compiled C++, SIMD, GPU, or specialized implementation could change the relative cost of Metropolis and overrelaxation sweeps.

Therefore, the result should not be summarized as a universal statement that Metropolis is faster than overrelaxation.

The evidence supports the narrower statement that, under the tested implementation, added overrelaxation cost dominated plaquette ESS/sec at all tested beta values, while M+1OR achieved the highest measured scalar ESS/sec at beta = 2.5.

### 11.5 Coupling dependence

The scalar winner changed only at the largest tested beta.

The study does not interpret this as a phase boundary.

Only three bare couplings were studied, and no physical scale setting was performed.

The result instead motivates a finer beta scan near the observed winner change.

A larger campaign could determine whether the M+1OR scalar advantage emerges gradually, sharply, or irregularly.

### 11.6 Repeated campaign behavior

An earlier development campaign and the later clean committed release campaign selected the same qualitative pattern:

- M for the plaquette at beta = 1.5;
- M for the plaquette at beta = 2.0;
- M for the plaquette at beta = 2.5;
- M for the scalar observable at beta = 1.5;
- M for the scalar observable at beta = 2.0;
- M+1OR for the scalar observable at beta = 2.5.

The precise beta = 2.5 scalar ratio moved between executions.

The clean committed release campaign is the quantitative source for this manuscript.

Repeated qualitative agreement is reported as an observation, not as a formal significance test.

---

## 12. Validation Architecture

The final campaign sits above a layered master validation suite.

Its registered checks include:

1. SU(2) identity and random-element validation.
2. Cold-start plaquette validation.
3. Cold-start Wilson-action validation.
4. Cold-start Wilson-loop validation.
5. Metropolis preservation of SU(2).
6. Exact synthetic area-law Creutz recovery.
7. Lattice dimension and geometry counts.
8. SU(3) identity and random-element validation.
9. Gell–Mann basis validation.
10. Local gauge-invariance validation.
11. Correlation-aware resampling validation.
12. Scalar glueball-style connected-correlator validation.
13. Periodic spectroscopy validation.
14. Local-global Wilson-action consistency.
15. Spatial smearing pipeline validation.
16. Operator correlation-matrix validation.
17. Generalized eigenvalue validation.
18. Correlated spectroscopy-fit validation.
19. GEVP bootstrap state-matching validation.
20. Correlated principal-state fit validation.
21. Microcanonical overrelaxation validation.
22. Generic GaugeLattice backend validation.
23. SU(3) structural lattice validation.
24. Ensemble Creutz bootstrap validation.
25. Reproducibility manifest validation.
26. Controlled research-campaign validation.

The clean campaign recorded a maximum microcanonical action error of

1.598721155460e-14.

All numerical record fields were finite in the final release audit.

These checks support internal computational consistency.

They are not a proof of continuum field theory.

---

## 13. Limitations

The present study has substantial limitations.

First, the lattice is small and finite.

Second, the simulation is three dimensional rather than a continuum four-dimensional quantum Yang–Mills construction.

Third, only beta = 1.5, 2.0, and 2.5 are examined.

Fourth, only three independent deterministic seeds are used.

Fifth, each measured schedule contains 180 hybrid updates.

Sixth, the integrated autocorrelation estimator uses the laboratory's positive-sequence diagnostic rather than a more advanced automatic-window estimator.

Seventh, finite-chain ESS estimates can be unstable.

Eighth, wall-clock timing is machine and implementation dependent.

Ninth, the implementation is primarily Python and NumPy.

Tenth, the scalar measurement is glueball-style and is not claimed to be a complete production 0++ spectroscopy observable.

Eleventh, no continuum extrapolation is performed.

Twelfth, no infinite-volume extrapolation is performed.

Thirteenth, no physical scale is set.

Fourteenth, the three beta values are not physical scale-matched ensembles.

Fifteenth, no formal statistical-significance claim is made from three seeds.

Sixteenth, the study does not prove existence of four-dimensional quantum Yang–Mills theory.

Seventeenth, the study does not prove a positive Yang–Mills mass gap.

The reported contribution is a finite-lattice computational-methodology result under explicit conditions.

---

## 14. Future Work

The first extension should increase the number of independent seeds and the number of measured updates per chain.

A finer beta grid near beta = 2.5 should test the apparent scalar schedule transition.

More robust autocorrelation-window estimators should be compared.

Paired block-bootstrap or jackknife uncertainty should be propagated directly to ESS/sec ratios.

A formal uncertainty interval for the difference between hybrid and Metropolis efficiency would strengthen schedule comparisons.

Compiled implementations should benchmark whether reducing overrelaxation wall-clock cost changes the observed efficiency frontier.

The campaign should also be repeated for:

- individual smearing levels;
- multiple scalar basis operators;
- generalized-eigenvalue principal correlators;
- Wilson loops of different sizes;
- Creutz-ratio observables;
- SU(3) finite-lattice observables.

A deeper mathematical question is whether observable-specific optimal update frequency can be related to overlap with slowly mixing modes of the Markov transition operator.

The current work motivates that question but does not answer it.

---

## 15. Conclusion

This study compared SU(2) Metropolis and microcanonical-overrelaxation hybrid schedules using observable-specific effective sample size per measured wall-clock second.

The schedules were

M,

M+1OR,

M+2OR,

M+4OR.

The controlled campaign evaluated three finite bare Wilson couplings and three independent deterministic seeds.

Within every beta-seed pair, schedules began from copies of the same thermalized gauge configuration.

Across 36 measured chains, plain Metropolis produced the largest measured mean plaquette ESS/sec at all three beta values.

For the scalar glueball-style composite observable, Metropolis also won at beta = 1.5 and beta = 2.0.

At beta = 2.5, M+1OR produced the greatest measured mean scalar ESS/sec.

The clean committed release campaign measured a paired scalar efficiency ratio of

1.250488,

corresponding to an approximately

25.05%

advantage relative to the paired Metropolis baseline.

Several hybrid schedules lowered estimated integrated autocorrelation time without increasing effective sample size per second.

The central empirical conclusion is therefore:

> **Autocorrelation reduction and computational efficiency are distinct quantities, and the value of microcanonical overrelaxation can depend on both the measured observable and the finite bare-coupling regime.**

The result is implementation dependent, finite lattice, and descriptive.

It is not a continuum mass-gap result.

Its contribution is methodological: a validated and reproducible framework for comparing non-Abelian lattice-gauge update schedules using observable-specific statistical efficiency normalized by actual computational cost.

---

## Data and Code Availability

The computational laboratory, validation suite, experiment code, and reproducibility infrastructure are available at:

https://github.com/Elsbernd09/yang-mills-mass-gap-lab

The final campaign configuration hash is:

`379d46d5354be739`

The clean release campaign was executed from Git commit:

`6b8ea687a07ece302bfa2c6e6d3a5c31ff01b458`

The campaign manifest records the Python environment, package versions, Git provenance, runtime, run status, and generated output files.

---

## Author Statement

This manuscript reports a finite-lattice computational-methodology study performed within the Yang–Mills Mass Gap Laboratory project.

The work does not claim a solution of the Clay Mathematics Institute Yang–Mills existence and mass-gap problem.

The quantitative campaign conclusions are derived from the newest completed clean-Git final release campaign selected by the manuscript-generation pipeline.
