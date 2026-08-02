# Adaptive Power-Blend Heuristic Experiment Framework

**Status:** Proposed execution basis

**Date:** 2026-08-02

## 1. Purpose

Build an unattended, reproducible framework for exploring the adaptive
power-blend model across preset light-field datasets. The framework must expose
when the challenger is useful, when the model falls back to the anchor, and how
trust, clipping, and smoothing interact. It must not select a winner from image
sharpness alone or hide a muted blend behind a superficially stable result.

The framework extends the repository's existing parameter-sweep machinery. It
reuses durable resume, adaptive memory scheduling, generation-aware outputs,
anchor/challenger field caching, workspace artifacts, metric records, the sweep
visualizer, and the metric explorer. It does not introduce a second general
sweep engine.

## 2. Goals

1. Characterize the complete useful operating region of the adaptive blend.
2. Separate the effects of trust, cap, and smoothing before testing their joint
   interactions.
3. Detect both failure extremes:
   - **muting:** a valid challenger produces little or no final correction;
   - **overreach:** corrections become unstable, discontinuous, or physically
     implausible.
4. Select configurations that are robust across datasets, including difficult
   low-texture, occlusion, depth-boundary, and large-disparity cases.
5. Run to completion without operator decisions, resume safely after
   interruption, and emit useful partial reports.
6. Produce an inspection experience that explains *why* a candidate behaved as
   it did, not only where it ranked.

## 3. Non-goals

- Replacing the existing rectification matcher or global alignment stage.
- Reintroducing best-versus-second-best score-gap confidence.
- Running an iterative optical-flow or variational solver.
- Treating angular coherence as a hard physical law at occlusions or
  non-Lambertian regions.
- Optimizing one dataset independently and then presenting it as a shared
  solution.
- Automatically changing the search space while a generation is running.

## 4. Model under test

For pixel `x` and angular view `v`, define the challenger residual:

\[
\Delta(x,v)=d_c(x,v)-d_a(x,v)
\]

where `d_a` is the power-1 anchor field and `d_c` is the raised-power
challenger field.

The model derives two independent support values:

- `C_angular(x,v)`: robust agreement with the local light-field angular model;
- `C_spatial(x,v)`: robust agreement with an edge-aware local vector-median
  model.

### 4.1 Support construction

The angular prediction is the fast light-field model:

\[
\widehat{\Delta}(x,v)=A(x)(q_v-q_{ref})
\]

where `q_v` is the two-dimensional angular coordinate. Estimate the `2 x 2`
matrix `A(x)` by a vectorized least-squares fit using a precomputed angular
pseudoinverse, followed by exactly one Tukey-reweighted refit. This is a bounded
two-pass calculation, not an iterative solver. Mark angular evidence unavailable
when the angular design is rank-deficient or the reweighted fit has insufficient
support.

The spatial prediction is the edge-aware vector median of neighboring residual
vectors. Guide-compatible neighbors come from the unpowered reference image.
Mark spatial evidence unavailable when too few finite, guide-compatible
neighbors remain.

For either available channel `i`, compute its vector prediction error `e_i`, a
local robust scale `s_i = 1.4826 * MAD(e_i)`, and:

\[
z_i=\frac{e_i}{\max(s_i,s_{floor})}
\]

The scale floor is fixed in the experiment spec and prevents numerical collapse
in perfectly coherent or very small neighborhoods. Convert normalized error to
support with Tukey's biweight:

\[
C_i=
\begin{cases}
\left(1-(z_i/\kappa_i)^2\right)^2,&z_i<\kappa_i\\
0,&z_i\geq\kappa_i
\end{cases}
\]

`kappa_i` is the channel's cutoff in robust-scale units. This supplies a smooth
support decay with an explicit contradiction point and does not depend on
matching-score gaps.

They combine by a soft logical OR, represented initially by `max`:

\[
S(x,v)=\max(C_{angular}(x,v),C_{spatial}(x,v))
\]

A residual is hard-rejected only when every available evidence channel strongly
contradicts it. If `A(x,v)` is the set of available channels:

\[
H(x,v)=[\mathcal{A}(x,v)=\varnothing]\;\lor\!
\bigwedge_{i\in\mathcal{A}(x,v)}[C_i=0]
\]

Missing evidence is unavailable, not failed evidence. For example, an angular
fit that cannot be estimated leaves the spatial channel responsible for the
decision. When neither channel is available, the safe result is exact anchor
fallback.

For non-rejected residuals, dynamic impact has a guaranteed floor:

\[
\alpha_{eff}=\alpha_{min}+(\alpha_{max}-\alpha_{min})
\operatorname{smoothstep}(S;\gamma)
\]

The local cap is a robust trust-region radius:

\[
c(x)=\operatorname{clamp}
\left(
\operatorname{median}_{N(x)}\|\Delta\|+
k_c\operatorname{MAD}_{N(x)}\|\Delta\|,
c_{min},c_{max}
\right)
\]

The raw update is:

\[
u_0(x,v)=
\begin{cases}
0,&H(x,v)\\
\alpha_{eff}\operatorname{clipNorm}(\Delta,c(x)),&\text{otherwise}
\end{cases}
\]

An edge-aware normalized filter `G_I` uses the unpowered reference image as its
guide. For smoothing strength `lambda`, compute:

\[
u_s=(1-\lambda)u_0+\lambda
\frac{G_I(Su_0)}{G_I(S)+\epsilon}
\]

It may fill locally ambiguous, non-rejected pixels from supported same-surface
neighbors. It must not fill hard-rejected pixels or propagate across strong
guide edges. After filtering, set hard-rejected updates back to zero and project
every other update to the local bound `alpha_max * c(x)` before forming:

\[
d_f=d_a+u
\]

This assignment of responsibilities is fixed during the first experiment:

- trust evidence controls rejection and alpha boost;
- local residual statistics control the cap;
- reference-image edges control propagation boundaries;
- no score-gap term attenuates the update;
- ordinary uncertainty cannot reduce impact below `alpha_min`;
- only explicit multi-channel contradiction produces exact anchor fallback.

## 5. Search-strategy decision

### 5.1 Considered strategies

**Full factorial grid.** Easy to understand, but the trust, cap, and smoothing
axes produce a combinatorial explosion. Most runs repeat uninformative extreme
combinations.

**Bayesian optimization.** Efficient for finding a scalar optimum, but opaque
for mechanism discovery, awkward with multiple safety/quality objectives, and
liable to overfit the selected datasets and ranking profile.

**Staged deterministic design of experiments.** First isolate each mechanism,
then cover joint space with a deterministic low-discrepancy design, and finally
refine representative Pareto seeds. This is the selected strategy because it
supports interpretation, broad coverage, reproducibility, and bounded cost.

### 5.2 Search invariants

- Candidate generation is deterministic from the normalized spec and seed.
- All successful candidates in a phase receive the same development-dataset
  coverage; there is no metric-driven early stopping in version 1.
- Raised power is a field-production axis. Blend candidates reuse cached field
  pairs and do not rerun matching.
- Parameters are expressed in pixels or robust normalized units so settings
  transfer across datasets.
- Structural alternatives are tested as explicit ablations rather than silently
  mixed into continuous tuning.
- Each phase has a hard candidate budget and deduplicates candidates by their
  canonical normalized configuration.

## 6. Dataset manifest

The experiment consumes a required, versioned dataset manifest. Every entry
contains:

- stable dataset ID;
- immutable input identity and path;
- role: `development` or `validation`;
- angular grid coordinates and reference-view identity;
- normalization policy;
- scene tags selected from `low_texture`, `high_texture`, `depth_edges`,
  `occlusion`, `specular`, `large_disparity`, and `photometric_range`;
- optional fixed diagnostic crops or regions of interest;
- optional validation-oracle inputs already supported by the repository.

The default policy requires at least three development datasets and one held-out
validation dataset. All scene tags present in validation must appear at least
once in development. The runner fails preflight if paths, input identities,
angular coordinates, shapes, or roles are invalid.

The validation set is invisible to candidate generation and development
ranking. It is used only after the development shortlist has been frozen.

## 7. Parameter groups

### 7.1 Field-production controls

These are expensive and therefore cached independently of blend parameters:

| Parameter | Initial exploration |
| --- | --- |
| baseline power | fixed at `1.0` |
| raised power | `1.5`, `2.0`, `2.5` |
| symmetric reference powering | fixed enabled |
| matching and combined-registration settings | fixed to [the approved symmetric power-sweep baseline](2026-07-28-symmetric-power-sweep-design.md) |

Direct anchor, direct challenger, and the current fixed-alpha/fixed-cap blend
are mandatory controls for every raised-power arm.

### 7.2 Trust controls

| Knob | Role | Initial bounds/levels |
| --- | --- | --- |
| `alpha_min` | guaranteed non-rejected impact | `0.10` to `0.50` |
| `alpha_max` | maximum supported impact | `0.50` to `1.00`, constrained `>= alpha_min` |
| `support_gamma` | response curvature | `0.5`, `1.0`, `2.0` |
| `angular_cutoff_mad` | Tukey support reaches zero | `1.5` to `4.0` robust scales |
| `spatial_cutoff_mad` | Tukey support reaches zero | `1.5` to `4.0` robust scales |
| `spatial_radius_px` | vector-median neighborhood | `1`, `3`, `5` pixels |

The `max` evidence fusion and AND rejection rule are fixed in the primary model.
`angular-only`, `spatial-only`, and multiplicative-fusion variants are retained
as diagnostic ablations, not tunable production candidates.

### 7.3 Cap controls

| Knob | Role | Initial bounds/levels |
| --- | --- | --- |
| `cap_min_px` | prevents a zero trust region | `0.10` to `0.75` pixels |
| `cap_mad_multiplier` | expands the local robust radius | `0.5` to `3.0` |
| `cap_max_px` | absolute local safety ceiling | `0.75` to `3.0` pixels |
| `cap_radius_px` | neighborhood for median/MAD | tied to or one level above `spatial_radius_px` |

The runner enforces `cap_min_px <= cap_max_px <= max_displacement`. A fixed-cap
variant using the previous mechanism is a mandatory ablation.

### 7.4 Smoothing controls

| Knob | Role | Initial bounds/levels |
| --- | --- | --- |
| `smooth_strength` | interpolation between raw and guided update | `0.0` to `0.75` |
| `smooth_radius_px` | spatial support | `1`, `3`, `5`, `9` pixels |
| `guide_edge_scale` | resistance to crossing intensity edges | robust gradient quantiles `0.50` to `0.95` |
| `min_propagation_support` | avoids unstable normalized division | `0.05` to `0.30` |

`smooth_strength=0` is always included. Gaussian update smoothing and no
smoothing are diagnostic ablations. The production candidate uses edge-aware
normalized smoothing and reprojects the result to the local update bound.

## 8. Experiment phases

Candidate counts below are defaults. The spec may lower or raise budgets while
preserving phase ordering and mandatory controls.

### Phase 0: preflight, field cache, and calibration

1. Validate the dataset manifest and experiment fingerprint.
2. Compute or reuse the anchor and raised-power challenger fields.
3. Record residual-magnitude, angular-error, spatial-error, and image-gradient
   distributions per dataset.
4. Resolve robust normalized thresholds into dataset-specific physical values.
5. Run mandatory controls and structural ablations.

Default blend/control budget: 16 candidates. This phase must complete before
later phases can be generated.

### Phase 1: mechanism-isolation screening

Run three bounded subphases:

1. **Trust screen:** permissive cap, smoothing off; vary trust knobs.
2. **Cap screen:** use representative low/medium/high trust seeds; vary cap
   knobs.
3. **Smoothing screen:** use representative trust-cap seeds; vary smoothing
   knobs.

Each subphase includes conservative, balanced, and aggressive sentinel points.
Use an orthogonal or maximin design within the active parameter group rather
than a full cross-product.

Default budget: 72 candidates total.

### Phase 2: joint trade-off coverage

Generate a deterministic Sobol or maximin Latin-hypercube design across the
continuous trust, cap, and smoothing axes. Add explicit corner cases only when
they are not already covered. Treat raised power and the small discrete radius
sets as balanced categorical strata.

Default budget: 64 candidates.

This phase estimates interactions such as:

- high `alpha_min` with a tight adaptive cap;
- low rejection thresholds with stronger smoothing;
- permissive caps with small edge-aware neighborhoods;
- raised power versus actual transferred residual energy.

### Phase 3: Pareto-guided local refinement

Freeze the Phase 0-2 development records. Select up to 12 diverse seeds from
the feasible Pareto surface, including:

- the balanced knee;
- best quality under the stability constraints;
- lowest muting among stable candidates;
- smoothest field without quality regression;
- conservative and aggressive boundary representatives;
- at least one representative per raised-power arm that remains competitive.

Generate one-axis and selected two-axis half-step neighbors around these seeds.
Do not produce a local Cartesian product. Deduplicate and stop at the fixed
budget.

Default budget: 64 candidates.

### Phase 4: held-out validation and materialization

Freeze a shortlist of no more than 12 candidates before accessing validation
data. Include the balanced recommendation, conservative/aggressive alternatives,
the anchor, direct challengers, the legacy fixed blend, and relevant ablations.

Run the shortlist on every validation dataset. Select the final recommendation
using robust cross-dataset aggregation and worst-case constraints. Materialize
full diagnostic artifacts for the final shortlist; non-shortlisted development
candidates retain compact artifacts.

## 9. Evaluation model

### 9.1 No single winning metric

Candidate selection uses hard feasibility checks followed by Pareto analysis.
A scalar robust score may order work inside a phase, but it does not erase the
individual objectives or diagnostics.

Primary objective families are:

1. **Image outcome:** existing image metrics, optional validation oracle, and
   human-review compatibility.
2. **Inter-view consistency:** mean, p95, and worst-view rectified disagreement.
3. **Update effectiveness:** how much supported challenger residual survives.
4. **Field stability:** spatial regularity, angular consistency, bounds, and
   geometric validity.
5. **Robustness:** mean performance, dispersion, and worst-dataset behavior.
6. **Cost:** matching cache misses, blend time, rectification time, peak memory,
   and materialization time.

### 9.2 Mandatory adaptive-blend diagnostics

Every candidate records aggregate and per-view statistics for:

- residual opportunity fraction: pixels where the challenger differs
  meaningfully from the anchor and at least one support channel agrees;
- transferred-residual ratio
  `||u|| / (||Delta|| + epsilon)` on opportunity pixels;
- no-op fraction on opportunity pixels;
- exact anchor-fallback fraction;
- support-source fractions: `both`, `angular_only`, `spatial_only`, `neither`;
- `C_angular`, `C_spatial`, combined support, and effective-alpha quantiles;
- adaptive-cap quantiles and cap-hit fraction;
- pre-smoothing versus post-smoothing update energy;
- propagation fraction and propagation distance;
- update-gradient mean, p95, and maximum;
- boundary saturation and out-of-bounds coordinate fractions;
- non-finite fraction;
- displacement Jacobian fold/sign-violation fraction where defined;
- edge-leakage score: update mixing across strong reference-image edges;
- per-angular-view residual-model error and outlier fraction.

The key empirical aggressiveness coordinate is the median transferred-residual
ratio, not a weighted sum of parameter values. Reports use it as a primary plot
axis.

### 9.3 Muting and overreach flags

A candidate receives a `muted` flag when all are true:

- residual opportunity coverage is non-trivial;
- median transferred-residual ratio is below the configured minimum;
- image/inter-view outcomes do not improve enough to justify the no-op.

A candidate receives an `overreach` flag when any hard safety constraint fails,
including non-finite output, excessive coordinate violations, excessive field
folds, or configured extreme update-gradient behavior.

Thresholds are declared in the experiment spec and copied verbatim into every
report. They are never inferred after viewing results.

### 9.4 Robust aggregation

Within each dataset, metrics are robustly normalized against mandatory controls.
Across datasets, ranking reports:

- median normalized outcome;
- lower-quartile and worst-dataset outcome;
- dataset-to-dataset dispersion;
- number of wins/ties/losses versus anchor and legacy blend;
- rank stability under leave-one-dataset-out aggregation.

Missing or failed dataset runs are penalized and shown explicitly. A candidate
cannot be final if it lacks complete validation coverage.

## 10. Artifact policy

The field cache is the expensive, reusable layer. Blend configuration IDs do not
enter its identity. Its identity includes dataset content, reference mode,
matching parameters, raised power, power application mode, and algorithm/cache
version.

To keep the broad search economical:

### Compact artifacts for every candidate

- canonical config and provenance;
- scalar metric record;
- compact preview;
- aggregate adaptive-blend diagnostics;
- optional downsampled diagnostic maps;
- runtime and memory telemetry;
- status, warning, and failure records.

### Full artifacts for controls and shortlist candidates

- rectified view stack and average image;
- anchor, challenger, and final displacement fields;
- raw and clipped residual fields;
- angular and spatial support maps;
- hard-reject and opportunity masks;
- effective-alpha and adaptive-cap maps;
- pre/post-smoothing update fields;
- propagation-support map;
- full workspace in NPZ/MAT-compatible form.

Shortlist materialization is a deterministic final pass and is resumable. It
must reproduce the compact-pass metrics within configured numerical tolerance.

## 11. Unattended execution and recovery

The framework accepts one normalized experiment spec and exposes one primary
CLI action. GUI configuration may generate the same spec but is not required for
execution.

The run lifecycle is:

1. strict preflight and dry-run report;
2. deterministic candidate manifest generation;
3. cached field production;
4. phase execution with the existing adaptive memory scheduler;
5. atomic phase checkpoint and provisional report;
6. automatic transition to the next phase;
7. shortlist validation and full materialization;
8. final report export.

Required operational behavior:

- content-addressed candidate IDs and experiment fingerprint;
- exact-spec resume and compatible extension through existing generation rules;
- atomic JSON/CSV/NPZ writes and backup behavior;
- corrupt cache detection followed by isolated recomputation;
- individual failure capture without aborting unrelated candidates;
- bounded retry for transient I/O or worker failures;
- memory enforcement and worker recycling inherited from the current runner;
- deterministic seed recorded in the manifest;
- partial leaderboards and reports generated after every phase and on graceful
  interruption;
- final non-zero process status if required validation coverage is incomplete,
  even though partial reports remain available.

No operator choice is required after a successful preflight.

## 12. Result-review experience

The primary deliverable is a self-contained static HTML report generated
automatically at the end of every phase. The existing desktop visualizer should
load the same report model for deeper inspection.

### 12.1 Overview / Pareto cockpit

- quality versus transferred-residual ratio;
- stability versus quality;
- color by smoothing strength or hard-fallback fraction;
- marker shape by raised power;
- marker size by runtime;
- visible anchor, challenger, and legacy controls;
- filters for phase, dataset, scene tag, power arm, status, and flags;
- labeled balanced, conservative, and aggressive recommendations.

### 12.2 Trade-off explorer

- one-parameter sensitivity and grouped value effects;
- trust-cap, trust-smoothing, and cap-smoothing heatmaps;
- empirical aggressiveness distributions;
- partial-dependence-style summaries from the joint design;
- phase-to-phase search coverage and refinement lineage;
- ablation deltas for angular support, spatial support, adaptive cap, and
  edge-aware smoothing.

Plots must show sample counts and coverage warnings so sparse interactions are
not mistaken for strong evidence.

### 12.3 Dataset robustness

- candidate-by-dataset normalized score heatmap;
- candidate-by-dataset transferred-residual heatmap;
- worst-case and lower-quartile outcomes;
- leave-one-dataset-out rank stability;
- scene-tag slices;
- validation-only panel clearly separated from development results.

### 12.4 Synchronized candidate inspector

Pin candidates A and B and inspect a common dataset, view, crop, contrast, zoom,
and pan across synchronized panels:

- rectified average and individual rectified view;
- anchor, challenger, and final displacement magnitude/vector field;
- raw residual and applied update magnitude/vector field;
- angular, spatial, and combined support;
- hard fallback and residual-opportunity masks;
- effective alpha and adaptive cap;
- pre/post-smoothing update and propagation support;
- difference against anchor, challenger, or a validation image;
- metric and diagnostic deltas with the preferred side identified.

Fixed diagnostic crops from the dataset manifest appear first, while arbitrary
zoom remains available. Identical normalization and color limits are used for A
and B unless the user explicitly unlocks them.

### 12.5 Pixel/angular microscope

For a selected pixel or small region, show:

- anchor, challenger, clipped, and final vectors for every angular view;
- observed residuals and angular-model predictions;
- per-view angular/spatial support and fallback state;
- local median/MAD cap construction;
- smoothing contribution from neighboring pixels;
- a compact explanation of the exact decision path.

This view is essential for distinguishing legitimate occlusion fallback from a
globally muted model.

### 12.6 Report outputs

The report export contains:

- `index.html`;
- candidate, dataset, Pareto, sensitivity, ablation, and failure CSV tables;
- machine-readable report context and recommendation JSON;
- reusable NPZ/MAT visualization bundles;
- preview and diagnostic assets;
- copied normalized experiment spec and dataset-manifest identity;
- warnings and coverage report;
- a concise `README.md` with commands to reopen the result in the desktop
  visualizer and metric explorer.

## 13. Recommendation policy

The framework emits three named recommendations rather than pretending there is
one universally correct operating point:

- **balanced:** Pareto knee with complete coverage and no safety flags;
- **conservative:** lowest transferred-residual ratio that still demonstrates a
  material improvement over anchor/legacy controls;
- **aggressive:** highest transferred-residual ratio that remains inside all
  stability constraints and does not regress the worst dataset materially.

The balanced candidate is the default downstream choice. Each recommendation
records the control deltas, dataset wins/losses, relevant flags, and nearest
Pareto alternatives. If no candidate satisfies the feasibility rules, the
framework recommends the anchor and reports why the adaptive model failed.

## 14. Test and verification strategy

### 14.1 Model-level synthetic cases

- perfectly angular-coherent residual field;
- spatially coherent but angular-incoherent occlusion region;
- angularly coherent depth edge with spatial discontinuity;
- isolated residual outlier;
- globally incoherent residual field;
- flat/zero residual field;
- non-finite anchor or challenger values;
- strong guide edge separating two valid residual regions;
- ambiguous pixel filled from same-surface neighbors;
- hard-rejected pixel that remains exact anchor after smoothing.

### 14.2 Properties

- `0 <= alpha_eff <= alpha_max <= 1`;
- `cap_min <= c(x) <= cap_max`;
- no non-finite final field from finite inputs;
- hard-rejected updates are exactly zero;
- final smoothed updates respect the local projected bound;
- disabling smoothing reproduces the raw adaptive update;
- fixed-mode compatibility reproduces the current fixed alpha/cap blend;
- candidate generation is deterministic and budget bounded;
- field-cache identity excludes blend-only parameters and includes every
  field-producing parameter.

### 14.3 Framework integration

- dry run lists exact candidates and dataset coverage;
- interruption and exact resume do not duplicate completed work;
- corrupt compact and full artifacts are independently recomputed;
- phase transitions and shortlist freezing are durable;
- validation data cannot affect development candidate generation;
- compact and materialized metrics agree within tolerance;
- partial and final HTML reports open with missing/corrupt assets represented as
  warnings;
- static report and desktop visualizer use the same canonical report model.

## 15. Downstream implementation boundaries

Implementation should be split into independently testable modules:

1. **Adaptive blend kernel:** support, rejection, alpha, cap, smoothing, and
   diagnostics from a cached field pair.
2. **Blend experiment schema:** typed parameter groups, validation, canonical
   IDs, phase budgets, and dataset roles.
3. **Candidate designer:** controls, ablations, group screens, low-discrepancy
   joint design, and Pareto-neighbor refinement.
4. **Blend evaluator:** compact execution, metrics, flags, aggregation, and
   shortlist materialization.
5. **Report model:** canonical tables, maps, provenance, and recommendations.
6. **Report UX:** headless HTML export and desktop visualizer extensions.
7. **Runner integration:** existing cache, scheduler, state, generation, resume,
   and reporting hooks.

The adaptive blend kernel must not depend on the sweep engine or GUI. The
candidate designer must operate on typed search bounds and prior phase records,
not filesystem layout. Both report front ends must consume the same report
model.

## 16. Acceptance criteria

The framework is ready for experimental use when:

1. One preset spec can run all phases unattended and resume after interruption.
2. Matching fields are computed at most once per unique dataset/field-producing
   configuration unless corruption is detected.
3. Mandatory controls, ablations, development coverage, and held-out validation
   are enforced automatically.
4. The report makes muted and overreaching candidates immediately identifiable.
5. A reviewer can trace any shortlisted pixel from raw residual through support,
   cap, alpha, smoothing, and final displacement.
6. Balanced, conservative, and aggressive recommendations include complete
   provenance and cross-dataset evidence.
7. Existing fixed power blending remains available and numerically unchanged.
8. The final selection does not depend solely on sharpness or on a hidden scalar
   score.
