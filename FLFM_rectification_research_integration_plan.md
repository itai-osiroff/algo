# FLFM Rectification Research Integration Plan

## Purpose

This document translates the attached research papers and research notes into a concrete plan for improving the current FLFM rectification workflow in this repository, with special attention to:

- global rectification
- pixelwise rectification
- in-loop scoring
- final validation
- staged adoption without deep learning

This is a planning document only. It does not propose immediate code edits.

## Scope and assumptions

- The target regime is sparse fluorescent FLFM data dominated by point-like or near point-like emitters.
- The current workflow already separates global alignment from pixelwise refinement and already supports sweep-based experimentation.
- The goal is not to maximize generic image sharpness. The goal is to improve geometric consistency and biological usefulness while staying lightweight, explainable, and computationally tractable.
- Recommendations below are strongest where multiple sources agree. Where the evidence is weaker or indirect, the document marks the item as exploratory.

## Current repository workflow snapshot

Based on the current repository:

- `moment_images/rectification_strategy.py` already separates:
  - global translation rectification based on cross-correlation of raw views
  - pixelwise patch rectification based on local displacement search
  - optional local metric families using intensity, gradient, Laplacian, or composite blurred correlation scores
  - optional sigma-pyramid refinement for pixelwise matching
- `moment_images/parameter_sweep.py` already supports:
  - intensity-only baseline ranking
  - optional metric-formulation experiments on shortlisted candidates
  - two-stage experimental flow
- `moment_images/sweep_analysis.py` already records quality metrics such as:
  - average intensity
  - max intensity
  - total variation
  - gradient energy
  - Laplacian variance
  - level-2 Haar DWT HV energy
- `moment_images/validate_rectification.py` already supports external-camera validation.

This is a strong starting point. The papers do not argue for replacing the workflow structure. They argue for changing what image representation is scored, which metrics should be trusted in-loop, and how final acceptance should be separated from local optimization.

## Executive summary

The strongest conclusion across the attached material is this:

**For sparse fluorescent FLFM rectification, the workflow should optimize detector-like guide maps rather than raw image aesthetics.**

That leads to five core design rules:

1. Preprocess for photon-limited fluorescence before rectification scoring.
2. Build a guide map that makes emitters easy to detect and hard for noise to imitate.
3. Use one family of metrics for in-loop optimization and a different family for final acceptance.
4. Use strong global initialization before pixelwise refinement.
5. Evaluate candidate rectifications by both local sharpness and cross-view consistency.

The most defensible default stack is:

- photometric cleanup
- local background suppression
- variance stabilization when needed
- light denoising
- matched-filter or multiscale LoG/DoG guide map
- guide-map-based global alignment
- guide-map-assisted pixelwise refinement with iterative local-average updating
- in-loop score built from confident peaks, guide-map sharpness, sparsity/concentration, width penalty, and cross-view consistency
- final acceptance based on raw-image resolution and localization quality, not on the transformed guide map alone

## What the source material contributes

### 1. `Lightweight Non-Deep-Learning Metrics and Guide Maps for Sparse Fluorescent FLFM Rectification.pdf`

Main contribution:

- The most relevant source for the target regime.
- Recommends detector-like guide maps over raw natural-image sharpness metrics.
- Strongly favors matched filtering, LoG/DoG, wavelet puncta maps, Tenengrad on the guide map, and final checks based on localization or PSF-fit behavior.
- Explicitly advises against using raw entropy, raw variance of Laplacian, or unconstrained deconvolution objectives as the primary rectification target.

Direct implication:

- The rectification loop should operate on a cleaned guide map, not directly on raw view intensities.

### 2. `FLFM Rectification_ Metrics and Maps.pdf`

Main contribution:

- Provides an explicit staged pipeline:
  - variance stabilization
  - LoG or Hessian guide-map construction
  - Vollath F4 as the in-loop objective on the guide map
  - optional cross-view geometric validation
  - 1FRC as a post-rectification absolute quality metric
- Distinguishes highly recommended in-loop metrics from final-evaluation metrics.
- Gives a useful shortlist for sparse fluorescent point emitters:
  - Vollath F4 for in-loop scoring
  - Hoyer and Gini as sparsity metrics
  - 1FRC for final evaluation
  - LoG as the default guide map
  - Hessian/anisotropic maps when aberration elongates peaks

Direct implication:

- This source provides the clearest blueprint for how to redesign the scoring hierarchy without changing the overall workflow shape.

### 3. `wavelet based image fusion in multi view 3D microscopy.pdf`

Main contribution:

- Shows that wavelet-domain fusion can combine complementary directional information from multiple registered views without requiring a fixed PSF.
- Uses scale- and orientation-aware decision maps.
- Suppresses noise by using minimum activity selection at the finest scale and maximum activity selection at coarser informative scales.

Direct implication:

- Wavelets are useful here not only for final fused reconstruction, but also for:
  - building guide maps
  - fusing multi-scale confidence maps
  - constructing robust coarse-to-fine global objectives

### 4. `sparsity metrtics for autofocusing.pdf`

Main contribution:

- Demonstrates that sparsity metrics become much more interpretable after high-pass filtering.
- Shows that many apparently different focus metrics cluster into similar behavior after proper filtering.
- Indicates that global concentration metrics can become reliable once low-frequency background is removed.
- Also shows that very simple measures can perform surprisingly well after the representation is cleaned.

Direct implication:

- Sparsity should not be measured on raw FLFM data.
- If Hoyer, Gini, or related concentration metrics are used, they should be applied to filtered guide maps or confident-peak maps.

### 5. `light sheet microscopy focusing.pdf`

Main contribution:

- Demonstrates an adaptive imaging framework that continuously optimizes image quality using a benchmarked metric.
- Finds that a use-case-specific metric can outperform generic focus measures when evaluated by:
  - localization accuracy
  - capture range
  - density of local maxima
  - processing speed
- Uses sparse reference locations and low-bandwidth monitoring rather than brute-force full-volume optimization.

Direct implication:

- For this repository, metric selection should be empirical and task-specific.
- The sweep framework should rank metrics not only by final sharpness but also by:
  - stability of the objective landscape
  - false-optimum rate
  - sensitivity to noise/background
  - compute cost per candidate

### 6. `Journal of Microscopy - 2010 - OSIBOTE - Automated focusing in bright-field microscopy for tuberculosis detection.pdf`

Main contribution:

- Among classical autofocus measures, Vollath F4 performed best overall on full stacks.
- Under curve fitting, Vollath F4 and Brenner were strongest.
- Good metrics are judged not only by peak value, but by:
  - accuracy
  - range
  - peak width
  - false maxima
  - speed

Direct implication:

- Vollath F4 is a strong candidate for the rectification loop because it balances sharpness with robustness and tends to avoid noisy local maxima.
- The sweep framework should explicitly measure objective smoothness and false peaks, not only final image quality.

### 7. `An_automated_parallel_image_registration_technique_based_on_the_correlation_of_wavelet_features.pdf`

Main contribution:

- Uses maxima of wavelet coefficients as the feature space for registration.
- Uses multiresolution coarse-to-fine search for rotation and translation.
- Shows that high-frequency wavelet features can support correlation-based registration with better speed and competitive accuracy.

Direct implication:

- Global FLFM rectification should consider a wavelet- or guide-map-based pyramid for coarse alignment instead of relying only on raw intensity correlation.
- This is especially relevant if the future global model expands from pure translation to translation + rotation + weak affine correction.

### 8. `An Information-theoretic method for multimodality medical image registration.pdf`

Main contribution:

- Tsallis-entropy-based divergence produced smoother registration landscapes, larger capture range, and better noise robustness than MI/NMI in its target setting.

Direct implication:

- This is not the first-choice metric for same-modality sparse FLFM view alignment.
- It is more relevant as an exploratory fallback for:
  - cross-camera validation
  - cross-representation alignment
  - cases where guide-map intensity relationships differ enough that correlation becomes brittle

## Key design decision: optimize a representation, not the raw image

The current code can already optimize raw intensity, gradient, Laplacian, or blended local scores. The paper set suggests a more important reframing:

**The main improvement should come from changing the representation being aligned and scored, not just from swapping one scalar metric for another on the raw image.**

Recommended representation stack:

1. Raw view
2. Photometrically corrected view
3. Background-suppressed view
4. Variance-stabilized and lightly denoised view
5. Guide map
6. Confident-peak map and derived peak attributes

The guide map should drive optimization. The raw rectified view should drive final acceptance.

## Recommended target pipeline

### Stage 0. Input assumptions and data split

Relevant sources:

- both internal FLFM documents
- wavelet registration paper
- light-sheet adaptive focusing paper

Benefit:

- Separates calibration-time assumptions from runtime optimization.
- Prevents the metric from absorbing camera artifacts that should have been corrected upstream.

How to incorporate:

- Treat sensor gain, dark offset, read noise, and expected PSF radius as configuration-level priors.
- Maintain a clean distinction between:
  - calibration data
  - optimization data
  - final validation data

### Stage 1. Photometric preprocessing

Relevant sources:

- both internal FLFM documents
- sparsity-metrics paper

Benefit:

- Reduces the chance that rectification improves the metric by amplifying background or shot noise.
- Makes guide-map construction more physically meaningful.

How to incorporate:

- Apply dark correction, flat-field correction, and bad-pixel handling before any alignment metric is computed.
- If the data are strongly photon-limited, apply variance stabilization such as generalized Anscombe-style preprocessing.
- Apply only light denoising. The purpose is to stabilize weak emitters, not to invent structure.

Recommendation strength:

- Strong.

### Stage 2. Background suppression

Relevant sources:

- both internal FLFM documents
- sparsity-metrics paper

Benefit:

- Makes global sparsity and detector-style metrics meaningful.
- Prevents haze or tissue background from dominating the score.

How to incorporate:

- Prefer local background removal before guide-map generation.
- Candidate methods:
  - rolling-ball background subtraction
  - white top-hat / morphological top-hat with support larger than the emitter
  - wavelet-based background-noise subtraction
- Use local or background-adaptive thresholds rather than a single global threshold.

Recommendation strength:

- Strong.

### Stage 3. Guide-map generation

Relevant sources:

- both internal FLFM documents
- wavelet fusion paper

Benefit:

- Converts the optimization target from "looks sharp" to "contains confident emitter evidence."
- Improves separability between true emitters and noise.

How to incorporate:

- Default guide map:
  - multiscale LoG or DoG on the cleaned image
- Preferred guide map when PSF is known:
  - matched filter or small PSF kernel bank
- Preferred guide map when aberration elongates emitters:
  - Hessian eigenvalue map, anisotropic gLoG, or anisotropic filter bank
- Optional alternative or supplement:
  - wavelet puncta map

Practical rule:

- The default production path should be:
  - LoG/DoG when no trusted PSF prior is available
  - matched filter when a stable PSF prior exists
  - Hessian/anisotropic bank when emitter shape is clearly elongated

Recommendation strength:

- Strong.

### Stage 4. Global rectification

Relevant sources:

- wavelet-feature registration paper
- both internal FLFM documents
- Osibote paper

Benefit:

- A better global initialization reduces the search burden and failure rate of pixelwise refinement.
- Coarse global correction prevents local patch search from explaining large rigid misalignment as local flow.

How to incorporate:

- Replace or augment raw-intensity correlation with guide-map-based global similarity.
- Run global alignment on:
  - guide maps directly, or
  - wavelet maxima / guide-map pyramid features
- Use a coarse-to-fine search strategy:
  - coarse levels estimate broad translation, rotation, and possibly weak affine terms
  - fine levels refine around the best coarse candidate
- Use a robust in-loop scalar such as:
  - correlation on guide-map features
  - Vollath F4 on the transformed guide map

Recommended global objective hierarchy:

1. Coarse stage:
   - wavelet or guide-map correlation over a pyramid
2. Fine stage:
   - Vollath F4 on the guide map
3. Sanity gate:
   - cross-view consistency or reprojection-style disagreement penalty

Important design note:

- The current repository global strategy is translation-only.
- The papers support keeping translation-only as the first practical target, but they also justify a future expansion to translation + rotation + weak affine correction.

Recommendation strength:

- Strong for guide-map-based global alignment.
- Medium for expanding the global model beyond translation.

### Stage 5. Pixelwise rectification

Relevant sources:

- both internal FLFM documents
- sparsity-metrics paper
- wavelet fusion paper

Benefit:

- Local refinement can correct residual non-rigid mismatch after good global pre-alignment.
- When driven by guide maps and confidence logic, local correction becomes less likely to chase noise or background.
- An iterative local-average loop can keep improving the local reference instead of freezing it after a single pass.

How to incorporate:

- Run pixelwise matching after global pre-alignment.
- Compute local displacement scores on one of these representations:
  - the guide map
  - a dual representation of cleaned intensity + guide map
  - a blended guide map + gradient/Laplacian representation only when benchmarking shows real gain
- Restrict local search windows around the globally aligned state.
- Use multiscale local search so larger residual shifts are resolved coarsely and only fine residuals are solved at full resolution.
- Build a pixelwise confidence map from:
  - local peak prominence in the score volume
  - first-vs-second-best displacement separation
  - local guide-map amplitude
  - local peak width
- Down-weight or freeze pixelwise updates in regions where emitter evidence is weak.

Recommended iterative local-average loop:

- After the first local rectification pass, do not treat the resulting local average as final.
- Let `A^(0)` be the initial local average after global alignment plus one pass of local rectification.
- For each locally registered view `V_i^(k)`, generate a candidate updated average:
  - unweighted form from the sketch:
    `A_i^(k+1) = (k * A^(k) + V_i^(k)) / (k + 1)`
  - preferred generalization for later work:
    `A_i^(k+1) = (W^(k) * A^(k) + w_i * V_i^(k)) / (W^(k) + w_i)`
- Score every candidate updated average with autofocus-style metrics.
- Promote the best-scoring candidate to become the new local reference `A^(k+1)`.
- Re-run local rectification against the improved local reference, then repeat.

Recommended loop structure:

1. Compute a global average.
2. Run local rectification once and form the first local average `A^(0)`.
3. For each outer iteration:
   - locally re-register all views to the current reference `A^(t)`
   - build one candidate updated average per registered view
   - score all candidate averages
   - accept the best candidate average as the new reference
4. Stop when the score gain, reference change, and displacement-field change all become small.

Why this loop is valuable:

- It makes the local reference adaptive rather than static.
- It lets the most informative registered views steer the next averaging step.
- It can prevent a mediocre first local average from limiting the next local-registration pass.
- It creates a bridge between local registration and autofocus-style quality assessment.

Practical constraints:

- The safest first version is greedy but controlled: evaluate all candidate updates at each step, accept only the best one, and then re-register.
- To avoid collapse toward a single repeatedly favored view, a view should normally be used once per inner round unless it is re-registered in a new outer round.
- The weighted form is better aligned with the confidence-map logic already recommended elsewhere in this plan, because strong locally reliable views do not need to count the same as weak ones.

Recommended local scoring logic:

- Prefer local objectives that reward:
  - confident emitter support
  - sharp compact peaks
  - consistency with neighboring views
- Penalize:
  - peak broadening
  - isolated high-score outliers
  - local displacements unsupported by nearby views

Recommendation strength:

- Strong for confidence-gated pixelwise refinement.
- Medium for blended local representations because that still needs benchmark evidence in this repo.

### Stage 6. In-loop optimization metrics

Relevant sources:

- both internal FLFM documents
- Osibote paper
- sparsity-metrics paper

Benefit:

- Keeps the optimizer aimed at physically plausible rectification rather than aesthetic artifacts.

Recommended in-loop metric family:

1. Primary scalar candidates for the iterative local-average loop:
   - Vollath F4 on the guide map
   - DCTS on the cleaned local average or on a guide-map-derived average
2. Secondary support terms:
   - confident-peak filtered amplitude or response
   - Tenengrad on the guide map
   - Hoyer/Gini sparsity or concentration on the guide map or confident-peak map
3. Penalties:
   - median peak-width penalty
   - cross-view inconsistency penalty
   - optional deformation regularity penalty for pixelwise fields

Suggested composite score:

`Q_loop = z_peak_response + z_tenengrad(G) + z_concentration(G) - z_width - lambda * z_cross_view_inconsistency`

Where:

- `G` is the guide map
- `z_*` are normalized per-candidate terms
- the penalty terms are scaled so they cannot be overwhelmed by a single noisy sharpness gain

Why this is better than a single raw metric:

- Vollath F4 provides a robust scalar target.
- DCTS is worth benchmarking in this specific loop because the light-sheet focusing literature found it to be fast, accurate, and resistant to false optima in focus selection tasks.
- Tenengrad on the guide map measures sharp local structure after noise suppression.
- Hoyer/Gini-like concentration captures the sparse-emitter expectation.
- Width and consistency penalties stop the optimizer from winning by making the wrong structure sharper.

Default recommendation for this repo:

- Use Vollath F4 as the default primary loop metric.
- Benchmark DCTS as a competing or tie-breaking metric for candidate local averages.
- Use Hoyer/Gini as support terms or penalties, not as the sole acceptance rule.

Recommendation strength:

- Strong.

### Stage 7. Final acceptance metrics

Relevant sources:

- `FLFM Rectification_ Metrics and Maps.pdf`
- `Lightweight Non-Deep-Learning Metrics and Guide Maps for Sparse Fluorescent FLFM Rectification.pdf`
- wavelet fusion paper

Benefit:

- Prevents overfitting the rectification loop to the guide map.
- Separates "good optimization surrogate" from "good final biological image."

Recommended final metrics:

1. 1FRC on the final raw rectified image or averaged raw rectified output
2. PSF-fit residuals on isolated peaks
3. localization-uncertainty surrogate on isolated peaks
4. cross-view agreement of detected emitter positions
5. external-camera validation where available

Important rule:

- Final acceptance should be computed on the rectified raw image or on biologically interpretable derived measurements, not only on the guide map.

Recommendation strength:

- Strong.

## What should be deprioritized or avoided

Across the sources, the following should not be treated as primary in-loop objectives for this task:

- raw Shannon entropy on raw images
- raw variance of Laplacian on raw images
- guide-less Brenner or Tenengrad as sole objectives
- unconstrained deconvolution output quality as the optimization target
- any metric that improves mostly by amplifying shot noise, ringing, or background texture

This does not mean these measures are useless. It means they should not drive the main optimization loop in sparse fluorescent FLFM rectification.

This also means that DCTS should be treated differently from raw entropy: if used, it should be computed on the cleaned local average or a guide-map-derived image, not on the raw noisy view stack.

## Recommended changes for the global rectification branch

### What is relevant

- wavelet-feature registration strongly supports coarse-to-fine feature-based global alignment
- both internal FLFM documents support guide-map-based optimization
- Osibote supports Vollath F4 as a robust scalar

### Why it is beneficial

- makes the global objective less sensitive to background haze and photon noise
- improves capture range relative to raw local patch scores
- reduces false local optima before local refinement
- creates a cleaner initialization for pixelwise search

### How it would be incorporated

Recommended global branch design:

1. Build cleaned guide maps for each view.
2. Build a coarse-to-fine representation:
   - image pyramid of guide maps
   - or wavelet-maxima representation
3. Estimate global translation first.
4. If needed later, benchmark translation + rotation or weak affine refinement.
5. Score each candidate using:
   - correlation on guide-map features at coarse scale
   - Vollath F4 on the fine-scale transformed guide map
6. Reject candidates with poor cross-view agreement.

Practical adoption note:

- The safest first integration is not "rewrite the whole global strategy."
- The safest first integration is "keep the same strategy shell, but change the scored representation from raw intensity to guide maps and add a coarse-to-fine search."

## Recommended changes for the pixelwise rectification branch

### What is relevant

- the internal FLFM documents support guide-map-first local refinement
- sparsity-metrics work supports concentration metrics after filtering
- wavelet ideas support multi-scale confidence logic

### Why it is beneficial

- local flow is less likely to lock onto background fluctuations
- guide-map-based confidence can suppress updates in uninformative regions
- multiscale search reduces the chance that a large residual shift is misread as noisy micro-displacements
- iterative local-average updates can improve the reference used by the next local-registration round

### How it would be incorporated

Recommended pixelwise branch design:

1. Start from globally rectified views.
2. Build per-view guide maps and optional confident-peak masks.
3. Search local displacement on the guide map or dual representation.
4. Derive a per-pixel confidence weight.
5. Warp the raw image with the accepted local field, not the guide map.
6. Build one candidate updated local average per registered view.
7. Score the candidate updated local averages with autofocus-style metrics.
8. Promote the best candidate average to become the next local reference.
9. Re-run local registration against that updated reference.
10. Smooth or regularize only where confidence supports it.
11. Penalize cross-view disagreement.

Important practical constraint:

- Pixelwise refinement should not be allowed to "rescue" a failed global alignment.
- It should only explain residual local mismatch after a plausible global solution.
- The iterative local-average loop should be nested inside a stable outer local-registration loop, not used as a substitute for global initialization.

## Role of wavelets in this plan

Wavelets appear in three different roles across the sources. They should not be collapsed into one idea.

### 1. Wavelets as registration features

Use:

- maxima of wavelet coefficients for coarse global search

Best for:

- robust coarse alignment
- improved capture range
- computational efficiency in coarse-to-fine search

### 2. Wavelets as guide-map or denoising tools

Use:

- wavelet shrinkage
- wavelet puncta maps
- wavelet background suppression

Best for:

- preprocessing
- guide-map construction
- stabilizing weak emitters

### 3. Wavelets as fusion tools

Use:

- multi-scale and multi-orientation combination of registered evidence

Best for:

- fusing guide maps or confidence maps across views or scales
- optional post-rectification evidence aggregation

Best practical recommendation:

- Use wavelets first for guide-map construction and coarse global alignment.
- Consider wavelet-domain fusion later for guide/confidence aggregation once the global and pixelwise branches are already stable.

## Role of information-theoretic registration in this plan

The Tsallis-divergence paper is valuable, but it should be treated carefully.

### What is relevant

- smoother objective landscape
- larger capture range
- better robustness to noise in a difficult registration setting

### What is less directly applicable

- the paper targets multimodal registration rather than same-modality sparse fluorescent view alignment

### Recommended role here

- not the first in-loop objective for normal FLFM rectification
- a secondary exploratory candidate for:
  - cross-camera validation alignment
  - aligning different processed representations
  - failure cases where correlation on guide maps becomes unstable

Recommendation strength:

- Exploratory.

## Proposed metric hierarchy for this repository

The current repository already distinguishes baseline displacement metrics from downstream quality metrics. The paper set suggests turning that into a stricter hierarchy:

### Tier 1: preprocessing metrics

Used to tune cleanup, not final rectification:

- background estimation quality
- signal-to-background stabilization
- confident-peak count

### Tier 2: in-loop global rectification metrics

Used to select the global transform:

- guide-map correlation
- Vollath F4 on guide map
- optional coarse wavelet-feature correlation

### Tier 3: in-loop pixelwise rectification metrics

Used to select local displacements:

- local guide-map score volume
- peak-prominence confidence
- cross-view local consistency

### Tier 4: candidate-ranking metrics

Used in sweeps and model comparison:

- composite loop score
- stability across nearby parameter settings
- false-optimum rate
- compute time

### Tier 5: final scientific acceptance metrics

Used to decide if the rectification is actually better:

- 1FRC
- localization residual
- localization uncertainty surrogate
- external-camera agreement
- biological interpretability checks

## How this maps onto the current repo structure

Without changing code in this document, the logical integration points are clear:

- `moment_images/rectification_strategy.py`
  - change the scored representation for global alignment
  - add guide-map-aware local scoring and confidence gating
  - keep final warping on raw views
- `moment_images/parameter_sweep.py`
  - benchmark alternative guide maps and objective stacks
  - compare raw-intensity vs guide-map pipelines explicitly
- `moment_images/sweep_analysis.py`
  - add metrics for objective smoothness, false local maxima, and confidence consistency
  - separate in-loop success from final-image success
- `moment_images/validate_rectification.py`
  - preserve external-camera validation as an independent check

## Suggested experimental program

The papers argue for a staged validation plan rather than a one-shot replacement.

### Phase 1. Representation benchmark

Question:

- Which representation is best for sparse FLFM in this repository?

Candidates:

- raw intensity
- cleaned intensity
- multiscale LoG guide map
- matched-filter guide map
- Hessian/anisotropic guide map
- wavelet puncta map

Primary outputs:

- objective smoothness
- false-maximum count
- sensitivity to noise/background
- compute cost
- final reconstruction quality

### Phase 2. Global objective benchmark

Question:

- Which global objective gives the most reliable initialization?

Candidates:

- raw correlation
- guide-map correlation
- Vollath F4 on guide map
- wavelet-feature correlation

Primary outputs:

- translation accuracy
- capture range
- robustness to synthetic perturbations
- consistency across neighboring views

### Phase 3. Pixelwise refinement benchmark

Question:

- Does guide-map-assisted local refinement improve results beyond intensity-only local refinement?

Candidates:

- intensity-only local score
- guide-map-only local score
- dual intensity + guide-map local score
- guide-map score with confidence gating

Primary outputs:

- field smoothness
- local consistency
- final raw-image 1FRC
- isolated-peak width and localization stability

### Phase 4. Final acceptance benchmark

Question:

- Which in-loop score best predicts final biologically useful output?

Primary outputs:

- correlation between in-loop ranking and 1FRC
- correlation between in-loop ranking and localization residual
- agreement with external validation images

## Recommended default plan of record

If only one practical non-deep-learning plan is adopted first, it should be this:

1. Keep the current global-then-pixelwise workflow structure.
2. Add photometric cleanup and local background suppression before metric computation.
3. Build a multiscale LoG guide map by default.
4. Run global alignment on guide maps using coarse-to-fine search.
5. Use Vollath F4 on the guide map as the main in-loop scalar.
6. Use cross-view inconsistency as a penalty term.
7. Run pixelwise refinement only after strong global initialization.
8. After the first local pass, enter an iterative local-average loop instead of freezing the first local average.
9. At each local iteration, score candidate updated averages with Vollath F4 by default, while benchmarking DCTS and Hoyer/Gini support terms.
10. Gate local updates with guide-map confidence.
11. Rank candidates with a composite score, not a single raw sharpness metric.
12. Accept final results using raw-image 1FRC plus localization-oriented checks.

This plan is the best compromise between:

- scientific defensibility
- compute cost
- compatibility with the current repo
- strength of support in the attached material

## Secondary options worth exploring later

These are promising, but they are not the first thing to trust in production:

- Tsallis-divergence-based fallback objectives
- explicit affine global correction beyond translation
- wavelet-domain fusion of guide maps across views
- adaptive switching between LoG and Hessian guide maps by depth or aberration state
- metric selection conditioned on specimen class or emitter density

## Risks and failure modes

### Risk 1. Guide map becomes the only truth

Problem:

- A candidate can score well on the guide map but harm the raw biological image.

Mitigation:

- keep final acceptance on raw-image metrics and localization checks

### Risk 2. Sparsity prior is violated

Problem:

- Dense structures can make concentration metrics misleading.

Mitigation:

- use this stack only where the sparse-emitter assumption is valid
- fall back to a different regime if density rises too far

### Risk 3. Pixelwise branch overfits noise

Problem:

- Local flow may invent structure in low-evidence regions.

Mitigation:

- confidence gating
- stronger global initialization
- regularity penalties
- cross-view consistency penalties

### Risk 4. Metric ranking does not predict final quality

Problem:

- A metric may look good in-loop but correlate poorly with final scientific usefulness.

Mitigation:

- explicitly benchmark predictive value against 1FRC, localization residuals, and external validation

### Risk 5. Greedy iterative averaging becomes order-dependent or collapses to a few views

Problem:

- A locally iterative average can improve quickly at first but then drift toward whichever view repeatedly gives the largest single-step score gain.

Mitigation:

- evaluate all candidate view insertions at each iteration rather than accepting the first one
- re-register before reopening the candidate set
- use weighted updates based on confidence rather than pure repeated counting
- stop when gains flatten, not when one metric can still be nudged upward by sacrificing view diversity

## Final recommendation

The attached material does not support a "just use a better sharpness metric" strategy.

It supports a more specific conclusion:

**For sparse fluorescent FLFM rectification, the right upgrade is a guide-map-centered workflow with detector-style preprocessing, robust global initialization, confidence-aware pixelwise refinement, an iterative local-average refinement loop, and a strict separation between in-loop optimization metrics and final biological acceptance metrics.**

If the work must start from one change only, it should be:

**Move the optimization target from raw views to cleaned multiscale guide maps, and use Vollath F4 plus consistency-aware penalties to drive candidate selection inside an iterative local-average refinement loop.**

That change is the most aligned with the source material and the current repository design.
