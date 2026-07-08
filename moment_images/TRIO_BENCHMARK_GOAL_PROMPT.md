# Trio Benchmark Registration Calibration Goal Prompt

Use this prompt after assembling a local five-scene trio benchmark manifest.

```text
/goal
Objective: Improve the moment_images trio benchmark registration/evaluation plumbing until the generated overlays are visually credible across the five-scene validation dataset.

Repository and branch:
- Work in C:\Users\user\Documents\git_clones\algo.
- Use the branch that contains moment_images/trio_benchmark.py and moment_images/trio_benchmark_batch.py.
- Do not mutate completed sweep outputs, run_state.json, sweep summaries, candidate workspaces, or validation workspaces.

Dataset:
- Use the local manifest at <ABSOLUTE_PATH_TO_TRIO_MANIFEST_JSON>.
- Each manifest scene has sweep_dir, dataset_id, and either validation_workspace or validation_image plus thorcam_image.
- Treat validation/ThorCam only as post-processing references. Do not feed them into LF reconstruction or sweep execution.

Primary task:
1. Run python -m moment_images.trio_benchmark_batch --manifest <ABSOLUTE_PATH_TO_TRIO_MANIFEST_JSON> --output-dir <ABSOLUTE_PATH_TO_OUTPUT_DIR>.
2. Inspect every per-scene validation/LF overlay and validation/ThorCam overlay in the generated figures folders.
3. Identify whether poor scores are caused by registration failure, metric behavior, missing/incorrect input artifacts, or true reconstruction quality.
4. Improve only the trio benchmark post-processing/diagnostic pipeline needed to make registration reliable and inspectable. Preserve the primary transform family as similarity-only: isotropic scale, rotation, translation, no shear.
5. Keep affine or nonrigid alignment diagnostic-only if used; it must not affect validation_hybrid_v1 unless explicitly approved later.
6. Re-run the batch after each meaningful change and compare overlays and summaries.

Acceptance criteria:
- For every scene, the best LF overlay and ThorCam overlay share plausible bead locations over the common FOV by visual inspection.
- No accepted primary result has an obvious large residual translation failure hidden by the score.
- If a scene still cannot be registered reliably, the batch summary and per-scene summary clearly warn that the registration is suspect instead of presenting the score as trustworthy.
- The aggregate trio_batch_summary.json records num_scenes, num_successful, scene status, best_lf_row_id, ThorCam score, LF score, LF-minus-ThorCam delta, warnings, and per-scene report paths.
- The implementation remains deterministic for the same manifest and inputs.
- The LF algorithm remains blind to validation/ThorCam images; only post-processing candidate selection and diagnostics may use those references.

Recommended investigation order:
1. First verify the manifest paths, dataset_id values, and reference source labels in each trio_benchmark_summary.json.
2. Inspect registration metadata for low registration_confidence, registration_suspect=true, registration_suspect_reasons, low matched_bead_count, high median_bead_error_px, large residual_translation_norm values, suspicious scale/rotation, and repeated warnings.
3. Prefer improving similarity registration initialization and diagnostics before changing metric weights.
4. If changing registration, add synthetic regression tests covering different FOVs, scale, rotation, sparse beads, and false local optima.
5. Only tune validation_hybrid_v1 after overlays are visually credible.

Verification:
- Run python -m pytest moment_images/test_trio_benchmark.py moment_images/test_trio_benchmark_batch.py -q.
- Run python -m pytest moment_images/test_validate_rectification.py -q.
- Run python -m py_compile moment_images\trio_benchmark.py moment_images\trio_benchmark_batch.py.
- Run python -m moment_images.trio_benchmark --help.
- Run python -m moment_images.trio_benchmark_batch --help.
- Run git diff --check.

Deliverables:
- Updated code and tests for the trio benchmark tooling.
- A short summary of which scenes pass visually, which remain suspect, and why.
- Paths to the final batch output folder and the most relevant overlay figures.
```
