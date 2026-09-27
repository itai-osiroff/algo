# Phase 3 Promotion Integrity Repair

## Global Constraints

- Bind a typed Phase-0 normalization-control receipt into Phase 0-2 before its terminal certificate is sealed.  No Phase-3 code may add, replace, or modify source-root evidence.
- Continue only from a V2 Phase 0-2 root that contains the new checksum-bound control receipt.  Older roots and all historical in-source Phase-3 runs fail closed; do not auto-migrate or rewrite them.
- A V1 Phase-3 adaptive round must be rejected before newer prerequisite checks with exactly `legacy_phase3_adaptive_rounds_unsupported`.
- Preserve the V2 Phase-3 continuation CLI and automatic Phase-4 handoff.  Retire the legacy in-source Phase-3 route with a clear migration error.
- Use a shared, production-backed execution-evidence builder for both the worker and roadmap proof.  Its Phase-3 certification records include verified normalization controls followed by Phase-3 stage results.
- Phase 4 may admit only a semantically valid, canonical terminal Phase-3 chain.  Certificate linkage or a permitted stop-reason string alone is insufficient.

## Task 1: Seal Phase-0 normalization controls into Phase 0-2 and consume them in V2 continuation

Implement the typed normalization-control receipt during Phase 0-2 generation.  It must contain and validate the required Phase-0 anchor, direct-challenger, and legacy-fixed-blend records.  Write it before terminal sealing, bind its canonical checksum as a required Phase 0-2 output, and extend Phase-3 admission/baseline schema to load and validate it.  Refuse roots without the receipt without source writes.

Introduce one shared continuation execution-evidence builder used by the V2 continuation worker and later roadmap tests.  It must copy verified baseline controls plus Phase-3 stage results into Phase-3 execution evidence; next-round model derivation must therefore receive complete certification records.  Remove reliance on a post-terminal source-root control snapshot.

Write regression tests first and capture RED/GREEN evidence for: terminal checksum binding, safe failure for missing/corrupt/substituted receipts, and a genuine V2 execution whose feasible normalized candidate changes the next planning-model hash.

## Task 2: Fail closed for the legacy Phase-3 route and historical rounds

Retire the in-source Phase-3 execution path behind `--run-adaptive-until-phase3`.  It must produce a migration error that directs callers to execute Phase 0-2 and then use `--continue-adaptive-phase3-from-phase2`; it must not create Phase-3 source artifacts.  Delete the post-terminal legacy control snapshot/write logic.

Classify canonical schema-v1 adaptive round decisions before checking newly introduced control prerequisites and return exactly `legacy_phase3_adaptive_rounds_unsupported`.  Other historical in-source Phase-3 artifacts must fail closed under a migration error and must never be automatically bridged or rewritten.

Write tests first and capture RED/GREEN evidence for the exact V1 blocker even with a historical control-less baseline and for the retired invocation's no-write behavior.

## Task 3: Recompute and enforce the terminal Phase-3 chain before Phase 4

Add one read-only terminal-chain validator that loads canonical committed continuation evidence for every completed round, replays planning lineage, reconstructs the expected exhaustion certificate, and requires exact canonical receipt equality.  Valid terminal causes are real empty-candidate exhaustion, exhausted round or budget limits, or sustained qualifying convergence; forged Pareto, coverage, hypervolume, certificate, or stop-reason data must be rejected.

Use the validator whenever Phase-3 orientation reaches completion and in Phase-4 admission.  A valid chain must retain the existing Phase-4 handoff and must not mutate Phase 0-2 source bytes.

Write tests first and capture RED/GREEN evidence for premature exhaustion, fake convergence, altered Pareto/coverage, mismatched certificates, and a valid chain reaching Phase 4.

## Task 4: Make the roadmap proof production-faithful and document the migration

Change the roadmap fixture so it produces all Phase 0-2 artifacts, including normalization controls and their checksum, before its only terminal certification.  Delete the fixture's delete-and-reseal sequence.  Route the promotion probe through the shared Phase-3 execution-evidence builder from Task 1 and assert that a feasible normalized round-one candidate changes the round-two planning model.

Keep `promoted_through: "phase3"` dependent on this production-backed proof.  Update the parameter-sweep guide for the retired legacy route and unchanged V2 continuation command.

Write tests first and capture RED/GREEN evidence that the fixture never reseals source evidence and that the probe follows the production builder.

## Final Verification

Run the focused Phase 0-2, Phase-3 continuation, lifecycle, Phase-4 continuation, roadmap-gate, supervisor, and parameter-sweep test suites.  Run `python -m compileall moment_images` and inspect the final diff for unplanned source-root writes.
