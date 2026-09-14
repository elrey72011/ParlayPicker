# Activation repository verification

HEAD before: 772d909735879db73105613565de0ca9c5e36ce9.
Baseline: 2,137 passed, 9,261 warnings, 100.60 seconds.

The supplied final activation prompt controls scope. The live-wager contract and sport deployment assessment were read as specifications and claims requiring code verification; their reported implementation and test results were not assumed.

| Dependency | Verified active code | Finding before these edits |
|---|---|---|
| Terminal authority | streamlit_app.py calls finalize_live_wagers before capture | Present; preserve gates |
| Evidence | prediction_evidence.capture_run, activation_validation.enrich | Missing canonical fields; whole-run count incorrectly represents per-game completeness; outcome provenance discarded |
| Close collection | capture_closing_lines.main | Automatic path only research totals; strict path requires external JSON |
| Validation | activation_validation.validate | Chronological receipt present; uncertainty coverage, prior comparison, ablation and strategy comparison missing |
| Policy | activation_policy.verify_sport | Hash/expiry present; advanced validation support not required |
| Maturity | candidate_maturity.assign | Active; runtime evidence production still incomplete |
| Exposure | exposure_ledger.snapshot | Active, owner configuration required; placement UI only JSON |
| Parlays | production_parlays and parlay_confirmation | Qualification/confirmation core present; owner form missing |
| Remote | evidence_remote | Interface present; local credentials absent, no claim of cloud absence |

Implement dependencies in this order: evidence identity/provenance and completeness; exact close/outcome collection; validation studies; runtime readiness; owner actions and acceptance proof. Never replace missing evidence with guessed metadata.

## Verified changes in this pass

- Canonical private candidate projection preserves the specified fields, original missingness, process ID, exact-line candidate identity and payload digest. Per-game completeness checks preserve the generator count. Repeated prospective snapshots do not multiply the validation population.
- Captured quote records now preserve provider namespace and event ID. The explicit `capture_closing_lines.py --prospective` job matches immutable candidates to live provider records for spreads and totals. Identity/book/direction ambiguity, absent namespaces, stale or post-start observations remain unavailable.
- Evidence refresh calls the deterministic result matcher for unresolved started games across dates. Append-only score revisions retain result source/event ID, and materialization checks score hashes.
- Validation v2 adds probability-band coverage diagnostics, prior/current/hierarchical comparisons, optional paired ML probability scoring, premium interval reporting, separate CLV metrics, and three frozen sizing comparisons. Strategies use saved pregame selections, not both opposing sides. Missing study inputs fail promotion; old incomplete active validation objects are rejected.
- `prospective_uncertainty.prepare_live` is called immediately before terminal authority. It requires an immutable frozen development plan and matching versions. Only earlier completed same-sport/market records contribute. Current-season conflict widens uncertainty. These forecasts do not set model/calibration validation flags or activate a policy.
- Owner forms support actual placement records with changed-line/value warnings, settlement events, and combined parlay price confirmation. The parlay CLI and owner form share `owner_parlay_confirmation.confirm_ticket`.
- A later RECOMMENDED ledger event cannot erase an existing COMMITTED exposure.
- `bootstrap_evidence.py` runs explicit closing/grading/validation/optional backup operations. It never activates a policy. Rendering does not request provider quotes or results.
- Owner readiness includes validation/version/expiry/closing counts and evidence-store health.

## Validation

HEAD after remains 772d909735879db73105613565de0ca9c5e36ce9; changes are uncommitted.

- Baseline: 2,137 passed, 9,261 warnings, 100.60 seconds.
- Final: **2,146 passed**, 9,261 warnings, **99.70 seconds** (`test-results/reverify-final.txt`).
- Activation marker: **27 passed**, 2,119 deselected, **5.29 seconds** (`test-results/reverify-acceptance.txt`).
- `git diff --check`: passed.
- An intermediate full run found a timeout test consuming persistent local Gemini budget. The test now mocks budget/cache access so it tests the transport failure without using owner request accounting. Production budget behavior was not weakened.
- `python scripts/bootstrap_evidence.py --capture-closes --grade`: exit 0; no candidates in closing window, zero revisions; all six sports remain UNVALIDATED.
- `python scripts/run_activation_proof.py --full-test-result test-results/reverify-final.txt --acceptance-result test-results/reverify-acceptance.txt`: completed; literal command outputs and hashes are in `output/activation-proof.json` and `.md`.

## Remaining evidence and integration boundaries

Do not treat this report as evidence that live activation has been earned or every upstream producer is complete. The repository's existing research forecast producers still do not supply all original sport/market model and calibration provenance required by this architecture. Missing training/availability facts remain missing; a code bundle hash is not a training record. The new uncertainty path requires genuine prior admissible records and frozen settings; it cannot turn the current empty store into evidence. Original model/calibration validation flags are still mandatory and are not automatically invented by the adapter.

All six sports have strict_n=0 and effective_n=0 locally. There is no frozen development plan, no strict prospective prediction cohort, and no verified closing cohort here. Empirical uncertainty calibration, prior decay/weight superiority, ML feature benefit, drift thresholds and Premium accuracy have therefore **not** been established. The implemented studies must run on actual future data; reports with missing fields cannot promote. No historical timestamp/close/75% claim was manufactured.

Remote restore lacks `PARLAYPICKER_DRIVE_FOLDER_ID` and `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT` locally. This does not prove remote cloud data is absent. No active sport policy or bankroll/limits were created. Snapshot generation correctly fails with BANKROLL_AND_LIMITS_NOT_CONFIGURED. Real dry-run funding is zero; synthetic engineering funding remains $2.50 normal/$1 outage. No automated execution exists in the added workflows.

Advanced source-producer provenance, paired ablation forecasts and original uncertainty/evidence inputs must be supplied by their actual producers; they cannot be inferred from these reports or old CSVs. This delivery does not certify the full live producer-to-activation chain against real inputs.

## Operator flow

1. Run live analysis to append candidate evidence. Supply genuine producer provenance and a frozen development plan; inspect exclusions, never fill missing dates with current time.
2. During the pregame close window run `python scripts/capture_closing_lines.py --prospective --database data/prediction_evidence/evidence.sqlite3`.
3. Refresh all unresolved evidence via `python scripts/prediction_evidence.py refresh --database data/prediction_evidence/evidence.sqlite3`.
4. For one explicit forward job: `python scripts/bootstrap_evidence.py --capture-closes --grade --backup`. Configure the existing Drive credentials first for backup/read-back. No scheduler was installed by this task.
5. Run the validation/build/verify/explicit activation commands documented in the final activation report. v2 now requires coverage and prior-study support; an older incomplete result cannot be activated.
6. Authenticated Preview & Publish → Show wager readiness and exposure → configure actual bankroll/limits. Use the new Mark a recommended wager as placed form only after actual external placement; settlement appends an event.
7. For an eligible saved ticket, open Confirm actual combined parlay price. Enter the actual same-book price and confirm unchanged legs. Failed checks return no recommendation; this never places a bet.

## Approval review

Automatic review rejected an edit that would infer model_validated/calibration_validated flags from policy context, citing the risk of treating unverified facts as validation. It was not applied. The safe adapter only reads existing quote/evidence field aliases and retains mandatory producer validation flags. No approval is pending for that rejected edit.

PROMPT PRECEDENCE CHECK: authoritative file and both references read; references treated as specifications/claims, not implementation proof; older prompts superseded. Repository discrepancies are listed above. The prior report's stated limitations were independently verified before edits.
