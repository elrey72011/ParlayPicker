# Capture prediction timestamp — 2026-09-15

HEAD BEFORE: `609b8f59c298de51618d8769df69b9486591a871` (main).
HEAD AFTER: commit containing this report on `codex/capture-prediction-timestamp`; exact hash in the PR handoff.

## Root cause and reproduction

`capture_run` populated prediction_generated_at only when the column was absent. Existing authoritative columns containing None, empty string, NaN, pd.NA or pd.NaT were saved without a prediction timestamp.

Before the production edit, the new regression group produced 6 failures and 3 passes (99 deselected; 2.95s). All five missing-value variants and a mixed-row case failed; missing column, explicit producer time and invalid nonblank preservation already passed.

## Implementation

The audit/final frame loop now uses the existing missing-value predicate to fill only missing cells with the same actual UTC `generated` value that supplies this capture's export_run_id. Object dtype permits filling null datetime columns without coercing valid producer values. Nonblank strings are neither parsed nor repaired. No timestamp is backdated or inferred from historical game data.

Production change: only `app_core/prediction_evidence.py`.

## Acceptance checks

- Missing column: PASS.
- None, empty string, NaN, pd.NA and pd.NaT: PASS.
- Explicit aware UTC producer timestamp preserved exactly: PASS.
- Mixed valid/missing rows: PASS; only the missing row changes.
- Malformed nonblank timestamp preserved: PASS; timing readiness remains blocked.
- capture_run → load_snapshots persistence: PASS for candidates and final card.
- Normal future fixture: prediction_generated_at <= export_run_id < game_start_utc: PASS.
- Valid fixture readiness: prediction_or_start_time_unverified absent; export_timing_unverified absent; model_provenance_missing still present.
- No model_version, model_trained_through or model_available_at is populated in the authoritative fixture.
- Append-only protection: PASS; recapture of the same snapshot is rejected, and stored candidate/decision payloads and hash remain unchanged. Existing database immutability tests also pass.

This applies only to new captures. Existing immutable snapshots are not repaired or rewritten. Model/calibration provenance, wager policy, maturity, thresholds, Kelly, quote binding, closing capture, exposure and Gemini logic remain unchanged. No sportsbook execution or funding authority was added.

## Validation

- Focused: 123 passed, 0 failed, 19 warnings; 13.09s.
- Full pytest -q: 2318 passed, 0 failed, 9828 warnings; 117.83s.
- Exact current CI production-safety command: 540 passed, 0 failed, 1430 warnings; 35.91s.
- Exact current CI production compilation: PASS.
- git diff --check: PASS.

Tests ran locally with the configured Python, test dependencies and Node runtime. Existing warnings were not changed. Hosted CI runs after push.

## Files changed

- app_core/prediction_evidence.py
- tests/test_prediction_evidence.py
- docs/audits/2026-09-15-capture-prediction-timestamp.md

Unrelated changes in this patch: NONE. The pre-existing untracked Medium article files remain untouched and excluded.
