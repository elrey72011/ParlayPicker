# Productionization verification — 2026-09-14

Authoritative task: parlaypicker_productionization_activation_prompt (1).md.
Current-state reference: 2026-09-14-activation-reverification.md (direct owner precedence).

## Preservation

HEAD BEFORE: 772d909735879db73105613565de0ca9c5e36ce9
Preservation commit: 2cdf5fdd (Finalize wager activation, evidence validation, and exposure controls).
Unrelated Medium article files excluded. Synthetic engineering configuration is tracked only in tests/fixtures/activation/config.json, explicitly test_only. No active policy, bankroll, or placement was created. A trailing blank line detected by the staged whitespace check was corrected in the subsequent changes.
Baseline: 2146 passed, 9261 warnings, 97.85 seconds.
Activation baseline: 27 passed, 2119 deselected, 5.59 seconds.

## Result

PRODUCTIONIZATION RESULT: FAIL — strict real capture is not yet proven and the producer provenance gap remains unresolved. This classification is about incomplete operational proof, not about the legitimate $0 recommendation.

The architecture was preserved. Original trained-through/model-available/calibration-available records are not supplied by the current research probability blend in core/streamlit_pipeline.py::compute_blended_probability. The H2H trainer is not evidence for a Spread/Total model. Copying policy validation flags or the current timestamp into these fields would fabricate facts. No such substitution was made. Original producer artifacts/records have been requested from the owner.

No development cutoff or plan was fabricated. A genuine versioned producer and deliberate frozen development choices must precede new holdout predictions. Collection cannot yet be certified as requiring no further producer integration.

## Runtime and real evidence

Local checks: Drive folder ID, service-account JSON, THE_ODDS_API_KEY, active policy file, and exposure ledger are missing. These checks reveal only configured/missing/invalid, never secret values. Streamlit Cloud secrets were not accessible and their absence is NOT inferred.

Remote restore attempted: yes, exit 1 (not_configured).
Prediction snapshots: 0. Score revisions: 0. Strict closing observations: 0. Frozen plans: 0.
LIVE STRICT CAPTURE: FAIL / unproven; new snapshots 0. No live provider run was attempted without credentials.
All NFL, NCAAF, MLB, NBA, NHL, NCAAB states: UNVALIDATED.
For each sport the next milestone is original sport/market producer provenance, a frozen prospective development plan, and admissible future snapshots with deterministic outcomes and strict closes. Required sample/study thresholds must come from that frozen plan, not be invented after seeing results.
Active policy: absent. Activated sports/validation IDs/expiry: none.
Bankroll source: no actual owner configuration. Fresh exposure snapshot: unavailable. Open committed exposure: no recorded events; this does not certify the owner's external account has no bets.
Real candidate count: 0; eligible/funded: 0; total stake: $0. No validated live policy, no strict cohort, and no owner exposure configuration.
Real Standard/Premium legs/pairs/triples: 0. Actual price confirmations: 0. Funded parlays: 0.

## Commands

Literal output and exit codes: output/activation-proof.json and output/activation-proof.md (generated local artifacts).

- prediction_evidence status: 0; restore: 1; status after restore: 0.
- prediction_evidence refresh: 0, no revisions.
- capture_closing_lines --prospective: 0, no_candidates_in_closing_window; no close fabricated.
- validate_sport_deployment --all-sports: 0; all UNVALIDATED with missing_frozen_development_plan, no_strict_admissible_evidence, no_tier_passed_frozen_holdout_criteria.
- build_wager_policy: 0; verify_wager_policy: 0. Candidate all-zero policy was not activated.
- exposure status: 0; snapshot: 1 (BANKROLL_AND_LIMITS_NOT_CONFIGURED); verify-snapshot: 1 (missing snapshot).
- validate_live_activation hermetic/production/replay/parlay: each 0.
- Synthetic normal $2.50/outage $1.00 proof remains engineering-only; no real authority is established by it.

## Changes after preservation

Existing evidence health now reports runtime configuration status, strict close count and frozen-plan count. The owner currency input starts blank instead of assuming USD. Actual bankroll and all caps still require owner input. Regression tests cover credential redaction and immutable operational table counts. No authority gates, validation thresholds, or historical decisions were changed.

Safety: Moneyline context-only, legacy APPROVED cannot fund, research locks cannot fund, expired policy/stale exposure fail closed, outage cannot promote, and no automated placement remain covered by the baseline/acceptance tests. There was no sportsbook interaction.

## Exact owner steps

1. Streamlit app → Manage app → Settings → Secrets: verify root-level PARLAYPICKER_DRIVE_FOLDER_ID, PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT (complete service-account JSON), and THE_ODDS_API_KEY. Do not post secret values in chat. Share the configured Drive folder with the service account using the existing storage instructions.
2. In authenticated Preview & Publish, open Show wager readiness and exposure and inspect runtime_configuration. Run the existing evidence restore/status commands in the configured runtime. Restore merges immutable records; do not replace the database with an unrelated archive.
3. Supply the location of original Spread/Total producer training/calibration records. Existing report text is not a replacement for those records.
4. Configure actual bankroll, unit value, currency, and total/daily/weekly/game/team fraction caps in the owner panel. Record any actual open commitments; never turn research locks into placements.
5. After the real producer chain is verified, freeze development choices BEFORE collecting the holdout; run live analyses, pregame --prospective closing capture, and refresh outcomes. Re-run validation/build/verify. Explicitly review/activate only legitimately validated policy.
6. Actual ticket confirmation requires a real saved qualified ticket and the owner's actual combined price. No invented confirmation was submitted.

Final tests: 2148 passed, 9261 warnings, 97.72 seconds. Final git diff --check: passed. No push performed.
