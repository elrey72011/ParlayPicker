# Five activation follow-up fixes

HEAD BEFORE: 890ad5c27693675226306c43d7ce435b1120412a (current origin/main at task start).
Branch: codex/five-activation-fixes. Final commit is reported in the PR and delivery response.

All five review findings remained present before edits. Baseline focused tests: 127 passed, 0 failed, 89 warnings, 4.52 seconds. Initial and final whitespace checks passed.

| Finding | Before | After | Regression coverage |
|---|---|---|---|
| Odds credential alias | Independent job only exposed THE_ODDS_API_KEY | Both aliases reference the same existing secrets.ODDS_API_KEY | test_activation_workflow_aliases_window_and_independence |
| Eastern window | Independent job performed off-window operations | Existing window script gates install, restore, collect/grade/backup and health; research job and cron unchanged; no job dependency added | Same workflow regression plus existing scheduler/window tests |
| Expanded candidate authority | Prepared/evaluated analysis_df; saved a different audit | Prepare/evaluate complete candidate_audit_df; persist the prepared pool; identify selected rows by candidate_id without replacing another candidate's identity | test_expanded_prepared_pool_is_evaluated_and_persisted; test_eligible_alternative_and_canonical_order |
| Validated family | Missing producer field blocked maturity | Clear candidate assertion; populate from active study only after sport, policy-validation hash, state, expiry, versions, prior holdout/slate and original provenance checks | Validated Spread/Total, missing/tampered/expired study, sport/family/version mismatch and missing producer-validation tests |
| Exposure | Daily/weekly remainder passed as absolute total ceiling | Pass absolute daily/weekly/total caps to allocator; each dimension subtracts committed exposure once | Daily/weekly finalizer regressions; total/game/team/sport, exhausted/over-cap and sequential allocation tests |

The source of validated_evidence_family is validation_results[sport].market_family. The production configuration loader already verifies the active policy artifact; the finalizer binds that study to the exact candidate before maturity. Model/calibration validation flags, training cutoff and availability are never inferred. A matching market name alone cannot authorize a family.

The selected candidate's original prepared fields are retained on the final card. Exact candidate IDs join maturity diagnostics to saved evidence. The legacy sync that overwrote the original selected audit row is not used after terminal authority. No candidate is dropped merely for losing initial research selection; no arbitrary frame concatenation was introduced.

## Validation

- Focused: 158 passed, 0 failed, 108 warnings, 6.34 seconds.
- Full: 2184 passed, 0 failed, 9279 warnings, 104.93 seconds.
- Existing CI production-safety command: 448 passed, 0 failed, 1430 warnings, 35.89 seconds.
- Existing CI production compile command: PASS.
- Workflow YAML: PASS. Both aliases, unchanged schedule, original research window, independent job and activation window verified.
- git diff --check: PASS.

Logs: test-results/five-baseline.txt, five-focused.txt, five-full.txt, five-safety.txt.

## Acceptance and safety

FIX-01 through FIX-30: PASS, through the focused regressions, full suite, CI safety suite and diff check above.

Moneyline remains context-only; UNVALIDATED stays $0; model/calibration flags are not inferred; factual Gemini veto remains blocking; outages do not promote maturity; research locks and legacy APPROVED cannot fund; production parlay stake still requires verified actual ticket price; no automated execution was added. No policy was activated or bankroll configured by this patch.

## Files changed

- .github/workflows/research-scheduler.yml
- streamlit_app.py
- core/live_wager_contract.py
- core/wager_decisions.py
- tests/activation_fixture.py
- tests/test_live_wager_contract.py
- tests/test_pipeline_identity_before_portfolio.py
- tests/test_research_scheduler.py
- tests/test_wager_integrity_audit.py
- This report.

UNRELATED CHANGES: NONE. Existing untracked Medium article files were left untouched and excluded. No model retraining, thresholds, formulas, historical records, locks, or public styling changed.
