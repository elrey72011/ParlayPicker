# True-parlay implementation and readiness record

**Recorded:** 2026-09-23, 19:20 UTC
**Baseline checkout:** `codex/validated-wager-readiness`, HEAD `334e1235fb881df5a46f6e3310f1f84480577cad` before this implementation.
**Default-branch comparison:** local HEAD is four commits ahead of local `main` (`a979b9824ff97f57c5bc66361b61bf01d8219532`). It already contains the MLB receipt/reconciliation work described by PR #2330. Remote PR state could not be verified here. No second receipt collector was created.

## Readiness

No validated real-money wager readiness is established. Standard, Same Game, and Cross Game parlays each remain **UNVALIDATED**, with no actual ticket quote or independently validated joint/product policy in the public pipeline. Every public product record has `production_eligible=false` and a $0 recommended stake. The engine and persistence layer are not connected to a production sportsbook quote feed or authenticated owner ticket route. Controlled trials default off; this work did not grant owner consent, configure a real reservation store, or activate trials. No bet was submitted. No merge or deployment was performed.

The positive engine fixtures use synthetic IDs, prices, models, owner authorization, and exposure. They prove mechanical gates only. They are not live validation evidence.

## Prioritized findings and evidence

| Priority | Finding / source | Evidence status and disposition |
| --- | --- | --- |
| P0 | Strict approval lacks authentic producer/model/calibration/policy provenance in the inspected prediction snapshot; `core/wager_decisions.py:candidate_decision`, `core/sport_policy.py` | The existing readiness report describes 66 MLB/WNBA candidate rows missing critical facts. Live source counts were not independently retrievable. No placeholder authority was inserted. New leg and joint gates in `core/true_parlay_engine.py` fail closed. |
| P0 | Independent training capacity is smaller than market-row count; `docs/audits/2026-09-23-mlb-receipt-readiness.md`, receipt preparation/training split code | A local 55-game/220-row settled export was inspected; spread had 110 decided rows, totals 108 decided plus two pushes. Internal saved-receipt hash, identity, and timestamp checks found no mismatches. At most 17 independent training units per family remain while preserving the later validation and holdout floors, and zero whole-Eastern-slate cutoff pairs are count-feasible. External source authenticity and remote backup are unverified; `training_authorized=false`. |
| P0 | Owner authority, exposure, and executable ticket price are required; `app_core/trial_authority.py`, `app_core/controlled_trial.py`, `core/true_parlay_engine.py:evaluate_ticket`, `core/exposure_ledger.py:snapshot` | Implemented default-off revocable trial consent, explicit durable reservation ledger, existing straight/parlay exposure accounting, exact quote/hash/timing validation, distinct product policy and joint evidence, and $0 on missing facts. No real owner grant or current configured production ledger was inspected. |
| P0 | Product-specific validation and correlation are unavailable; `core/true_parlay_engine.py:_joint_blockers`, `app_core/true_parlay_public.py:build_product_board` | Standard requires verified dependence, SGP requires its own validated joint model and actual sportsbook SGP quote, and Cross keeps an SGP as a validated component block. Public generation reports bounded/truncated research candidates only. All three products remain UNVALIDATED. |
| P1 | Saved board reasons and browser-time expiry were conflated; `app_core/board_diagnostics.py`, `app_core/public_board.py`, `publishing/board.html` | Versioned diagnostics now freeze one trace per selected overall row. Browser checks keep current blockers separately, display per-cohort analysis/quote ages and publication age, and show primary counts that reconcile to selected games plus non-additive overlaps. The full market candidate count remains null because the selected-row serializer cannot recover upstream candidate volume. |
| P1 | Trial review stopped at the final two-selection cap and trial price math omitted explicit push semantics; `app_core/controlled_trial.py`, `app_core/controlled_trial_pipeline.py`, `core/price_value.py` | Review budget is now six, final count remains two, unreviewed/held/rejected candidates have explicit states, and strict/trial value calculations use shared push-aware math. Exact changed-line probabilities still require upstream provenance and recomputation before live use. |
| P1 | Package/source/host identity was incomplete; `scripts/publish_board.py`, `publishing/site.js` | Manifest now separates source Git SHA and dirty flag, source fingerprint, and board-content hash; new package fields are validated against frozen rows. This is locally verified only. The hosted `index.html`, `board-data.json`, and `version.json` could not be fetched through this environment. |
| P1 | Selection results and actual accepted returns needed separate immutable evidence; `app_core/parlay_persistence.py` | Idempotent append-only SQLite tables freeze identity, ticket, legs, quote, gates, result revisions, and validation evidence. Accepted wager P/L requires an actual receipt, accepted price/stake, and placement time; selection-only grading remains separate. No real accepted wager was recorded. |

## Historical board and evidence gap

The exact 17-game screenshot payload, its `board-data.json`/`version.json` pair, and the corresponding producer candidate stream were unavailable. Consequently, no per-game or per-candidate rejection waterfall for that historical board can be asserted. The new diagnostics run only on future packages built by this checkout; they explicitly distinguish **selected overall games** from **all market candidates**. The public site was unreachable from this environment, so no hosted asset hash, source SHA, quote age, or deployment reconciliation can be claimed.

The pre-existing Streamlit history described zero validation plans and zero closing observations. The collector workflow is present locally but its authorized live run, Drive backup verification, and new receipt totals were not verified. A historical log indicating bundles and snapshots is not proof of a current live run.

## Implementation and schema

The ticket engine provides exact leg admission, hashes, ticket quote binding, joint/dependence gates, product-specific validation states, push/void-aware value and settlement, minimum acceptable price, deterministic portfolio sizing, full underlying exposure accounting, and machine-readable blockers. Gemini review can confirm, reduce, or hold, but cannot override deterministic failures or increase stake. Candidate generation is bounded and reports truncation. The public serializer recomputes the research records and funnel from the saved input before publishing.

The SQLite migration is additive: `parlay_identity`, `parlay_ticket`, `parlay_leg`, `parlay_quote`, `parlay_gate`, `parlay_result`, and `parlay_validation_evidence`. Foreign keys prevent orphan rows; triggers reject update/delete of saved evidence. Repricing appends a decision under the stable parlay identity; regrading is idempotent for identical inputs and appends revisions for corrections.

Public packages gain optional `board_diagnostics`, `parlay_product_policy`, `parlay_products`, and `parlay_product_funnel`. Legacy packages remain readable. Product records include exact legs and hashes, sportsbook quote identity and timing, joint/model/calibration/validation fields, dependence, price floor, EV/edge, exposure snapshot identity, eligibility, stake, and blockers. Missing facts are null. The Parlays view has All/Standard/Same Game/Cross filters, ticket cards, stage counts, and browser-time fail-closed status checks.

## Changed files

| Area | Files | Why |
| --- | --- | --- |
| Engine and risk | `core/true_parlay_engine.py` (new), `core/price_value.py` (new), `core/exposure_ledger.py`, `core/wager_decisions.py` | Product gates, pricing, settlement, and shared exposure/value. |
| Persistence and authority | `app_core/parlay_persistence.py` (new), `app_core/trial_authority.py` (new), `scripts/trial_authority.py` (new), `app_core/controlled_trial.py`, `app_core/controlled_trial_pipeline.py`, `streamlit_app.py` | Immutable ticket evidence and explicit controlled-trial consent/reservations at the runtime call site. |
| Public package and UI | `app_core/true_parlay_public.py` (new), `app_core/board_diagnostics.py` (new), `app_core/public_board.py`, `scripts/publish_board.py`, `publishing/board.html`, `publishing/site.js` | Research product pipeline, diagnostics, source manifest, and browser display. |
| Lightweight matching | `app_core/mlb_team_aliases.py` (new), `app_core/public_prop_timing.py`, `app_core/theover_ingest.py`, `app_core/public_history.py` | Share MLB aliases without loading the prediction stack during publication and preserve city/full-name grading. |
| Tests | `tests/test_true_parlay_engine_unittest.py` (new), `tests/test_true_parlay_public_unittest.py` (new), `tests/test_parlay_persistence.py` (new), `tests/test_board_diagnostics.py` (new), `tests/test_price_value.py` (new), `tests/public_parlay_products.cjs` (new), `tests/test_controlled_trial.py`, `tests/test_controlled_trial_integration.py`, `tests/test_public_board.py`, `tests/test_research_parlays.py`, `tests/public_site_browser.cjs` | Synthetic gates, evidence immutability, diagnostics, authority, publication parity, and UI regressions. |

## Verification

Executed from this checkout:

```text
NODE_BINARY=/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node /private/tmp/parlaypicker-test-venv/bin/python -m pytest -q tests/test_parlay_persistence.py tests/test_board_diagnostics.py tests/test_controlled_trial.py tests/test_price_value.py tests/test_controlled_trial_integration.py
# 48 passed, 1 pandas FutureWarning

PATH=/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH NODE_BINARY=/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node /private/tmp/parlaypicker-test-venv/bin/python -m pytest -q tests/test_public_board.py tests/test_public_parlays.py tests/test_research_parlays.py tests/test_quote_freshness.py tests/test_public_prop_timing.py tests/test_public_refresh.py tests/test_parlay_safety.py tests/test_parlay_correlation.py tests/test_wager_integrity_audit.py --tb=short
# 143 passed

/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_true_parlay_engine_unittest.py tests/test_true_parlay_public_unittest.py -q
# 27 passed

/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node tests/public_parlay_products.cjs
# passed
/Users/robertoavelarde/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node tests/public_refresh.cjs
# passed
git diff --check
# passed
```

An installed-Chrome `tests/public_site_browser.cjs` run and a diagnostic-package DOM check also passed. A broader 247-case subset ended **216 passed, 31 failed** because this test environment lacks `streamlit` and `xgboost`; it is not a clean full-suite result. Those 31 cases must be rerun in the project's complete dependency environment.

## Remaining owner work

1. Retrieve the exact historical board or explicitly close its 17-game waterfall as unavailable. On a new authorized run, save source/runtime/config/build IDs, exact candidate traces, public assets, and hashes.
2. Verify the collector and Drive backup in the authorized environment; accumulate enough distinct settled games and outcome-available whole Eastern slates for a declared chronological train/validation/holdout split. Freeze cutoffs before looking at performance.
3. Produce real model availability, calibration, joint-dependence, product validation, prospective ticket quotes, settlement rules, closing observations, and predeclared price-aware holdout results for each product independently. Connect a verified ticket-quote feed and authenticated owner route only through a separately authorized rollout.
4. Configure and inspect current owner consent, bankroll, exposure and reservation ledgers through the existing authorization process; verify upstream probability recomputation for every changed line. Keep placement manual.
5. Rerun the full test suite with project dependencies, review this branch, and reconcile local SHA/build/hash with the actual hosted assets before any separately authorized rollout.

No validation gate was weakened or fabricated. Zero actionable parlays and zero trials remain valid outcomes.
