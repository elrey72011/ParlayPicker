# Two PR #2278 P1 evidence-capture fixes

HEAD BEFORE: `2cbd6c48af1d32f524919e03e74bc412ce6c7a0c` (current main).
HEAD AFTER: commit containing this report on `codex/authoritative-evidence-binding`.

## Before-edit reproduction

Clean tracked working tree and `git diff --check` confirmed before edits.
`test-results/p1-before.txt`: three failing assertions through real authoritative
capture reproduce the two findings (binding, semantics, UTC start).

| P1 | Reproduced behavior | Root cause | Fix |
|---|---|---|---|
| A: quote binding | Exact raw provider quote exists, saved binding is null | Authoritative capture skips binder | Bind before preparation; preserve consistent complete binding; same helper at capture |
| B: derived fields | Semantics and Eastern-to-UTC fallback are overwritten with null | Blanket private-field restoration after derivation | Source facts first, capture metadata next, canonical derivation last; no blanket restoration |

## P1-A: authoritative quote binding

`ensure_authoritative_quote_binding` is deterministic, side-effect-free and uses
existing `bind_quote`/`matching_quotes`. It preserves a complete internally
consistent canonical binding without consulting alternative provider quotes.
Otherwise, it restricts matching to the candidate's own book and any supplied
namespaced provider identity, then requires exactly one strict line/price/market
match with an admissible provider timestamp. Conflicting aliases, partial provider
identity, missing, ambiguous and mismatched quotes remain unverified. No clock or
observation-time substitution is used. An opposing-price bookmaker cannot replace
the candidate's own book.

The app binds the real private frame before `prepare_live` and terminal authority.
Capture applies the same helper, preserving the earlier binding. A failed binding
also clears exact-quote verification so an old alias cannot bypass that failure.
A successful new binding updates its canonical verification alias; it does not
set model/calibration validation or override an explicit integrity veto.

## P1-B: explicit precedence

1. Authoritative input facts are the initial record and are not replaced by
   selected reporting-row adjustments.
2. Capture owns `snapshot_id`, `export_run_id`, `created_process_id` and
   `decision_bundle_version`.
3. Valid explicit start times and supported probability semantics survive.
   Existing canonical logic derives semantics and Eastern-to-UTC start fallback
   when explicit valid facts are absent.
4. There is no restoration step after derivation, so None/NaN/pd.NA/empty
   placeholders cannot erase derived facts. False and numeric zero remain values.

Authoritative capture does not synthesize missing model version, training cutoff
or model availability from a bundle manifest. Legacy callers retain their prior
binding, synchronization and descriptive artifact behavior.

A single private-contract addition, `market_push_probability`, is necessary to
preserve the supporting source fact used by existing explicit push-aware
semantics. No probability formula, public schema or validation threshold changed.

## End-to-end evidence

Real `build_best_picks_df` private frame, real app preparation and terminal
finalizer, real immutable capture, then SQLite readback: PASS. No final persisted
fields are injected after capture. Hermetic policy/exposure fixtures are test-only.

| Exact field | Verified stored value |
|---|---|
| candidate_id | original source candidate ID |
| quote_binding_verified | True |
| quote_bookmaker | draftkings |
| odds_recorded_at | 2026-09-14T14:55:00+00:00 (provider time) |
| provider_event_id | one (synthetic provider fixture) |
| provider_namespace | odds_api |
| probability_semantics | win_conditional_on_decision |
| game_start_utc | 2026-09-14T18:00:00+00:00, derived from 02:00 PM Eastern |
| model/calibration provenance | Exact source versions and availability/cutoff values unchanged |

Explicit valid unconditional-with-push and conditional semantics, canonical
starts, False, zero and capture-owned metadata are separately tested. The raw
quote fixture has no prebound canonical fields. The fallback-start scenario is
an evidence-persistence test, not authorization to use a missing start at wager
evaluation; existing start gates remain intact.

## Acceptance matrix

| IDs | Coverage | Result |
|---|---|---|
| QB-01–07 | Existing binding, unique binding, ambiguous/missing/mismatched rejection, provider time, end-to-end persistence | PASS |
| DF-01–08 | Derived and explicit semantics/start; nulls, False, zero, capture metadata | PASS |
| E2E-01 | Real app path and persisted readback | PASS |
| SAFE-01–07 | Ambiguous/missing quote, stale quote, missing validation, UNVALIDATED, Moneyline, factual Gemini veto, legacy labels | PASS; no funding |
| SAFE-08 | No ticket execution | PASS; no execution code added |
| LEGACY-01 | Existing non-authoritative capture tests | PASS |

## Validation

- Focused: **155 passed, 0 failed, 626 warnings, 17.85 seconds**.
- Full suite: **2,224 passed, 0 failed, 9,778 warnings, 103.95 seconds**.
- Exact CI production-safety command: **476 passed, 0 failed, 1,430 warnings, 33.94 seconds**.
- CI production compilation: **PASS**.
- `git diff --check`: **PASS**.

Logs are under `test-results/`: `p1-before.txt`, `p1-focused-verified.txt`,
`p1-full-final.txt`, `p1-safety-verified.txt`, `p1-compile-verified.txt`.

## Files and scope

- app_core/prediction_evidence.py: binding helper and authoritative precedence.
- streamlit_app.py: bind before prospective preparation.
- app_core/candidate_evidence_schema.py: preserve supporting market push mass.
- tests/test_prediction_evidence.py: binding/derivation/control/legacy regressions.
- tests/test_candidate_authority_projection.py: real pipeline and saved readback.
- tests/test_pipeline_identity_before_portfolio.py: raw provider quotes in fixture.
- This report.

No thresholds, validation standards, weights, model/calibration formulas, active
policies, deployment states, bankroll/exposure arithmetic, Gemini rules, parlay
rules, scheduler configuration, public styling/schemas, historical locks or
outcomes changed. No real wager authority activated.

UNRELATED CHANGES: NONE. Existing untracked article files remain untouched.
