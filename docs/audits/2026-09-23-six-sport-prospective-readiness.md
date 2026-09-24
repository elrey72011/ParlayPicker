# Six-sport prospective wager readiness, 2026-09-23

## Scope and current result

The research registry covers exactly NFL, NCAAF, NBA, NCAAB, MLB, and NHL. Its
market families are independent: football and basketball spreads and totals,
MLB run lines and totals, and NHL puck lines and totals. The canonical
prospective evidence store has a separate readiness row for every one of the
12 combinations. On this checkout, **none of the private research SQLite
stores or canonical prospective evidence database is present**. The 12 local
canonical rows therefore each report zero prospective predictions, zero
independent settled events, no frozen validation plan, `UNVALIDATED`,
`production_eligible=false`, and a recommended stake of `$0`. The local zero
is not a claim that the remote Drive backups contain no evidence.

| Sport | Market | Local canonical predictions / settled events | Model | Calibration | Plan | Deployment | Next canonical blocker |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| NFL | SPREAD | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NFL | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NCAAF | SPREAD | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NCAAF | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NBA | SPREAD | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NBA | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NCAAB | SPREAD | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NCAAB | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| MLB | RUN_LINE | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| MLB | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NHL | PUCK_LINE | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |
| NHL | TOTAL | 0 / 0 | Missing | Missing | Missing | UNVALIDATED | NO_PROSPECTIVE_EVENTS |

“Missing” in this table means absent from the **new local canonical store**.
It does not erase older research models. For example, NCAAF retains its
frozen-model/recovery path, and MLB retains its research receipt and challenger
model history. Their historical research records are read-only projections,
not retroactive prospective validation. A separate owner-supplied MLB snapshot
was audited on September 23: 308 receipts across 77 games, 45 settled games
and 180 settled market records. That dated external snapshot was too small for
the required chronological train/validation/holdout split and is not included
in the zero local canonical counts. See `2026-09-23-mlb-receipt-readiness.md`.

## Architecture and persistence

- `app_core/prospective_sport_adapters.py` registers exactly six adapters.
  Existing MLB, NCAAF, and NFL capture and grading stay behind their current
  sport-specific implementations. NBA, NCAAB, and NHL use the same bounded
  provider contract but write to separate append-only sport research stores.
- `app_core/prospective_evidence.py` adds `prospective-evidence.sqlite3` beside
  the existing prediction evidence database. The schema is additive, enforces
  foreign keys, disallows updates/deletes to evidence and review tables, hashes
  payload/source bytes, rejects nonfinite numbers, and checks timezone-aware
  chronology. It stores events, exact quotes, close observations, results and
  correction revisions, predictions, model training result IDs, calibration
  fit result IDs, frozen plans, reproducible validation reports, and independent
  deployment reviews. Reading an absent database returns 12 `$0` rows without
  creating a file. No existing sport database is migrated or rewritten.
- `app_core/prospective_source_view.py` projects old MLB/NCAAF/NFL and new
  NBA/NCAAB/NHL research stores into one descriptive field contract, leaving
  absent model, calibration, provider, close, or settlement facts null with
  blockers. Its separate 12-row local source inventory counts research quotes,
  captured events and observed scores without claiming the canonical validation
  store contains those records. The readiness screen displays that inventory
  alongside the canonical deployment table. Projection cannot insert a
  production prediction or grant authority.
- Model training and calibration registration are scoped to one sport/market,
  reference exact available result IDs, and preserve the training/fit cutoffs.
  Frozen validation plans predeclare independent event sampling, validation and
  holdout windows, probability/price/CLV/ROI policies, and the target tier.
  Metrics and artifact validity are recomputed after result corrections or new
  evidence. Reported ROI is hypothetical paper ROI; accepted-wager ROI remains
  unavailable until actual accepted wagers exist.
- The straight-wager terminal check consumes the exact market's reviewed
  validation artifact plus a separate hash-bound owner activation and current
  bankroll/exposure authority. A candidate also needs its exact model,
  calibration, provider event/team identity, fresh verified quote, chronology,
  conservative value, and existing context/news gates. Parlay leg admission
  retains its separate product validation after straight-leg eligibility.
  Merely passing validation never sets stake or activates a market.

## Scheduler and provider requirements

The scheduled research workflow now defaults to all six sports after adapter
contract and zero-slate coverage, while `RESEARCH_SPORTS` and manual workflow
input can select an exact subset. `RESEARCH_SCHEDULER_ENABLED=true` still
controls execution. Each sport restores its own remote store before mutation,
and backup uses remote read-back verification. The health report separates
restore, capture, grade, close capture, backup, discovered/captured/graded
counts, sanitized provider/config blockers, and per-sport API budget. A failed
requested sport prevents `requested_slate_success` even if other sports pass.

| Sport | Current authentic feeds and operating needs | Readiness limitation |
| --- | --- | --- |
| NFL | The Odds API key for provider market tracking and scores; verified Drive credentials; an NFL producer with replayable training/runtime, injury and recent-result context, exact prices and comparable closes | Existing market tracker alone has no validated prediction/calibration or context/close proof. |
| NCAAF | CFBD API key and The Odds API key; existing frozen model and provider season/week/event mapping; verified Drive credentials | Recovery never retroactively makes a model trained or deployed; replayable exact quotes, calibration, closes and independent validation are still needed. |
| NBA | The Odds API v4 `/events`, `/participants`, `/odds` (`spreads,totals`), and `/scores` access; US sportsbook markets; verified Drive credentials | Market capture is research only; no sport/market producer, calibration, settlement rules or validated close authority. |
| NCAAB | Same Odds API v4 endpoints for `basketball_ncaab`, with exact provider participant IDs; verified Drive credentials | Fuzzy school/team matching is barred from production; no model/calibration/validation. |
| MLB | MLB Stats API and existing research receipt feeds, The Odds API exact line/price source, frozen model where used, verified Drive credentials | Existing receipt chronology is insufficient for the independent split in the dated audit; no automatic promotion of the challenger. |
| NHL | Same Odds API v4 endpoints for `icehockey_nhl`, with `spreads` treated as puck line and `totals` separate; verified Drive credentials | Moneyline cannot stand in for puck line. Push-aware market settlement, producer/calibration and comparable closes remain unverified. |

The three new adapters retain raw provider event, participant, odds and score
responses with exact IDs and response times. They reject ambiguous identity,
incomplete/stale market pairs, bad prices and nontransient provider failures.
The current close path retains **pregame close candidates** as research only;
the canonical writer rejects `close_verified=true` until a certified final-close
source and replay verifier exist. It does not label a live snapshot as a
verified close or calculate CLV from it. Scores remain research observations
until market-specific settlement is verified. Missing credentials/access,
source identity, legal chronology, a frozen plan, or verified backup are
operating blockers that cannot be replaced with synthetic records.

## Changed files

New implementation files:

```text
app_core/odds_market_store.py
app_core/odds_research_adapter.py
app_core/prospective_evidence.py
app_core/prospective_legacy_view.py
app_core/prospective_source_view.py
app_core/prospective_sport_adapters.py
core/sport_market_activation.py
core/sport_market_gate.py
scripts/activate_sport_market.py
```

Existing integration files:

```text
.github/workflows/research-scheduler.yml
app/ui/readiness_dashboard.py
app_core/candidate_evidence_schema.py
app_core/production_parlays.py
app_core/research_api_budget.py
app_core/research_scheduler.py
core/candidate_maturity.py
core/live_wager_contract.py
core/market_policy.py
core/true_parlay_engine.py
core/wager_decisions.py
scripts/run_research_scheduler.py
```

New test files are `tests/test_prospective_evidence.py`,
`tests/test_prospective_legacy_view.py`,
`tests/test_prospective_source_view.py`,
`tests/test_six_sport_adapters.py`, and
`tests/test_sport_market_gate.py`. Existing affected fixtures and regressions
were updated in `tests/activation_fixture.py`,
`tests/test_live_wager_contract.py`,
`tests/test_true_parlay_engine_unittest.py`, and
`tests/test_wager_integrity_audit.py`.

## Rollout and test evidence

The implemented framework supports phases A–E of the task: common adapters,
research-only new-sport capture, legacy source projection, all-six scheduling,
and separately scoped registration/validation machinery. Phases F–H require
future authentic prospective observations, frozen plans before holdouts,
independent review, and explicit owner activation. No market was activated
with pooled sport or market data; no order placement was added.

Local verification after the final implementation: full pytest **2,845 passed,
15 skipped, 38 subtests passed**; canonical production-safety selection **553
passed**; canonical CI shard 1 **1,486 passed, 10 skipped**; canonical CI shard
2 **1,359 passed, 5 skipped**. Five Node/browser scripts passed (public
refresh, parlay products, public results filters, live site behavior, and a
generated bundle in Chrome). `compileall` and `git diff --check` passed. The
remote PR checks are reported with the PR once complete.
