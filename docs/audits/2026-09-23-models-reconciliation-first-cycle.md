# Model, reconciliation, and first-cycle audit — 2026-09-23

Baseline: merged `main` commit `e3c30108827df6da1f7626fa04a552caa8b2c46c`. This change implements the prospective research path; it does not activate a market, fund a stake, or place a wager.

## Six new research model scopes

| Scope | Research target and features | Legal fitted observations here | Model/calibration |
| --- | --- | ---: | --- |
| NBA/SPREAD | Home final-score margin distribution; prior authenticated scores, team form, home/away and rest; transform against exact spread line. | 0 | `INSUFFICIENT_EVIDENCE` / none |
| NBA/TOTAL | Final combined score distribution; prior authenticated scores, scoring/allowing form and rest; transform against exact total. | 0 | `INSUFFICIENT_EVIDENCE` / none |
| NCAAB/SPREAD | Provider-bound home margin distribution with the same chronological form/rest features; no fuzzy school identity. | 0 | `INSUFFICIENT_EVIDENCE` / none |
| NCAAB/TOTAL | Provider-bound combined score distribution with chronological form/rest features. | 0 | `INSUFFICIENT_EVIDENCE` / none |
| NHL/PUCK_LINE | Home goal margin distribution with prior authenticated goals, form and rest; exact puck-line transform, never moneyline substitution. | 0 | `INSUFFICIENT_EVIDENCE` / none |
| NHL/TOTAL | Combined goal distribution with prior authenticated goals, form and rest; exact total transform. | 0 | `INSUFFICIENT_EVIDENCE` / none |

The pipeline needs at least 60 independent legal events **per exact scope**, including at least 18 chronological validation events. It selects ridge regularization using development/validation score distributions, then persists the selected model, result IDs, features, hashes, runtime, source commit, chronological cutoffs, Brier/log loss/calibration/coverage and ROC/PR diagnostics. Features use only results available before the pregame observation. Scores are research training targets; they are not certified book settlements. Until an authentic settlement adapter exists, no new-sport market calibration is fit. Injury/lineup, ranking/efficiency, and goalie features are omitted where timestamped authenticated inputs do not exist. Every future fitted output remains `UNVALIDATED`, `production_eligible=false`, `$0`.

## Legacy reconciliation and sample integrity

The NFL, NCAAF and MLB reconciler reads source SQLite stores in read-only mode, verifies immutable raw/source hashes, and appends source-bound research facts to separate canonical ledger tables. It includes older MLB prospective forecast records and immutable MLB receipts without rewriting either source. Repeating the import leaves counts unchanged; changed evidence under an immutable ID fails. Old forecasts and score comparisons never become live `prospective_prediction` rows. MLB report fields distinguish receipts, games, market rows, settled units and slates; market rows do not inflate independent games.

The local workspace has **no native source store for any of the six sports**, so local reconciliation saw no authentic records. NFL/NCAAF/MLB source, accepted, and duplicate counts are locally zero; remote counts are **unknown** until an authenticated restore. The previous MLB receipt audit numbers are historical external context, not this run's canonical counts.

## Frozen policy and remote state

The version 1 plan source declares exactly one current plan for every sport/market family, with a hash-bound fixed methodology and future validation/holdout windows. The [plan audit](2026-09-23-prospective-validation-plans.md) lists all 12 stable plan IDs and policy hashes. Installation in an authenticated canonical store remains blocked by missing Drive credentials. A local install can verify code and produce local artifact hashes, but cannot establish remote durability.

The six-sport scheduled entry point was invoked after tests with `NFL,NCAAF,NBA,NCAAB,MLB,NHL`. It returned `MISSING_CREDENTIALS`, `requested_slate_success=false`, before any provider or remote call. Missing configuration: `PARLAYPICKER_DRIVE_FOLDER_ID`, `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`, `ODDS_API_KEY`, and `CFBD_API_KEY`. Thus there is no authenticated run ID, discovered event/quote/prediction/grade count, restored remote source count, or verified remote backup to report. The sanitized local blocker audit is `/private/tmp/parlaypicker-next-six-sport-cycle-audit.json` for this workstation only. The updated Actions workflow retains a sanitized JSON audit artifact when a credentialed run occurs.

## Verification

- Full pytest: 2,878 passed, 15 skipped, 38 subtests passed.
- Canonical CI shards: 1,552 passed / 3 skipped and 1,326 passed / 12 skipped.
- Production safety selection: 553 passed.
- Node: public refresh and parlay products passed.
- Chrome: public site, results filters, and generated bundle passed.
- Python compileall and `git diff --check`: passed.

The dashboard exposes all 12 rows with source observations, canonical predictions, independent validation/holdout counts, model/calibration/plan identities, price/close status, deployment state, and next blocker. It labels local counts `LOCAL_COUNTS_ONLY_REMOTE_UNKNOWN` until all remote stores are restored and the canonical backup is read back. Verified closes remain unavailable and CLV cannot be claimed from near-start candidates. Straight-leg and parlay-product gates retain separate validation and activation requirements. No evidence was fabricated, no gate was weakened, no market was auto-activated, and no wager was placed.
