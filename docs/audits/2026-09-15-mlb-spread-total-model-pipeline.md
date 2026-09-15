# MLB Spread/Total trained challenger pipeline — 2026-09-15

## Outcome and evidence boundary

The repository now has a genuine estimator-fitting, immutable artifact, verified loader and inference pipeline for MLB Spread/Total challengers. **No real-data model was trained or promoted in this change.** The available repository datasets do not establish the required historical point-in-time line and feature provenance. Tests train temporary synthetic fixtures solely to verify software behavior; none is shipped as a model or counted as validation evidence.

HEAD BEFORE: `5e3c292e` (main after merged PR #2283).
HEAD AFTER: commit containing this report; exact hash in delivery.

TRAINING DATA SOURCE: none meeting the complete point-in-time contract was found.
POINT-IN-TIME GUARANTEE: **FAIL for available historical data; fail-closed rejection implemented. PASS for contractual chronology/leakage regression tests.** A content hash verifies integrity, not authenticity of the original observation. A trusted capture/storage process must establish observation times; this trainer cannot make a retrospective file contemporaneous.
SPREAD TRAINING ROWS: 0 verified real rows used.
TOTAL TRAINING ROWS: 0 verified real rows used.
TRAINING PERIOD / VALIDATION PERIOD / HOLDOUT PERIOD: unavailable for a defensible real-data run.

### Existing sources inspected

- `app_core/mlb_history.py` and `output/mlb-history/audit.json`: 7,287 normalized games, 6,817 statistical feature rows across 2023–2025. The audit explicitly states that historical corrections/publication times are unverified and no historical odds were collected. Previous-game final-play timestamps alone cannot prove when a correction became known.
- `app_core/mlb_research.py`, `app_core/mlb_pitcher_history.py`, and `app_core/mlb_prospective.py`: prior ridge margin/total development and paired score forecasting. These are not fitted exact-line win-probability models with the new provenance contract. Reused the established six team-scoring features and timezone parser, not its historical evidence claims or calibration.
- Local `output/mlb-prospective-v1/report.json`: one captured game/cohort, zero graded games; no betting prices or win probabilities. This is not sufficient training/validation/holdout data.
- Local `output/mlb-evidence-audit/report.json`: master data has zero MLB rows; 885 descriptive recap rows across 68 exports have no verified pregame timestamps and cannot establish a chronological model benchmark.
- The live deterministic score distribution, configured weights, H2H trainer, blend fitting/calibration, candidate projection, prospective uncertainty, readiness/activation checks and immutable snapshot path were inspected. The H2H output is not reused as a Spread/Total target.

No old snapshot, history, closing observation, exposure record, active policy, existing fitted artifact or production probability formula was modified.

## Architecture

FEATURES: home/away prior runs scored per game, runs allowed per game and win percentage, plus exact reference line. All six statistics are recomputed from the same-season completed games actually present in the pregame observation; caller-supplied aggregate features are not consumed. Each team requires ten prior games. Each prior game needs a unique MLB-scoped ID, a Final status, valid scores and completion/availability no later than the pregame observation. Target-game rows, future observations and duplicate identities are rejected.

TARGETS: `spread_home`, `spread_away`, `total_over`, `total_under`.
MODEL FAMILY: separate ridge-logistic classifiers for home cover and total over. Away spreads negate the selected spread into the home reference line; away/under probabilities are the exact complement of the corresponding reference forecast. Complementary duplicates are deduplicated before fitting. Conflicting paired features or labels are rejected. This is not a Moneyline classifier.
PROBABILITY SEMANTICS: `win_conditional_on_decision`, P(win | win or loss). Exact-line PUSH and VOID labels are explicit and excluded from binary fitting and decided-result metrics, with exclusion counts retained in evaluation. No push is converted to a loss.

Transformations, ridge strength and optimizer settings are fixed in the versioned source; normalization is fit on development rows only. There is no hyperparameter search, holdout refit or automatic champion selection. CLI callers must predeclare whole Eastern-date train and validation endpoints; remaining slates are the later holdout. Chronology checks reject overlapping event times, same-slate overlap and prior-period outcomes unavailable at the next evaluation origin. Changing split dates after looking at results would constitute a new exploratory experiment, not prospective validation.

Validation and holdout metrics include sample and unique-event counts, Brier score, log loss, accuracy, decided-selection unit-stake ROI, season/probability-band breakdowns and outcome counts. ROI is descriptive for all recorded decided selections, not a staking strategy. Baseline comparisons use identical eligible rows and separately report coverage: raw bookmaker implied probability, contemporaneously recorded deterministic probability and contemporaneously recorded configured blend. Missing baseline observations are reported, not reconstructed using today's code. CLV is explicitly unavailable because the accepted schema has no verified closing observations.

MODEL ARTIFACT PATH: a successful future run writes `models/mlb_spread_total/<model_version>/estimators.json` and `manifest.json`. No real bundle is included here.
MODEL VERSION: unavailable for real data; runtime computes SHA-256 over canonical manifest inputs including exact fitted artifact hashes and feature schema, excluding only its own version field. Reusing identical artifacts/configuration/completion receipts reproduces the identity. A genuinely new training completion has a new receipt and therefore may have a new bundle version even if coefficients match.
MODEL TRAINED THROUGH: maximum outcome availability among the decided rows actually consumed by fitting. This is a conservative information cutoff, not merely scheduled start. Validation/holdout results never fit or tune coefficients. Their diagnostic metrics are labeled historical OOS replay and must not masquerade as prospective model-availability evidence.
TRAINING CUTOFF BASIS: `max outcome available_at among fitted decided training rows; no validation/holdout fitting or tuning`.
MODEL AVAILABLE AT: actual UTC trainer completion receipt after fitted bytes have been staged, immediately before completing publication of the immutable bundle. Readers reject incomplete bundles. Neither Git, filesystem mtime, deployment time nor inference time supplies this field. Existing version directories cannot be overwritten.

The loader checks version/digest, expected files, numeric parameter dimensions, positive scales, feature schema, semantics, family/targets, research state and chronology. Inference additionally checks artifact availability before prediction and prediction before game start, receipt integrity and feature/quote chronology. No fallback provenance is emitted.

## Live integration and authority isolation

The optional `PARLAYPICKER_MLB_CHALLENGER_MODEL` points to an exact version directory; there is no latest-model auto-selection or promotion. At the expanded candidate boundary, a candidate must provide a matching `mlb_pregame_receipt`, including MLB-scoped event ID, teams, exact start, target and line. Without those facts the challenger remains unavailable. The current live aggregate feature path does **not** synthesize a receipt from today's aggregates.

The verified inference response contains genuine `model_version`, `model_trained_through`, `model_available_at` and `training_cutoff_basis`, but is transported inside `mlb_challenger_result` alongside its own probability, semantics, generation time and receipt hash. `mlb_challenger_status` and status counts explain configuration/artifact/input failures. These private fields survive candidate projection, prepare/finalize and authoritative immutable snapshot capture/load. Baseline model/market/configured probabilities remain in their existing separate columns; baseline model provenance stays missing where it was missing. This avoids mislabeling the current blend as a prediction by the new model.

A freshly trained challenger returns RESEARCH, production_eligible=false, and $0. No production validator flags are minted. The existing sport/model/calibration/exposure gates remain required for any future promotion. `core/live_wager_contract.py` and `core/wager_decisions.py` are unchanged.

CALIBRATION COMPATIBILITY: scalar conditional probabilities are compatible with the existing numerical calibration routines. Any future calibration must be fitted from model/version-scoped OOS or prospective predictions and carry genuine availability and field scope. Historical replay in the trainer is never written as live evidence.
CALIBRATION ATTACHED: **NO**. The legacy blend calibration does not describe this model. Inference rejects a supplied calibration rather than silently applying or relabeling it. A model-specific OOS calibration producer and earned promotion are later work, not authority granted by training.

## Input and operation

Run from the repository root after audited, genuinely observed data is available:

```text
python scripts/train_mlb_spread_total_model.py --dataset <receipts.json> --source <audited-source-description> --train-through <YYYY-MM-DD> --validation-through <YYYY-MM-DD> --output models/mlb_spread_total
```

`receipts.json` is a nonempty list of records. Each record contains:

- `snapshot.payload`: `schema_version=mlb-pregame-receipt-v1`; `provider_namespace=mlb`; provider_event_id, home_team_id, away_team_id, season, game_start_utc; captured_at and prediction_cutoff; exact quote; observed prior games; optional contemporaneous baseline forecasts.
- `snapshot.sha256`: SHA-256 of UTF-8 JSON for payload with sorted keys, no NaN and compact separators. Use the module's `digest` function. This is an integrity receipt, not a way to manufacture provenance.
- `quote`: market_type, line, decimal_odds, sportsbook and observed_at. Selected spread sign is preserved; totals use the exact total.
- `prior_games`: MLB namespace, game_id, season, home_id, away_id, home_score, away_score, status=FINAL, completed_at, available_at for every contributing observed game. IDs and observations must come from trusted pregame capture, not later reconstructed files.
- Optional baselines keyed deterministic/configured_blend: probability, probability_semantics, source_version and generated_at, all valid before capture.
- `outcome`: the same provider/team/season/start identity, status FINAL or VOID, home_score/away_score for a final, and the actual observed outcome available_at. Outcomes are separate from pregame features.

All timestamps must have explicit timezones. The CLI exits nonzero with a blocked reason for legacy/malformed/missing data and does not create a model. A collector/provider adapter that jointly retains genuine pregame team-stat observations and exact line/price receipts is the concrete missing data prerequisite. Do not backfill those observation timestamps into old evidence.

## Acceptance and validation

MODEL VERSION REPRODUCIBLE: PASS on temporary fixture bundles; material model/config changes alter identity and existing versions cannot be overwritten.
LEAKAGE TESTS: PASS for future/target-game features, later quotes, capture timing, duplicates, same-slate boundaries and holdout independence.
SPREAD OOS METRICS: unavailable for real evidence; only fixture calculations tested.
TOTAL OOS METRICS: unavailable for real evidence; only fixture calculations tested.
BASELINE COMPARISON: implemented with paired coverage; no defensible real-data performance claim.
AUTHORITATIVE PROVENANCE ROUND TRIP: PASS, namespaced challenger facts unchanged and baseline facts not fabricated.
TAMPER FAIL-CLOSED: PASS for bytes, manifest digest, schema, semantics and chronology.
AUTOMATIC PROMOTION: NO.
PRODUCTION ELIGIBILITY AFTER THIS PR: NO new eligibility or nonzero authority.

### Required test coverage

| Requirement | Regression evidence |
| --- | --- |
| MODEL-01/02 | Explicit chronological split, strict event order and disjoint Eastern slates. |
| MODEL-03 | Recomputed prior-game features reject target IDs, future availability, later quotes and invalid capture times. |
| MODEL-04/05 | Exact spread/total WIN/LOSS/PUSH/VOID labels; pushes/voids cannot satisfy minimum binary training rows. |
| MODEL-06/07/08 | Repeated fixed bundle inputs reproduce version; changed configuration/bytes change identity; cutoff derives from fitting input and precedes availability. |
| MODEL-09/10 | Byte tampering, malformed manifest, rehashed unknown schema/semantics and invalid chronology fail closed. |
| MODEL-11/12/13 | All four artifact provenance fields returned and retained, scoped to the challenger, through real builder/prepare/finalize/authoritative capture/load. |
| MODEL-14/15 | Missing artifact supplies no provenance; prediction before availability or at/after game start is rejected. |
| MODEL-16 | Blend calibration cannot attach to the new trained model. |
| MODEL-17 | Artifact and inference remain RESEARCH/$0; live terminal path does not gain authority from challenger facts. |
| MODEL-18 | Moneyline target rejected; existing Moneyline isolation tests retained and passed. |

FULL TESTS: 2,371 passed (final run).
PRODUCTION SAFETY: 545 passed using the current CI test list.
FOCUSED TESTS: 209 integration/provenance/target checks passed; final new-model suite 42 passed after malformed-input guards.
PRODUCTION COMPILE: PASS, including the new trainer.
GIT DIFF CHECK: PASS.
Validation ran locally on Windows; hosted CI runs after the PR is opened.

## Files changed

- `app_core/mlb_spread_total_model.py`
- `scripts/train_mlb_spread_total_model.py`
- `core/streamlit_pipeline.py`
- `app_core/candidate_evidence_schema.py`
- `tests/test_mlb_spread_total_training.py`
- `tests/test_mlb_spread_total_model.py`
- `docs/audits/2026-09-15-mlb-spread-total-model-pipeline.md`

UNRELATED CHANGES: NONE. Pre-existing Medium article files are excluded.
