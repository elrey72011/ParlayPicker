# Sport deployment evidence assessment — 2026-09-14

## Decision

All six currently supported sports remain **UNVALIDATED ($0)** based on the repository artifacts inspected. This is a provenance/validation finding, not a universal sample-count rule or an assertion that remote evidence does not exist. No remote private evidence store was retrieved during this assessment. No validated policy ID or probability was fabricated.

| Sport | Graded backtest CSV rows | Master feature rows | Deployment | Missing support |
|---|---:|---:|---|---|
| NFL | 37 | 16 | UNVALIDATED | Timestamped pregame model/calibration provenance; frozen historical-prior weighting validation; current-season update validation; paired prior-slate CLV; uncertainty coverage validation |
| NCAAF | 7 | 65 | UNVALIDATED | Same football-specific provenance, historical-prior, updating, CLV and uncertainty evidence; cannot infer from pooled sports |
| MLB | 885 | 0 | UNVALIDATED | Sport/market-specific prospective or chronological validation with pregame provenance, current-season weighting, paired CLV and conservative uncertainty coverage |
| NBA | 2 | 44 | UNVALIDATED | Same sport-specific validation/provenance, current-season evidence weighting, CLV and uncertainty controls |
| NHL | 4 | 65 | UNVALIDATED | Same sport-specific validation/provenance, current-season evidence weighting, CLV and uncertainty controls |
| NCAAB | 0 | 227 | UNVALIDATED | Pregame prediction/outcome validation (feature rows alone are not predictions), current-season weighting, calibration, CLV and uncertainty controls |

Counts are raw rows, not independent games, effective sample sizes or validation successes. Backtest CSVs also contain 66 WNBA rows; WNBA is outside the current SportPolicy sport set. No inference from WNBA authorizes another sport.

## Inspected artifacts

- `data/prediction_evidence/evidence.sqlite3`, opened read-only: 82 bundles, **0 snapshots, 0 score revisions**. Bundles freeze code; they are not prospective predictions or outcomes.
- `data/backtest_exports/*.csv`: columns include outcomes and effective probabilities, but no prediction-generated timestamp, training cutoff, model availability/version, calibration version, exact quote timestamp or closing quote. Filename dates do not establish those facts. Historical graded exports can support research, but do not establish leak-safe production validation alone.
- `data/master_all_sports.csv`: 417 feature/outcome rows. Has game start and identity columns but lacks recorded pregame prediction, frozen model/training and calibration provenance. These cannot be counted as prospective prediction records.
- `data/tier_results/tier_results_log.csv`: 122 rows with date/matchup/pick/tier/outcome, no sport column, original model/calibration/probability or quote provenance. Display tiers are not validation.
- `data/calibration/effective_prob_calibration.json`: pooled calibration reports 1,155 graded rows, 924 training and 231 holdout rows, and a promotable flag. The artifact is not sport-specific and does not provide the missing pregame provenance or paired CLV. Its train-end and test-start timestamps both show 2026-08-16 00:00 UTC; this does not establish slate-disjoint validation. The flag is not deployment authority.
- `core/football_evidence.py`: a research-only freeze/prior/uncertainty implementation. No validated live policy/evidence artifact was found for its prior decay, effective-sample cap, current-season weights or uncertainty quantile. The existence of the algorithm does not validate its outputs.

## Implemented deployment semantics

`SportPolicy.deployment_state` defaults to UNVALIDATED. The explicit states are UNVALIDATED, PROVISIONAL_VALIDATED, STANDARD_VALIDATED and PREMIUM_VALIDATED. A state's ceiling can only block a candidate maturity above it; it never promotes a candidate. A validation ID alone no longer enables a policy whose state is UNVALIDATED. Unknown state strings are rejected.

PROVISIONAL_VALIDATED permits only independently qualified Provisional straight recommendations. Standard permits Standard and lower; Premium permits Premium and lower. The existing candidate maturity, positive conservative EV, exact quote, calibration, model, identity, review and portfolio gates all still apply. Provisional remains excluded from production parlay legs. Legacy APPROVED fields have no authority.

`provisional_minimum_evidence` is separate from `minimum_evidence`. It applies to effective evidence, not a required large current-season exact bucket. Each sport may have independently validated provisional settings, including higher-frequency sports; enabling a setting for one sport changes no other policy. NFL/NCAAF may use leak-safe historical plus current-season evidence only after the weighting/calibration/CLV/uncertainty policy is validated. No new empirical cutoff was invented in this change. Missing effective evidence still blocks, including when the configured numerical minimum is zero.

The previous report's statement that automatic maturity assignment was omitted still applies: deployment state is a ceiling, not an automatic candidate promotion. This clarification is implemented without needing such promotion.

## Required next evidence, before funding

For each sport, collect/freeze actual pregame predictions with model availability/training cutoff, calibration version, exact market/line/price and timestamp, then append independently matched outcomes and prior-slate closing quotes. Validate sport-specific chronological/slate-separated calibration and uncertainty coverage, prior/current-season weights and exposure policy on evidence held out from tuning. NFL/NCAAF validation may pool appropriately weighted older seasons and parent markets with uncertainty control; it need not wait for a large child bucket. Higher-frequency policies should explicitly validate their current-season weighting. Standard/Premium require stronger separately supported criteria, not merely a larger display label.

The fresh committed-exposure snapshot remains mandatory before any non-zero allocation. No live policy was activated or deployed, and no historical record was changed.

## Verification

Full regression suite: **2,119 passed**, 9,261 warnings, 97.31 seconds. Includes 27 new state/football evidence tests across all six sports. git diff --check passed. An initial focused invocation encountered a Windows temporary-directory permission error; the full run used the writable test directory and passed. Changes remain local and uncommitted.
