# Progress against the ParlayPicker master brief

The supplied `ParlayPicker Plan.md` is a requirements brief asking for an implementation plan, not a numbered execution checklist. This map uses its actual requirement groups. Status describes repository implementation; a merged feature is not proof of live operation or predictive quality.

**Current position: core selection/evidence infrastructure exists; NCAAF is entering prospective research validation. The six-league product is not production-validated or launch-complete.** No defensible overall completion percentage follows from the brief.

| Master brief requirement | Current status | What remains |
|---|---|---|
| One recommendation per game, including No Bet | Implemented foundation | Continue end-to-end validation across every required league and market. |
| Schedules, odds, normalization and missing-data guards | Partial | Provider coverage gaps, stale inputs and cross-source identity require continued validation. |
| All six leagues: NFL, NBA, MLB, NHL, NCAAF, NCAAB | Partial | Feed/UI presence does not imply independently validated models for all six. |
| Rich pregame features: injuries, starters, rest, travel, weather, opponent strength | Partial | The NCAAF experiment uses lagged scoring/yardage and neutral-site status; it does not implement the full requested feature set. |
| Separate sport/market probabilities | Partial | Existing score models have limited league coverage; NCAAF remains isolated research, not an approved production model. |
| Baselines and chronological train/calibrate/test | Implemented for first NCAAF research experiment | Demonstrate improvement and validate future changes on new data. |
| Historical data collection, deduplication and coverage audit | Implemented for NCAAF research | Provider schedule completeness and historical publication/correction times remain unverified. |
| Calibrated win/push/loss, EV and quality gates | Implemented foundations; research assumptions remain | Validate probability calibration and execution semantics by sport/market. |
| Prediction snapshots, timestamps, versioning, audit trails | Implemented foundation | Operational monitoring, repeatable recovery drills and complete capture coverage. |
| Durable external evidence | Shared Drive adapters and read-back verification implemented | Run and verify each new prospective record backup in deployment. |
| Prospective NCAAF predictions at dated market prices | Implemented by this change; live verification pending | Freeze, refresh, capture before kickoff, back up, and accumulate outcomes. |
| ROI, hit rate, Brier/log loss and calibration | Research/paper reporting implemented | Adequate prospective sample; actual-execution evidence and stronger uncertainty analysis. |
| Closing-line value, opening lines, line movement | Manual closing proxies and exact-line price CLV implemented; deployment verification pending | Automated final-close coverage, no-vig comparisons and opening-line history remain. |
| Automated pregame batches, controlled refreshes and alerts | Incomplete | Prospective workflow is currently manual; scheduling/monitoring must be implemented separately. |
| Dashboard, picks, parlays, portfolio and exports | Implemented foundations | Complete UX acceptance, accessibility and cross-league behavior checks. |
| Retraining, drift detection, rollback | Partial | Version hashes exist; production monitoring and governed retraining/rollback need completion. |
| Security, licensing, compliance and responsible-gambling launch checks | Not established by this work | Complete documented operational/product review; secrets configuration alone is not the checklist. |
| Load, security and live end-to-end tests; incident response | Partial | Unit/UI/integrity tests do not substitute for launch acceptance and operational drills. |
| Proven betting edge / profitability | Not established | Collect prospective evidence. The brief explicitly prohibits promising profitability or fabricating edge. |

## Evidence in the repository

- Selection and data guards: `core/streamlit_pipeline.py`, `core/line_evidence.py`, `core/run_readiness.py`.
- General evidence and recovery: `app_core/prediction_evidence.py`, `app_core/evidence_remote.py`, `app_core/evidence_drive.py`.
- NCAAF history: `app_core/ncaaf_history.py`, [collection workflow](ncaaf-historical-collection.md).
- Fixed research protocol: `app_core/ncaaf_research.py`, [experiment protocol](ncaaf-research-protocol.md).
- This step: `app_core/ncaaf_prospective.py`, `app_core/ncaaf_prospective_store.py`, [prospective operation guide](ncaaf-prospective-evaluation.md).

## Next acceptance milestones

1. Verify a frozen cohort and a pregame capture in deployed Streamlit, including a successful Drive restore/read-back.
2. Grade actual completed games and review capture exclusions, probability calibration and paper returns without outcome-based filter changes.
3. Verify manual closing-proxy capture, then add controlled pregame scheduling with failure visibility.
4. Decide on model improvement using new evidence; require a new evaluation period for tuned models.
5. Extend the same evidence discipline to remaining required leagues and complete the operational launch checklist.

The user's 75% aspiration is not a completion criterion specified in the brief and is not demonstrated by implementing these features. Winner prediction accuracy and profitable approved wagers remain different measurements.

## MLB historical dataset milestone

A bounded, resumable [MLB CLI collector](mlb-historical-collection.md) now builds same-season features from prior completed games with source game IDs and separate targets. Full collection and chronological benchmark evaluation remain pending. This extends historical-data infrastructure; it does not establish MLB accuracy or production eligibility.

## MLB fixed research comparison

The [MLB protocol](mlb-research-protocol.md) now implements a 2023 train / 2024 calibration / 2025 test comparison of ridge regression, constant and lagged-scoring baselines. The first retrospective run used 2,273 / 2,270 / 2,274 eligible games. Ridge winner accuracy was 53.1% versus 53.7% for the scoring baseline; ridge also had worse Brier score and log loss. There is no demonstrated improvement supporting promotion. Historical odds, prospective MLB validation and the richer feature set remain outstanding.

## MLB pitcher feature development

The [starter development workflow](mlb-pitcher-development.md) adds resumable 2023–2024 boxscore collection, strictly prior-appearance pitching rates, and a matched team-only versus starter-enhanced ridge comparison. Full pitcher collection and development metrics remain pending. Actual starters are retrospective; pregame announcement evidence and prospective evaluation remain required. No 2025 retest or production activation is included.

## MLB prospective paired forecasts

[Manual MLB prospective evaluation](mlb-prospective-evaluation.md) now freezes both development models and captures paired score forecasts with timestamped provider-listed probable pitchers. Grading retains starter changes and reports each cohort separately; Drive evidence is isolated from NCAAF. A local live capture was verified. Deployed Streamlit capture and Drive recovery remain to verify. This does not establish confirmed-lineup timing, betting returns or model promotion.

## Scheduled research operation

An opt-in [GitHub Actions research scheduler](research-scheduler.md) now orchestrates existing MLB/NCAAF frozen captures, bounded grading and verified Drive backup. Repository secret configuration, first live scheduled run and deployed restore verification remain pending. Scheduling is best-effort; no production wagering or predictive accuracy claim changes.

## Scheduler spending and hours

Automation now reserves durable paid API budgets (500 CFBD requests / 5,000 Odds credits per rolling 31 days, with daily caps) and operates every 30 minutes from 11:45 a.m. through 2:15 a.m. Eastern the following morning, with a 2:30 a.m. cutoff. Manual/prior API usage is outside these counters. Scheduled NCAAF was temporarily paused during implementation; restore `RESEARCH_SPORTS=MLB,NCAAF` after merging and verifying this budgeted workflow.

## NFL market-tracking milestone

[Odds-only NFL tracking](nfl-market-tracking.md) adds scheduled pregame moneyline, spread and total snapshots, final-score comparisons, verified Drive backup, and a Streamlit restore/export panel. It uses the existing Odds key and shares a 7,500-credit rolling-31-day automation budget with NCAAF (200 credits/day). This addresses NFL data collection and operational evidence. Independent NFL features, model development, calibration, prospective prediction validation, and wager approval remain outstanding. A zero-record run can verify access but cannot establish capture/grading success or predictive quality.

## Cleaner daily interface

The Streamlit private beta now opens on Today, followed by Pick Details and Results. Today uses the reconciled game export and displays only funded, approved game wagers; research passes retain their explanations in Pick Details. Player props, parlays, candidate exports, readiness and storage diagnostics remain in Workspace. Sidebar uploads, model controls and prospective tools are grouped under Settings & research. Final-score refresh is explicit instead of running automatically on first page load.

This advances the brief's usable dashboard milestone. It does not complete the public subscription product: authentication, user isolation, payments, data redistribution permissions and evidence supporting public recommendation claims remain separate work. Prediction models, approval thresholds, frozen cohorts and scheduler budgets are unchanged.

## Results evidence overview

Results now summarizes the loaded game recap by sport and separates explicitly approved positive-stake rows from research/unapproved rows. It displays win/loss/push counts, unresolved and void counts, and the decision denominator. Paper returns use one unit per settled row with valid exported American odds; unavailable prices are excluded without a replacement price. Actual betting returns remain unavailable without execution evidence.

The evidence panel reads local MLB forecast comparisons, NCAAF paper reports by frozen cohort/model, and NFL market-tracking counts separately. It shows saved capture/score times and the general evidence store's process-local sync status without inferring separate sport-store or scheduler backup health. No new scheduled API calls are added. Score fetching is a one-click action, and refreshed recap data replaces stale cached source data while preserving edits on unchanged sources.

## Featured daily pick views

Today adds Overall Best Pick, Sides (moneylines/spreads), and Totals (over/under) views. Each features one selection from the finalized game card, ranked by final production win probability, then production edge and EV. Approved rows lead; when none qualify, the best available research row remains explicitly PASS. A composite ranking score is never substituted for a win probability, and unavailable final probabilities, unusable prices, and explicitly started games are not featured. These are category views of the final one-selection-per-game export, not independent selections from every candidate market. Estimates explain the ranking and do not establish a win or achieved accuracy. No production thresholds or wager authorizations change.

## Per-game Overall, Sides and Totals boards

The three Today views now retain one row for every game in the finalized slate. Overall preserves the final selection. Sides and Totals independently use the candidate audit's per-family rank, matched to the game and run; a missing category stays visible as No Bet/market unavailable. Exact final-ticket matches retain final production metrics and approval. Alternatives remain zero-stake research PASS rows with their own model estimates and candidate ranking scores. Composite scores are never displayed as probabilities. The earlier single-featured-pick view is superseded by these per-game boards.

### Per-game selection and wager status clarification — September 8, 2026

Each game board now labels the selection Best Overall, Best Side, or Best Total separately from wager status and displays its wager explanation. Missing markets are Unavailable. Alternatives retain zero approved stake and explain that final wager and portfolio checks have not approved them. Pick Details uses the same production EV and edge fields as the overall board.

Ranking investigation: the supplied Athletics exports show Under 9.5 with selection score 0.531906 and EV -0.012129, versus Athletics +1.5 with score 0.509219 and EV +0.072360. This agrees with the existing composite-score-first ranking contract; it does not establish that the higher-EV alternative passes production checks. Ranking and approval thresholds remain unchanged. Existing gate explanations may describe an earlier calibration-stage value rather than the final displayed production edge; they remain preserved as the recorded decision explanation.

Validation: 13 focused selection and Streamlit tests passed, including positive-EV alternatives, unavailable markets, and separation of selection from approval. Live deployment verification remains pending merge.

### Cleaner boards and measurable Gemini review — September 8, 2026

Primary game tables now show game, pick, odds, win estimate, EV estimate, and wager status. Full selection diagnostics remain expandable and downloadable. Gemini review metadata is visible in Pick Details and retained in immutable snapshots for future grading. Results includes a descriptive comparison of first pregame reviewed model selections against the Gemini-agreement subset, with model versions separated.

Structured reviews use a configurable local daily request budget (default 20, including retries) and ten-minute exact-batch caching. This is not an account-wide or redeployment-proof cap. Additional context is admitted only with a recent source timestamp; feeds without those fields remain unavailable. No new provider subscriptions or fetches were added, and ranking/probability/approval rules remain unchanged. See `docs/gemini-review.md` for configuration and limitations.

Validation: full regression suite 1,559 passed; 40 focused tests passed after exact-candidate context recovery and capture-to-grading checks. Deployment and future statistical validation remain pending.
