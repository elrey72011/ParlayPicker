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
