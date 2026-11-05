# ParlayPicker roadmap: the best pick for each game

Version 3, 2026-11-05. This supersedes version 2 (same day). Version 2 was written
without a full audit of the repository and planned parallel systems for things the
repository already does. Version 3 is built around the authority chain that already
exists. Statuses were checked against `main` `cc42eac` plus open PR #2457.

## 1. The goal, and the path that already exists

**The goal.** For every game: one recommendation, either a bet or No Bet. A pick gets
a stake only when its exact sport and market has *proven* it beats the market on
prospective evidence.

**The path already exists in the repository.** Every step below is implemented. A
pick is funded only when all five hold:

| Step | What it requires | Where |
|---|---|---|
| 1. Frozen validation plan | One plan per sport × market, 12 in all: NFL, NCAAF, NBA and NCAAB SPREAD/TOTAL, MLB RUN_LINE/TOTAL, NHL PUCK_LINE/TOTAL. Thresholds: ≥ 200 independent events, Brier ≤ 0.24, log loss ≤ 0.68, calibration error ≤ 0.08, ≥ 90% coverage, ≥ 80% certified closes, paper ROI ≥ 0, on both validation and holdout | `app_core/prospective_validation_plans.py`, frozen by the research scheduler |
| 2. Prospective evidence | Events, verified entry quotes, certified closes, results, models, calibrations and predictions, each immutable and hash-bound | `app_core/prospective_evidence.py`, backed up to Drive |
| 3. Validation and deployment review | `evaluate_validation_plan` → `create_validation_artifact` → `record_deployment_review` → `deployment_state` | `app_core/prospective_evidence.py` |
| 4. Owner market activation | Signed, expiring, bound to the exact deployment | `core/sport_market_activation.py`, `data/policies/active_markets/` |
| 5. Exposure ledger | Current bankroll, unit and limits | `core/exposure_ledger.py` |

Ordinary Refresh applies the whole chain to every candidate
(`core/live_wager_contract.finalize_live_wagers` and `core/sport_market_gate.py`).
Owner-authorized **controlled trials** (`app_core/controlled_trial.py`) are the only
other way to recommend a capped pick while evidence accumulates.

**Today no step-4 activation exists, so the funded card is empty by design.** That
is the correct state until a market passes its plan.

## 2. Where we are (2026-11-05)

### Built and working in the repository

- **Streamlit app.**
  - Main tabs: Today, Pick Details and Results.
  - Workspace sub-tabs: Odds, Analysis, Full Pick Board, Parlays, Portfolio, Diagnostics, Strategy Lab, and Preview & Publish.
  - Research tools for NFL, NCAAF and MLB, and Gemini review.
- **Refresh covers all leagues.** NFL, NCAAF, NBA, NCAAB, MLB, NHL and WNBA. Each pick's probability blends Kalshi, the market, the ML model, TheOver and sentiment, then applies bucket calibration (`core/streamlit_pipeline.py`).
- **Beyond game lines:** player props, DraftKings lineups and parlays.
- **Publishing and subscribers:** Netlify and cPanel publishing, and the paid subscriber site (Postgres). Paid launch is blocked; see `docs/paid-launch/paid-launch-owner-actions.md`.
- **Research automation, for all six sports:**
  - `research-scheduler.yml` runs every 30 minutes and captures prospective events and quotes within paid-API budgets.
  - It grades results and backs everything up to Drive.
  - It retrains the NBA, NCAAB and NHL score-distribution research models after each capture (`app_core/prospective_research_models.py`).
  - Separate workflows cover MLB receipt reconciliation, football stage-1 collection and daily activation grading.
- **Evaluation tools:** `core/clv.py`, `core/walk_forward.py`, `scripts/evaluate_walk_forward.py`, `app_core/pick_accuracy.py`, Strategy Lab, and `prospective_evidence.market_readiness`, which names each market's next blocker.

### Gaps found by the audit (verify against Drive evidence first)

| Gap | What the code shows | Effect |
|---|---|---|
| **G1. No certified-close producer** | Nothing outside `prospective_evidence.py` calls `insert_close`. Closing capture writes `closing_observations` in the separate prediction-evidence store (`app_core/activation_closing.py`) | Every plan needs ≥ 80% certified closes, so no market can pass |
| **G2. Plans are unbound** | Version-1 plans are `UNBOUND_REQUIRES_NEW_VERSION_BEFORE_VALIDATION`. Research models are trained, but nothing freezes a plan version that binds a model and calibration. Football v2 plans exist (`football_validation_v2.py`); their binding still needs to be checked | No market can enter validation |
| **G3. No deployment-review caller** | `record_deployment_review` has no caller. It is meant to be an independent review step | Expected; it becomes relevant once G1 and G2 are fixed |
| **G4. Readiness has not been read this session** | The canonical evidence lives in Drive, which this session could not reach | Each market's real next blocker is unknown |

### Built this session, and how it fits

| Work | Status | Fit with the chain |
|---|---|---|
| M0: CI repair and protected-change guard (#2455) | Merged | Keep. It unblocks all work |
| Shared pipeline store, operations, NFL adapter, orchestrator (#2456) | Merged; shadow-only, off by default (`PARLAYPICKER_PIPELINE_V2`) | **Duplicates** the prospective-evidence store and adapters. Freeze it: do not extend it, and decide in D10 whether to retire it |
| Scoreboard and CLV report on legacy exports (#2457) | Open PR | Partly duplicates `core/clv.py` and `core/walk_forward.py`. Useful as a view of the *legacy* picks. **Rework (C3)** to reuse `core/clv.py` and report progress against the frozen-plan thresholds |
| Closing capture in its own workflow (#2457) | Open PR | Keep. It stops the research job from starving it. It does not fix G1 |
| EV selector, shadow-only (`PARLAYPICKER_EV_SHADOW`, off) | Local branch | **Reworked.** Qualification now comes from `deployment_state` and `market_readiness`; the panel shows each market's next blocker. De-vig, EV and families reuse `core.clv`, `core.price_value` and `core.market_policy` |
| D1 thresholds (150 picks, CI > 0) | Local branch | **Withdrawn.** It duplicated the frozen plans with a looser rule |
| NBA ESPN model (D9) | Local branch | **Offline benchmark only** (`app_core/nba_espn_benchmark.py`). It lacks quote lineage, so the evidence store cannot accept it. Its artifact and shadow wiring were withdrawn before any push |

### Measured quality

- **Legacy picks:** 1,155 graded picks through 2026-08-28. Skill versus the de-vigged market is +0.001 (95% CI −0.006 to +0.008), and ROI is +0.3%. **No demonstrated edge** (`docs/evaluation/scoreboard-2026-11-05.md`).
- **Retrospective MLB ridge:** did not beat the scoring baseline (`docs/master-plan-progress.md`).
- **NFL score model:** needs QB and opponent features (`docs/nfl-model-validation.md`).

## 3. Milestones, in order

### C0: Read where every market stands (first; needs Drive; ~0.5 day)

- **What:** restore the canonical prospective evidence from Drive and run `prospective_evidence.all_market_readiness()`.
- **Output:** a table of the 12 markets with events, verified entry prices, verified closes, model, plan and next blocker. Commit it as `docs/evaluation/market-readiness-<date>.md`.
- **Needs** a session with Drive access (see `docs/roadmap/HANDOFF-2026-11-05.md`).
- **Effect:** confirms or corrects G1, G2 and G4, and sets the real priority among C1–C4.

### C1: Certified closes into the prospective store (fixes G1; ~3–5 days)

- **What:** for each captured prospective quote, record a certified close. That means the same event, market, selection, book and line, captured close to the start and replayable from the provider response, written with `prospective_evidence.insert_close`.
- **Source:** reuse the closing-capture workflow and its quota guard. Do not build a second capture.
- **Acceptance:** `market_readiness(...).verified_closes / unique_events ≥ 0.80` on a live slate for at least one market.
- **Cost:** may raise Odds API use. D8 (15-minute cadence) and the research budget cap apply.

### C2: Bind models to plans (fixes G2; ~2–4 days per sport family; owner review)

- **What:** for each market with a trained model (NBA, NCAAB and NHL research models first; then MLB; then football v2), freeze a **new plan version** bound to the exact model and calibration IDs.
- **Rule:** the plan must be frozen before its validation window starts, and the holdout must never be used for selection.
- **Owner role:** reviews each binding (decision D11).

### C3: One progress view against the frozen plans (~2–3 days)

- **What:** one Results panel showing, per market, progress toward each frozen-plan threshold: events, Brier, log loss, calibration error, coverage, close coverage and paper ROI.
- **How:** use `evaluate_validation_plan` and `market_readiness`.
- **Rework #2457's scoreboard into this:** keep the legacy-pick view as a separate, labelled section, and make it use `core/clv.py` and `core/walk_forward.py` instead of its own copies.

### C4: Validation, review, activation (as evidence accrues; owner)

- **Per market:**
  1. once validation and holdout both pass, create the validation artifact;
  2. an independent reviewer records the deployment review;
  3. the owner signs a market activation with an expiry and configures the exposure ledger.
- **Result:** funded picks appear on Today for that market only.
- **No code is needed for these steps.** Any gaps found while doing them are fixed then.

### C5: Better models, each with a fresh holdout (parallel; ongoing)

- **NFL:** QB availability and opponent-adjusted features (`docs/nfl-model-validation.md`).
- **MLB:** pregame starters (`docs/mlb-pitcher-development.md`).
- **NBA:** compare the ESPN benchmark (two seasons) with the research model's chronological diagnostics on the same games. Only a clear gain justifies proposing a longer history window, as a new plan version.
- **Injuries and sharp prices:** blocked on D4.
- **Rule:** every tuned model needs a new frozen plan version and a new evaluation window.

### C6: Selector switch (after C4 for at least one market)

- **Run the EV shadow** next to the current selector on live slates.
- **Compare** both on the same graded outcomes.
- **Switch** only on D7.

### Paid launch (separate track)

The 20 blockers in `docs/paid-launch/paid-launch-owner-actions.md` stay as they are. The first, `NO_COMMERCIALLY_ENABLED_QUALIFIED_MARKET`, is cleared by C4.

## 4. Owner decisions

| ID | Decision | Status |
|---|---|---|
| D1 | Qualification thresholds | **Superseded 2026-11-05 at owner direction ("rewrite and rework the local branch").** Qualification follows the frozen validation plans (§1). The 150-pick rule is withdrawn |
| D2 | NBA/NHL in parallel with NFL | **Moot.** The research scheduler already covers all six sports |
| D3 | Moneyline in scope; WNBA in the policy | Open. Moneyline stays context-only (`core/market_policy.py`); WNBA has no validation plan |
| D4 | Sharp-price and injury sources: rights, cost and quota | Open. Use existing Odds API/Novig first; no new paid plan without review |
| D5 | Budgets for authentic NFL acceptance | Open |
| D6 | Hosted NCAAF custody | Open |
| D7 | Switch the selector to EV with No Bet | Open. Only after C4 and a shadow comparison |
| D8 | Closing capture every 15 minutes instead of 30 | Open. Decide together with C1 |
| D9 | NBA ESPN download | **Owner-approved 2026-11-05; done.** The result is an offline benchmark |
| D10 | Retire or keep the shadow shared pipeline (#2456) | Open. Recommendation: keep it off and frozen; retire it once C3 lands |
| D11 | Who reviews plan bindings and deployment reviews | Open |

## 5. Guardrails (unchanged)

- **What gets saved:**
  - Original predictions are frozen before outcomes are read.
  - Grading is append-only.
  - Spent claims are never reset.
- **What the model may claim:**
  - No pick is fabricated to remove an Unavailable label.
  - Home-win probability never stands in for a cover or total probability.
- **What this roadmap does not authorize:** paid plans, provider traffic, live inference, production database writes, merges or deployments. Each needs the owner's explicit approval, case by case.
- **How results are described:** software completion ≠ predictive quality. Model-only results, private scenarios, ordinary Refresh acceptance and accuracy stay separately labelled.

## 6. Backlog

| ID | Task | Depends on | Status |
|---|---|---|---|
| M0 | CI repair and protected-change guard (#2455) | — | MERGED |
| M1 | Shared pipeline, shadow only (#2456) | — | MERGED; frozen (D10) |
| E1 | Legacy scoreboard and closing-capture workflow (#2457) | — | OPEN PR; scoreboard to be reworked under C3 |
| P1 | EV shadow selector on the authority chain | — | IMPLEMENTED (local); off by default |
| NBA-B | NBA ESPN offline benchmark | D9 | IMPLEMENTED (local); feeds C5 |
| C0 | Market readiness from Drive evidence | Drive session | NOT STARTED. **Next task** |
| C1 | Certified closes into the prospective store | C0 | NOT STARTED |
| C2 | Bind models to new plan versions | C0, D11 | NOT STARTED |
| C3 | Frozen-plan progress view; rework the scoreboard | C0 | NOT STARTED |
| C4 | Validation, review and activation | C1, C2 | NOT STARTED (owner steps) |
| C5 | Model improvements with fresh holdouts | C2 | NOT STARTED |
| C6 | Selector switch | C4, D7 | NOT STARTED |
| E1.1/E1.2 | Grade legacy picks since 2026-08-28; refit legacy calibration | Drive | BLOCKED (Drive). Lower priority than C0 |
