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
| **G1. No certified-close producer** (**confirmed 2026-11-05**) | Nothing outside `prospective_evidence.py` calls `insert_close`; `prospective_close` holds 0 rows for every sport. Closing capture writes `closing_observations` in the separate prediction-evidence store (`app_core/activation_closing.py`) | Every plan needs ≥ 80% certified closes, so no market can pass |
| **G2. Plans are unbound** (**confirmed 2026-11-05, including football v2**) | All 16 frozen plans (12 v1 and 4 football v2) have no model or calibration, and `prospective_prediction` holds 0 rows. Football has the training data (NCAAF 10,184 rows, NFL 3,331), but no model is registered | No market can enter validation |
| **G3. No deployment-review caller** | `record_deployment_review` has no caller. It is meant to be an independent review step | Expected; it becomes relevant once G1 and G2 are fixed **Fixed 2026-11-06 (D11):** the Deployment review workflow calls it after an audit |
| **G4. Readiness** (**read 2026-11-05**) | See `docs/evaluation/market-readiness-2026-11-05.md`. All canonical evidence is football; NBA, NCAAB and NHL have nothing canonical; MLB has reconciled facts only | Priority: C2 for NCAAF first, then C1 |
| **G5. Research sports never captured** (**found 2026-11-05**) | The NBA, NCAAB and NHL Drive stores hold 0 records: the repository variable `RESEARCH_SPORTS` is `MLB,NCAAF,NFL` | Owner: set `RESEARCH_SPORTS=NFL,NCAAF,NBA,NCAAB,MLB,NHL` |
| **G6. Scheduler runs time out** (**found 2026-11-05**) | Most active research runs end "cancelled" at the job limit, so capture rarely completes. A run spends about 27 of 48 minutes on 12 identical full-folder Drive listings (about 2 min each), and MLB receipts re-download about 800 MB each run because their cache lives in the temporary evidence folder | Opt-in listing reuse for evidence restores (15 min; lock reads stay fresh), and callers' caches moved to the kept workflow cache |
| **G7. Upload throughput** (**found 2026-11-05, run 299**) | The G6 listing reuse worked: 7 listings instead of 12, and restores take seconds. Each evidence record is still its own Drive object, uploaded at about 2.4 per second (41 seconds per 100). Run 299 uploaded 1,631 NCAAB records in 11 of its 49 minutes. A large NCAAB slate (~150 games, ~5,500 records) needs ~40 minutes, and a run killed mid-upload loses unsaved records | Research job limit raised from 60 to 150 minutes (public repository: free minutes). Bundling records into fewer Drive objects is a possible later change to the storage format |
| **G8. App picks: no CLV, broken headless slate** (**found 2026-11-05**) | (1) The app has saved 197 boards and graded 326 score revisions, but recorded **0 closing observations**. A close is saved only within 30 minutes of kickoff, and GitHub starts the 30-minute capture runs hours apart. (2) On GitHub runners, `dateutil.gettz("America/New_York")` gives no UTC offset for current dates (compact zoneinfo files), so `odds_api.iso8601_to_est` failed for every game. The headless pick run and `capture_live` therefore saw no live slate | `zoneinfo` for ET conversion. A **slate-loop closing capture** that checks every 5 minutes through each evening, plus weekend mornings. A **paper track record** (`core/track_record.py`) of the saved best picks, published by the daily Activation grading run: W-L-P, hit rate with a 95% interval, break-even, units, ROI and beat-close share by sport and market |
| **G9. Board probabilities: fake Kalshi and price edges** (**found 2026-11-05 from the owner's compact export**) | (1) A nearest-line Kalshi proxy up to 10.5 points away (NCAAB spreads) was only decayed toward 50%, so a -9.5 pick kept most of P(cover -7): the 14 positive-EV rows on the board (e.g. Iona -9.5, Kalshi 57.9% vs book 47.9%). (2) The edge column compared the paired-quote de-vig (WinProbability) with a re-derived de-vig against a synthetic opposite price that paired -108 with "+88", a fake +1.8 pts on every such row. (3) Gemini showed "skipped" because no row passed the deterministic checks; it was never called | (1) Proxies are shifted to the pick's line (normal approximation, per-league sd) before the decay. (2) The app recompute uses the real opposite quote; the synthetic price keeps a 20-cent line (-108 faces -112). streamlit_app.py change owner-approved and the NCAAF runtime rebound. All in #2466. "Line unresolved" rows stay listed at $0 by design (pinned by tests) |

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

### C0: Read where every market stands (first; ~0.5 day)

- **How to run it:** Actions → **Market readiness (read-only)** → **Run workflow** (`.github/workflows/market-readiness.yml`, `scripts/market_readiness_report.py`).
  - It restores the Drive evidence into an empty runner directory.
  - It reports `all_market_readiness()` for the 12 markets: events, verified entry prices, verified closes, model, plan, deployment state and next blocker.
  - Output: the run summary, plus the `market-readiness` artifact (JSON and Markdown).
- **Read-only:** the Drive client refuses every upload, and there are no Odds API or CFBD calls. It uses the research scheduler's Drive secrets.
- **Commit the result** as `docs/evaluation/market-readiness-<date>.md`.
- **Effect:** confirms or corrects G1, G2 and G4, and sets the real priority among C1–C4.
- **Also check: research runs are timing out.** The last two scheduled research runs (2026-11-04 21:15 and 2026-11-05 01:46 UTC) were cancelled at the job's 60-minute limit. Cut-short cycles slow every market's evidence.

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
| D4 | Sharp-price and injury sources: rights, cost and quota | **Owner-chosen 2026-11-06: option A, Pinnacle as the fair-price reference for NBA and NCAAB spreads; no new paid plan.** The capture request names the 11 US-region books plus Pinnacle (`bookmakers=`, 12 books = 2 region equivalents), so one raw response holds every book and each quote still replays. A request costs 4 credits instead of 2; the extra 2 come from their own allowance (60/day, 1,500 per 31 days, `PINNACLE_LIMITS`), and capture falls back to `regions=us` when it is spent. Pinnacle is reference-only (`REFERENCE_BOOKS`): never a plan's entry quote, a research pick, a selector price or a certified-close request. The price screen reports EV against Pinnacle's no-vig price next to the cross-book median. Off until the repository variable `PARLAYPICKER_PINNACLE_SPORTS=NBA,NCAAB` is set. Options B (NBA official injury reports), C (NCAAB ratings) and D (paid feed) deferred **NCAAF added 2026-11-06 (owner option B+).** `ncaaf_prospective.py` requests the same 12 books when `PARLAYPICKER_PINNACLE_SPORTS` names NCAAF (3 markets x 2 region equivalents = 6 credits; the extra 3 from the Pinnacle allowance) and `quote_candidates` never selects Pinnacle. The frozen NCAAF cohort carries over through a reviewed runtime transition (`PINNACLE_RUNTIME`); the new frozen `docs/paid-launch/ncaaf-compatibility-binding-v2.json` accepts the reviewed bytes (v1 unchanged); the NCAAF runtime is rebound. Model, artifact, lineage and aliases unchanged Option B (NBA official injury reports) is next **if** the Pinnacle numbers show no pricing edge (owner-agreed 2026-11-06). The NCAAF capture now also saves Pinnacle's quotes per game as `reference_quotes` (the first B+ build requested them but did not keep them); compatibility binding v3, reviewed transition to `REFERENCE_RUNTIME`. **Pinnacle as the app's market anchor, in shadow (owner 2026-11-06):** the app's odds request names Pinnacle as a fifth book (still under 10 books, so no extra region cost); Pinnacle lines are reference-only `pinnacle_*` fields, never in the bettable book lists; `core/pinnacle_anchor.py` gives Pinnacle's no-vig probability at the pick's exact line from a quote at most 20 minutes old, saved with each pick as `pinnacle_fair_probability`. The blend-weight study compares book vs Pinnacle anchors once 100 settled picks have both. Switching the live anchor is an owner decision |
| D5 | Budgets for authentic NFL acceptance | Open |
| D6 | Hosted NCAAF custody | Open |
| D7 | Switch the selector to EV with No Bet | Open. Only after C4 and a shadow comparison. **Shadow comparison built 2026-11-06** (`app_core/selector_shadow_compare.py`, in the Research picks report): on the same replayed NBA/NCAAB games and model, today's rule (bet the best-EV side when EV >= 0) vs the EV selector (market nudged toward the model by at most 5 points, bet only with a 2-point edge), each graded on results and CLV against the last captured pregame consensus **Blend-weight study 2026-11-06** (`core/blend_weights.py`, in the activation grading report): the app's hand-set blend (Kalshi 40% only when >= 55%) refitted as a market-anchored log-odds model, logit(p) = logit(market) + sum w x (logit(input) - logit(market)), weights by L2 logistic regression with bootstrap 90% intervals, scored on later picks against the market alone and the current blend. Kalshi and ML only (TheOver and sentiment are not saved with picks). Any switch is an owner decision after the study and a shadow comparison |
| D8 | Closing capture every 15 minutes instead of 30 | Open. Decide together with C1 **Owner 2026-11-06: wait** for the morning report's close-capture results. |
| D9 | NBA ESPN download | **Owner-approved 2026-11-05; done.** The result is an offline benchmark |
| D10 | Retire or keep the shadow shared pipeline (#2456) | Open. Recommendation: keep it off and frozen; retire it once C3 lands **Owner-agreed 2026-11-06: retire** the frozen shadow pipeline now that C3 has landed (separate change; it touches runtime-bound files). |
| D11 | Who reviews plan bindings and deployment reviews | **Owner-chosen 2026-11-06: option B, the owner plus an automated audit.** Plan bindings: the owner freezes a reviewed draft hash (Bound plan workflow). Deployment reviews: the "Deployment review" workflow first runs `app_core/deployment_audit.py` read-only on a passed plan (FAIL: evaluation not passing, chronology violations, reference-book entry prices; WARN: duplicate predictions, profit concentrated in the top 5 wins, holdout Brier drift > 0.01, negative CLV) and publishes the audit hash. The owner's `record` step recomputes it, refuses on a hash mismatch or any FAIL, then writes the validation artifact and the deployment review with the reviewer bound to the audit hash. Activation stays a separate owner step. Can later add a reviewer-is-not-activator rule (option C) |
| D12 | Which plan family an NCAAF/NFL model targets: v1 rules (200 events), v2 (9,451 decided events per market, unreachable in one season), or a new v3 | **Owner-chosen 2026-11-05: C, design a v3.** First, run Football Stage 2 (owner-approved, after #2460 merges). Then draft a v3 proposal whose minimum sample is derived from Stage 2's measured variance, push rate and dependence. The proposal is reviewed before anything is frozen. **Update 2026-11-05:** Stage 2 ran (`docs/evaluation/football-stage2-2026-11-05.md`). No scope can train (NCAAF 156 of about 380 independent games; NFL 47; two seasons required, all data from 2026), so v3 is deferred until the first trained model gives measured variance **Owner 2026-11-06: NCAAF picks this season.** Qualified (staked) NCAAF picks are not reachable this season under any plan family: validation and holdout each need 200 settled games and about 260 remain (4 Saturdays plus bowls). This season NCAAF gets research-grade picks from the NCAAF Pinnacle value board (`app_core/ncaaf_pinnacle_value.py`, in the Research picks report); the plan family is decided before 2027. |
| D13 | Odds API credits for certified closes (plan quota 20,000/month; CFBD/CBBD 30,000/month) | **Owner-chosen 2026-11-05: option B, NBA spread only.** A separate allowance of 100 credits/day and 2,500 per 31 days (`CERTIFIED_CLOSE_LIMITS`), about 80 credits/day used. Enabled when the repository variable `PARLAYPICKER_CERTIFIED_CLOSE_SPORTS=NBA` is set. The capture allowance (7,500 per 31 days) is unchanged |
| D14 | Plan windows and the first research market | **Owner-chosen 2026-11-05: windows are sized by game count, and NCAAB spread goes first.** Each validation and holdout window must reach the plan's 200 independent settled games at the observed games-per-day rate, with 80% coverage and a 1.25 margin (`prospective_calibration.window_days`), instead of v1's year-long windows. NCAAB (25–40 games/day) needs about 11 days per window, against about 45 for NBA. NCAAB certified closes have their own allowance (400/day, 4,500 per 31 days, about 400 games in total). Enable it by setting `PARLAYPICKER_CERTIFIED_CLOSE_SPORTS=NBA,NCAAB` **Owner-set 2026-11-06: `PARLAYPICKER_CERTIFIED_CLOSE_SPORTS=NBA,NCAAB`.** |
| D15 | NCAAF runtime after alias-only edits | **Owner-approved and merged 2026-11-05 (#2462); NCAAF capture works again (run 299).** NCAAF capture has failed with `stale_frozen_model` since 2026-10-03 because 93a7f236 (NCAAF identity aliases) and 1ccf0a28 (NHL mapper names) changed files inside the NCAAF runtime hash. The model, feature and history modules are byte-identical. The reviewed recovery accepts exactly d5acdd2c/bd837b0b → 8283b60a; the model stays research-only. See `docs/evaluation/drive-sources-2026-11-05.md` |
| D16 | Research capture window | **Owner-set 2026-11-05: `PARLAYPICKER_RESEARCH_CAPTURE_HOURS=18` (#2462).** NBA/NCAAB/NHL prices were captured only for games starting within 2 hours of a run, and GitHub starts about 4 runs a day, so most evening tip-offs were missed. The frozen plans require only strictly pregame prices. `PARLAYPICKER_RESEARCH_CAPTURE_HOURS` (default 2, range 2–24) sets the window. A run now captures up to 3 batches of 24 events (2 credits each). The certified-close lookback is 24 hours. Recommended: 18. The trade-off is that entry prices are earlier, so they sit further from the close |
| D17 | NHL settlement | **Owner-set 2026-11-05: `PARLAYPICKER_NHL_SETTLEMENT=official_check` (#2462).** The Odds API does not document how it reports shootout games. US books settle NHL puck lines and totals on the official final, including overtime and the shootout goal, which is the existing `nhl_puck_line_evidence` target. The NHL's free official feed (api-web.nhle.com) reports exactly that (e.g. EDM 3 @ NJD 4, `lastPeriodType: SO`, 2026-11-03). An NHL game settles only when the provider final **equals** the official final, and the official game ID, period type and hash are recorded in the settlement. A mismatch is never settled. Enable with `PARLAYPICKER_NHL_SETTLEMENT=official_check` |
| D18 | NCAAF model trained on history | **Owner-approved 2026-11-05. Step 1 done: no edge.** The canonical store accepts only games discovered before kickoff, so history is not loaded into it. The historically trained model (`ncaaf_research`, artifact 9334f2a4…, confirmed by rebuild) was backtested on 614 2025 games against archived lines. No variant beats the 52.4% break-even beyond noise, and every Brier score is worse than a coin flip (`docs/evaluation/ncaaf-line-backtest-2026-11-05.md`). Step 2 (registration and a v3 plan) is not recommended. Next: C5, a market-anchored NCAAF model screened by the same backtest |

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
| E1 | Legacy scoreboard and closing-capture workflow (#2457) | — | MERGED; scoreboard to be reworked under C3 |
| P1 | EV shadow selector on the authority chain | — | IMPLEMENTED (local); off by default |
| NBA-B | NBA ESPN offline benchmark | D9 | IMPLEMENTED (local); feeds C5 |
| C0 | Market readiness from Drive evidence | — | **DONE 2026-11-05** (`docs/evaluation/market-readiness-2026-11-05.md`). No market is in validation. **Next: C2 for NCAAF** |
| C1 | Certified closes into the prospective store | C0 | **DONE** (#2461). NBA and NCAAB spread allowances (D13, D14). Turn on with `PARLAYPICKER_CERTIFIED_CLOSE_SPORTS` |
| C2 | Bind models to new plan versions | C0, D11, D12 | **BUILT** (#2462). Settlement, the pinned candidate, calibration, bound predictions and the reviewed freeze (Bound plan workflow). It fills in as NBA, NCAAB and NHL games are captured (run 299 onward). Football waits for a better model (D18) |
| C3 | Frozen-plan progress view; rework the scoreboard | C0 | **DONE** (#2463). The Qualification progress panel loads the daily readiness report with a read-only token. A paper track record of the app's saved picks is published by the daily Activation grading run (#2465). The research picks board loads in the same panel (#2466) |
| C4 | Validation, review and activation | C1, C2 | NOT STARTED (owner steps) |
| C5 | Model improvements with fresh holdouts | C2 | IN PROGRESS. The NCAAF score model shows no edge over archived lines (D18). The first C5 piece is a **cross-book price screen** (`app_core/price_edge_screen.py`, shown in the readiness report). It de-vigs every book, takes the median as fair, flags the best price with EV at or above a threshold, and grades each flag against its certified close (CLV) and the final score. It uses no model. It shows whether price shopping beats the close before any selector is bound to a plan. **Research picks** (`app_core/research_picks.py`, "Research picks" workflow, Qualification panel): the app's pipeline has no spread/total model for NBA/NCAAB/NHL (its ML model targets the home win), so these come from the in-season research score models. Each pick is the best-EV side and line at the best available price, shown with the cross-book consensus. It is graded by replaying each finished game as of its first captured price **NCAAB line screen** (`app_core/ncaab_line_backtest.py`, "NCAAB line backtest" workflow): a market-anchored model on archived CollegeBasketballData lines (team form as of each game plus the line), fit on 2023-24, calibrated on the first half of 2024-25 and scored once on its second half. It answers early, before the in-season NCAAB models reach 60 games, whether form beats the line **First NCAAB screen (2026-11-06): no edge.** Spread 52.3% on 480 picks at P >= 52% (-0.2% ROI), 49.4% at 55%; totals below break-even; Brier 0.250; the model sits within ~0.9 pts of the line. That run saw only the first 3,000 games per season (the API's row cap). **Full-season re-run (2026-11-06, 6,118 / 6,117 games, 5,196 / 5,309 with a line): no edge, confirmed.** Scored on 2,231 second-half 2024-25 games: spread 50.0% on 1,075 picks at P >= 52% (-4.5% ROI), 48.0% on 173 at 55%; totals 49.9% on 607 at 52%; Brier 0.251 / 0.250; the model sits within ~0.6 pts of the line. Team form alone does not beat the line in NCAAF or NCAAB, which points to new information sources (D4) rather than more tuning. **Pinnacle value board** (`app_core/pinnacle_value.py`, in the Research picks report and panel): no model; US-book prices that beat Pinnacle's no-vig line by 2%+ at the same line, one pick per game, graded on results and on CLV against Pinnacle's last pregame no-vig line |
| C6 | Selector switch | C4, D7 | **PLANNED 2026-11-06** (`docs/roadmap/C6-PLAN-2026-11-06.md`): a bound-prediction bridge puts the qualified market's own predictions into the app pipeline as best-pick rows carrying every field `market_gate` checks, so they pass the existing wager contract, Gemini review and stake sizing. Build by early December. **Step 2 built 2026-11-06** (`app_core/qualified_bridge.py`): its rows pass every row-level `market_gate` check against a synthetic validated deployment, leaving only the owner's activation. **Step 3 built 2026-11-06**: `run_analysis_pipeline` replaces each qualified market's app rows with bridge rows before eligibility and Gemini (off unless `PARLAYPICKER_QUALIFIED_BRIDGE=true`; an error leaves rows unchanged). Remaining: near-kickoff fresh bound predictions (step 1), so quotes are at most 30 minutes old |
| E1.1/E1.2 | Grade legacy picks since 2026-08-28; refit legacy calibration | Drive | BLOCKED (Drive). Lower priority than C0 |
