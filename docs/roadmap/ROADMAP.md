# ParlayPicker roadmap: the best pick for each game

Version 2, 2026-11-05. This supersedes the 2026-11-01 brief's sequencing and keeps
its guardrails. The baseline below was verified against the repository at `main`
`db3e22c` (after PR #2455). Each task has a stable ID, and its status is tracked
in the backlog at the end of this file.

## 1. What "best pick" means

For every eligible game, the app shows **one pick or an explicit No Bet**:

1. **Candidates.** The game's supported markets: full-game side (spread / run line /
   puck line), total, and moneyline once that market is qualified.
2. **Fair probability.** For each candidate, a calibrated probability that it wins
   (pushes handled natively), at the exact signed line and price that is offered.
3. **Edge.** That probability minus the de-vigged *market* probability for the
   same line, and the expected value (EV) at the offered price.
4. **The pick.** The candidate with the highest EV, provided that the edge exceeds
   a per-league/market threshold and the model is *qualified* for that market
   (§4). Otherwise the result is **No Bet**, with the specific reason: unsupported
   market, model not qualified, stale quote, missing inputs, or edge too small.

A pick is only "best" if it is measured to be good. We track three numbers, per
league and market, on games graded *after* the prediction was frozen:

| Measure | What it shows | Target before calling a market qualified |
|---|---|---|
| **Closing-line value (CLV)** | Did our price beat the closing price? This is the fastest reliable signal of a real edge. | Positive mean CLV, with a confidence interval above 0, over ≥ 150 picks |
| **Log loss / Brier vs. market** | Are our probabilities better than the de-vigged closing market? | Model log loss < market log loss on an untouched holdout |
| **ROI at the offered price** | Did it make money? This is noisy, so it is reported but not used alone. | Reported with a confidence interval, and never inferred from one slate |

These thresholds are proposals. The owner sets them before evaluation starts
(decision D1).

## 2. Where we are (verified 2026-11-05)

**Operational status.**
- **M0 is complete.** PR #2455 is merged with all 25 CI checks green. The protected-change
  guard is re-baselined. Test shards take 3–15 minutes (they took 35–49).
- **Ordinary Refresh already runs a model for NFL, MLB and WNBA spreads and totals**
  (`score-distribution-v1`, in `app_core/market_probability_model.py`).
  - NFL features come from point-in-time `nfl_data_py` schedules: only games
    before the slate date are used (`app_core/feature_processing.py:1791`).
  - Rows are blended with market, Kalshi and TheOver inputs
    (`compute_blended_probability`).
  - Each game's highest `calibrated_probability` is selected (`probability-first-v1`).
- **The Guided private NFL workflow calls the same model function.** It adds an
  owner-reviewed, authenticated scoring history, one-use claims and a signed
  Postgres store (`app_core/nfl_workflow_store.py`). It is spreads-only and
  never stakes.
- **Private NFL routing is fragmented.** Eight packet versions
  (`nfl_private_routes.py`, v1–v8, about 2,400 lines) each re-implement the same
  admission gates, one per source format.
- **NCAAF private Refresh** bypasses the ordinary pipeline. It is spreads-only and
  stores results in local SQLite.
- **Durable records are split across at least 15 SQLite/Postgres stores.**

**Predictive status.** This is the gap that matters most.
- **No demonstrated edge.** The only recorded held-out test of the calibrated
  probability (231 graded picks, train ≤ Aug 16) scored **Brier 0.249 and log loss
  0.691**; a coin flip scores 0.250 and 0.693. 226 of the 231 picks were predicted
  at 50–60%.
- **Calibration is stale.** It was last refit on 2026-08-29 from 1,155 graded
  picks. Nothing has been graded into it since 2026-08-28.
- **The NFL model is a fixed formula.** Its parameters are hand-set (0.45
  reliability shrink, 0.20–0.80 clip), and it has no injury, quarterback, rest,
  travel or weather inputs.
- **Model coverage by league:**
  - NBA, NHL and NCAAB have **no production model**. Research ridge code exists
    in `prospective_research_models.py`, and NHL has a puck-line research route.
  - WNBA has model parameters but is **not in the sport/market policy**, so it
    cannot be activated.
  - **No league supports moneyline** as a production market. The only moneyline
    signal is the generic home-win XGBoost model.
- **Measurement tooling exists but isn't run routinely:** a walk-forward evaluator,
  calibration refresh and grading scripts (`scripts/evaluate_walk_forward.py`,
  `refresh_calibration.py`, `grade_slate.py`).

**Consequence:** adding leagues does not produce better picks. The critical path is
**measure → beat the market in one league → expand**.

## 3. Tracks

There are three parallel tracks. Neither status implies the other.

- **E: Evaluation loop.** Grade everything, refresh calibration, and publish the
  scoreboard. *Start now; everything else depends on it.*
- **P: Predictive quality.** Benchmark against the market, then improve the models.
- **M: Operational pipeline.** One shared Refresh path, with durable records and
  reopening.

## 4. Milestones

### E1 — Restart grading and the scoreboard (start immediately; ~3–5 days)

- **E1.1** Grade every pick saved since 2026-08-28 from authoritative final scores
  (existing grading scripts). Keep the original predictions as they were; grading
  is append-only.
- **E1.2** Refit calibration with `refresh_calibration.py`. Promote the new fit
  only if it beats the current one on a later holdout, and keep both versions.
- **E1.3** Build the **scoreboard**: per league × market, report count, coverage,
  Brier, log loss, calibration bins, CLV and ROI with confidence intervals.
  Compare against two benchmarks: the de-vigged market and a constant 50%.
  Display it in Streamlit and export it. It must never alter a stored pick.
- **E1.4** Capture closing lines automatically for every pick, so CLV is
  available. The NFL market tracker already snapshots prices; extend it to every
  graded market.

*Exit:* the scoreboard runs on demand and after each grading pass. Every number
has a sample count and a confidence interval.

### P1 — Market-anchored baseline (~3–5 days; after E1.3)

- **P1.1** Make the de-vigged market probability the explicit prior, with an
  explicit vig-removal method per book. The model contributes a bounded
  *adjustment*, not an independent probability averaged in. This stops a weak
  model from overriding a strong market.
- **P1.2** Change pick selection from "highest probability" to "highest EV, only
  when edge > threshold and the market is qualified". Otherwise return No Bet
  with a reason. In shadow mode, the current and new selectors both record their
  picks and the scoreboard compares them. The switch happens only by decision.
- **P1.3** Add a sharp-price reference (Novig, or another approved sharp book) as
  the market benchmark where licensing and budget allow (decision D4).

*Exit:* every game shows its edge vs. market and a pick or No Bet with a reason;
the shadow comparison is running.

### M1 — Shared pipeline contract (~8–12 days)

- **M1.1** Generalize `nfl_workflow_store.Store` into a sport-neutral
  `app_core/pipeline/store.py`. Keep its signed records, owner scope, one-use
  claims, and Postgres/SQLite backends. Existing NFL records stay readable,
  unchanged, through an adapter.
- **M1.2** Build the **operation state machine**: ready → awaiting-review →
  running → completed / failed / expired / cancelled. Provider requests and
  calculations get separate claims. A rerender, duplicate click, logout or
  restart never repeats a provider call.
- **M1.3** Define the **SportAdapter** interface. Each adapter provides:
  schedule/identity, history and features (as-of cutoff), the model call,
  supported markets with settlement rules, and grading.
  - NFL is the first adapter.
  - The eight NFL packet versions collapse into one versioned *source* contract:
    the format differs, the admission gates are shared.
  - Old packets stay readable through compatibility readers.
- **M1.4** Ordinary Refresh calls the orchestrator. Streamlit shows reasons for
  any game that can't proceed, plus quote ages and request costs.

*Exit:* the same orchestrator handles two different synthetic games, a mixed batch,
failures, stale data, duplicate clicks and restart, with no game-specific code
edits and no hidden network calls.

### M2 — NFL end-to-end through ordinary Refresh (~5–8 days; after M1, E1)

- **M2.1** Run the NFL adapter in ordinary Refresh: schedule → mapping → point-in-time
  features → model → spread/total probabilities → edge/EV → pick or No Bet →
  durable record → display → reopen after restart → grade.
- **M2.2** Authentic acceptance on ≥ 2 eligible future games, within approved
  budgets (decision D5). Results must reopen after a fresh session and after a
  restart, with no recalculation.

*Exit:* the first league gets real picks through the product, graded automatically
into the scoreboard. That establishes operational acceptance only, not predictive
qualification.

### P2 — NFL model that can beat the market (~10–15 days; parallel to M1/M2)

- **P2.1** Historical dataset from nflverse (2015–2025 seasons): play-by-play
  EPA/success rate, quarterback identity, rest/travel, weather, and closing lines.
  Only data available before kickoff is used.
- **P2.2** Fit margin and total models (ridge or gradient-boosted residual models
  on top of the market line), with walk-forward validation by season.
- **P2.3** Injury/availability inputs, only from an authenticated source with an
  observation time; missing data is disclosed, never invented (decision D4).
- **P2.4** Promotion gate: beat the market's log loss on an untouched holdout
  season, plus positive simulated CLV. The production version stays frozen during
  each evaluation period.

*Exit:* NFL spread/total either qualify under §1's targets or stay
"research only" with the evidence shown. Both outcomes count as done.

### M3 — NCAAF on the shared path (~6–10 days)

Port the existing NCAAF history, model and review contracts to an NCAAF adapter.
Neutral-site handling, lags and cutoffs stay as they are. Resolve hosted custody
(decision D6). Then apply the same scoreboard and the same P-track gates.

### M4 — NBA and NHL (~15–25 days combined)

These leagues have large, fast-accumulating samples: 1,230 NBA and 1,312 NHL
regular-season games. **They are the quickest route to a statistically meaningful
evaluation** (decision D2).

- **M4.1** NBA adapter, plus a model built from `prospective_research_models`:
  rest/back-to-back, availability, pace. Spread/total first.
- **M4.2** NHL adapter, starting from the puck-line route. Regulation vs.
  including-OT/shootout rules are explicit. Goalie input only with an
  authenticated observation clock. Totals need their own target.

### M5 — NCAAB, WNBA, MLB (~15–25 days; in season order)

- **NCAAB:** CBBD identity bridge, neutral sites, season-year convention.
- **WNBA:** add it to the sport/market policy first (decision D3). Do not reuse
  NBA probabilities.
- **MLB:** listed pitchers, doubleheaders and run line. MLB stays frozen until
  the owner's sequence reaches it.

### P3 — Moneyline (per league, after that league's spread/total qualifies; ~2–4 days each)

Derive P(win) from the same score-distribution model, so that sides and moneyline
stay consistent; overtime/shootout rules are explicit. It has its own scoreboard
row and its own promotion gate, and it needs scope approval (decision D3).

### M6 — Routine operation (ongoing; finish after M2)

- Scheduled grading and calibration refresh.
- Quota accounting.
- Backup and restore drills.
- Owner guide.
- Parlays/props stay later scope.

## 5. Critical path and staged delivery

```
E1 (grading + scoreboard) ──► P1 (market anchor, EV/No-Bet) ──┐
M1 (shared contract) ──► M2 (NFL in Refresh) ──────────────────┼─► first real NFL picks under measurement
P2 (NFL model) ──────────────────────────────────────────────────┘
then M3 NCAAF → M4 NBA/NHL → M5 → P3 per league
```

**Realistic staging** (engineering days, assuming one engineer plus Codex/Claude,
with owner reviews within ~1 day):

| Stage | Done by | What exists then |
|---|---|---|
| E1 + P1 | about 1–2 weeks | Honest scoreboard. Every game shows an edge vs. market, and No Bet is explicit. |
| M1 + M2 | about 3–5 weeks | NFL picks run through ordinary Refresh, stored and graded automatically. |
| P2 | about 4–6 weeks, parallel | Fitted NFL model. Qualification depends on evidence, not on the date. |
| M3, M4 | about 6–12 weeks | NCAAF, then NBA/NHL on the same path. NBA/NHL evaluate fastest. |
| M5, P3 | about 12–20 weeks | Remaining leagues and moneyline. |

The NFL regular season ends in early January, so only about 9 weeks of prospective
NFL games remain. That argues for starting NBA/NHL model work (M4) in parallel
(decision D2).

**Predictive qualification cannot be scheduled.** It depends on sample size and
whether an edge actually exists. The honest outcome for some league/markets will
be "No Bet; no demonstrated edge".

## 6. Owner decisions needed

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Qualification thresholds (CLV, log loss vs. market, minimum sample) | **Owner-approved 2026-11-05: the §1 targets.** ≥ 150 graded picks, skill vs. market CI above 0, ≥ 150 closes with mean line CLV CI above 0. Encoded in `app_core/selection/qualification.py` |
| D2 | Start NBA/NHL in parallel with NFL, given the NFL season length | **Owner-approved 2026-11-05: yes.** M4.1 (NBA) started after M1.3 |
| D3 | Moneyline in scope; add WNBA to the policy | Moneyline yes, after spread/total; WNBA next season |
| D4 | Sharp-price and injury sources: rights, cost and quota | Use existing Odds API/Novig first; no new paid plan without review |
| D5 | Budgets for authentic NFL acceptance (M2.2) | ≤ 2 games, as the existing prepared inventory allows |
| D6 | Hosted NCAAF custody (Neon vs. local-only first release) | Local-only first, labelled clearly |
| D7 | Switch selector from highest probability to highest EV with No Bet | After 2+ weeks of shadow comparison |
| D8 | Closing capture every 15 minutes instead of 30. It catches more closes, and costs Odds API credits only when picks are within 30 minutes of kickoff | Yes, once the 30-minute run is confirmed reliable |
| D9 | Download NBA game history (2024-25 and 2025-26 regular seasons, plus the current season to date) via the free public `nba_api` endpoint, to train the NBA model. No Odds API credits | **Owner-approved 2026-11-05.** One bounded download; review the holdout report before committing the artifact |

## 7. Guardrails (unchanged from the 2026-11-01 brief)

- **What gets saved:**
  - Original predictions are frozen before outcomes are read.
  - Grading is append-only.
  - Spent claims are never reset.
- **What the model may claim:**
  - No pick is fabricated to remove an Unavailable label.
  - Home-win probability never stands in for a cover or total probability.
- **What this roadmap does not authorize:** paid plans, provider traffic, live
  inference, production database writes, merges or deployments. Those need the
  owner's explicit approval, case by case.
- **How results are described:** software completion ≠ predictive quality.
  Model-only results, private scenarios, ordinary Refresh acceptance and accuracy
  stay separately labelled.

## 8. Backlog

Statuses: NOT STARTED, IN PROGRESS, BLOCKED, IMPLEMENTED, RELEASE-QUALIFIED,
AUTHENTICALLY ACCEPTED. Deployment and model quality are tracked separately.

| ID | Task | Depends on | Status | Effort |
|---|---|---|---|---|
| M0 | Release repair (#2455) | — | RELEASE-QUALIFIED (merged `db3e22c`) | done |
| E1.1 | Grade picks since 2026-08-28 | owner exports | BLOCKED: post-Aug-28 exports/recaps live in Drive, not the repo. Upload them in Results → scoreboard, or hand them over | 1 d |
| E1.2 | Refresh + holdout-validate calibration | E1.1 | BLOCKED (E1.1) | 0.5–1 d |
| E1.3 | Scoreboard (Brier, log loss, CLV, ROI vs. market) in Streamlit | — | IMPLEMENTED (`app_core/evaluation/scoreboard.py`, `scripts/scoreboard.py`, Results-tab panel). Baseline: `docs/evaluation/scoreboard-2026-11-05.md`, no demonstrated edge | done |
| E1.4 | Automatic closing-line capture | — | IMPLEMENTED. Capture moved to its own workflow (`closing-capture.yml`): it was starved by the hour-long research job in a shared queue. CLV report in `app_core/evaluation/clv.py`; 15-minute cadence is D8 | done |
| P1.1 | Market-anchored probability | E1.3 | IMPLEMENTED (`app_core/selection/ev_selector.py`): two-way de-vig; the independent model nudges the market by at most ±5 points (weight 0.25 in log-odds) | done |
| P1.2 | EV/No Bet selector in shadow mode | P1.1 | IMPLEMENTED (shadow behind `PARLAYPICKER_EV_SHADOW`, off; decisions recorded to the pipeline store when configured). Grading shadow vs. current picks is next | done |
| P1.3 | Sharp-price benchmark | D4 | BLOCKED (D4) | 1–2 d |
| M1.1 | Sport-neutral signed store (`app_core/pipeline/store.py`) | — | IMPLEMENTED (SQLite + real-Postgres tests) | done |
| M1.2 | Operation state machine (`app_core/pipeline/operations.py`) | M1.1 | IMPLEMENTED (lifecycle, claims, expiry, crash, batch tests) | done |
| M1.3 | SportAdapter + NFL adapter (`app_core/pipeline/adapter.py`, `nfl.py`) | M1.2 | IMPLEMENTED (feature parity with `fetch_nfl_stats`). Physical merge of the 8 legacy packet modules is deferred: old packets keep their unchanged reader, and new sources use the adapter | done |
| M1.4 | Ordinary Refresh calls the orchestrator (`app_core/pipeline/orchestrator.py`) | M1.3 | IMPLEMENTED in shadow mode behind `PARLAYPICKER_PIPELINE_V2` (off). Switching selection to it is P1.2/D7 | done |
| M2.1 | NFL end-to-end in Refresh | M1.4, E1 | NOT STARTED | 3–5 d |
| M2.2 | Authentic NFL acceptance (2 games) | M2.1, D5 | BLOCKED (D5) | 1–2 d |
| P2.1–P2.4 | NFL fitted model + promotion gate | E1.3 | NOT STARTED | 10–15 d |
| M3 | NCAAF adapter | M1.3, D6 | NOT STARTED | 6–10 d |
| M4.1 | NBA adapter + score model (`app_core/pipeline/nba.py`, `nba_history.py`, `scripts/train_nba_model.py`) | M1.3, D2, D9 | IMPLEMENTED (research-only): trained on 2,525 ESPN games (holdout margin RMSE 15.55 vs 17.56 naive; no market comparison yet). The artifact `models/nba-score-ridge-v1.json` is committed as `RESEARCH_ONLY`. NBA runs in the shadow only with `PARLAYPICKER_PIPELINE_NBA=true`; it is never registered as a supported sport, and only D1 qualification can make an NBA market a pick | 8–12 d |
| M4.2 | NHL adapter + model | M1.3, D2 | NOT STARTED | 7–13 d |
| M5 | NCAAB, WNBA, MLB | M1.3, D3 | NOT STARTED | 15–25 d |
| P3 | Moneyline per league | league qualified, D3 | NOT STARTED | 2–4 d each |

**Next executable task:** E1.1, grading saved picks since 2026-08-28. It can run
offline against stored results, using no provider traffic beyond already-approved
score fetches. M1.1 can proceed in parallel.
