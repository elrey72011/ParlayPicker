# Getting picks validated: audit and implementation roadmap (2026-11-09)

Audit of `main` `55c31456`, the readiness report of 1:47 PM ET (run 42677113016), two read-only
plan drafts at 2:17 PM ET (runs 42680978167 and 42680985674), a go-live rehearsal on a synthetic
store, and an inventory of every edge source in the repository. It extends
`docs/roadmap/ROADMAP.md` (version 3); the authority chain and guardrails there are unchanged.

## 1. Verdict

No pick can be validated today, for three separate reasons. They are listed in the order they bite.

1. **No strategy produces bets.** The only bound plan (NCAAB spread, Pinnacle model, v2) has 2 bets
   in 14 games. Today's replay of the same idea with Novig as an entry book and first-edge entry
   found **0 bets at 1% and at 2%** in 53 NCAAB games (3.9 days) and 0 in 15 NBA games. A plan needs
   50 bets in each of two windows.
2. **The plan gates rarely pass a strategy that does work.** In a simulation of the current gates, a
   model with a true 2% edge passes both windows 16% to 38% of the time at the plan's 200 games per
   window (section 3).
3. **A market that passes still stakes $0.** The rehearsal took a synthetic passing plan through
   validation, audit, review, activation and the market gate without a problem, then lost the pick
   between the bridge and the stake at three code defects (section 4).

The infrastructure up to the review is sound. What is missing is an edge with volume, a plan that
can recognise it, and three fixes at the end of the chain.

## 2. Where each market stands

| Market | Stored games | Settled predictions | Certified closes | Next blocker |
|---|---|---|---|---|
| NCAAB SPREAD | 213 | 72 | 1,738 | Plan v2 in validation: 14 games, 2 bets, 21 days left |
| NCAAB TOTAL | 213 | 50 | 478 | No candidate model (needs 120 training games; has 0) |
| NBA SPREAD | 33 | 15 | 238 | No candidate model; Pinnacle model has no calibration yet |
| NBA TOTAL | 33 | 9 | 108 | Same |
| NHL PUCK_LINE / TOTAL | 35 | 0 | 0 | Score settlement not enabled; no model |
| NFL, NCAAF, MLB | 0 in the research store | — | — | No prospective events (football capture paused, D25; MLB out of season) |

The store holds 0 validation artifacts and 0 deployment reviews. The research score models have 0
training games because they need two earlier games per team and the seasons are a week old.

Other measurements that shape the plan:

- **Pinnacle coverage.** Pinnacle prices 30% of stored NCAAB games (53 of 179) and 60% of NBA games.
- **Price screen** (every candidate at the best price): mean EV −2.6% and ROI about −2.6% for NBA and
  NCAAB spreads; 5 of 552 NCAAB spread candidates and 1 of 100 NBA reach 2% EV; totals have none.
- **Close coverage falls with entry age.** NCAAB spread entries get a same-book, same-line certified
  close 90% of the time at 30–60 minutes before the start, 69% at 1–3 hours, 54% at 3–6 hours and
  42% earlier. The plan needs 80%. Research capture prices a game once, hours ahead.
- **Plan v2** is expected to fail: its calibration was fitted before the guard (slope −13.2) and it
  entered before Novig was captured.

## 3. The gates' power (new finding)

`scripts/plan_gate_power.py` simulates one plan window: the model's probability differs from the
stake book's by a normal amount, and the truth sits between them. It is stylized (prices near even,
independent games), so read the size of the numbers, not the decimals.

Current gates, 2% minimum edge, chance of passing **both** windows:

| Games per window | Bets per window | Model fully right | Model half right | Market right |
|---|---|---|---|---|
| 200 | 54 | 16% | 10% | 1% |
| 200 (wider disagreement) | 92 | 38% | 21% | 2% |
| 400 | 108–184 | 32–49% | 18–21% | 1–2% |
| 800 | 215–369 | 40–59% | 18–22% | 1–2% |

The gates almost never pass a model with no edge, which is their job. They also fail a real edge
most of the time, because two of them are noisy at this sample: Brier and log loss "no worse than
the market" on 200 games is close to a coin flip when the model and market differ by 2 points, and
ROI ≥ 0 on 50 bets has a standard error near 13 points.

Closing-line value is the one precise measure. Requiring mean bet CLV to be two standard errors
above zero:

| Bets | True mean CLV 0% | 0.5% | 1% | 2% |
|---|---|---|---|---|
| 50 | 3% pass | 41% | 93% | 100% |
| 100 | 3% | 69% | 100% | 100% |
| 200 | 2% | 94% | 100% | 100% |

So a CLV-led plan separates an edge from no edge in 50 to 100 bets, where the current gates cannot
in 800 games. This is the basis of decision D26 below. Frozen plans stay as they are; a different
design applies only to plans frozen after the decision.

## 4. Go-live rehearsal: three defects after the review

A synthetic NCAAB spread Pinnacle plan (650 predictions, temporary store, no network) reached
`VALIDATION_PASSED`, passed the deployment audit, was reviewed to `PROVISIONAL_VALIDATED`, was
activated, and cleared `market_gate`. The stake was still $0. With all three defects patched by
hand it was $5.00 on a $1,000 bankroll, so nothing else blocks.

| # | Defect | Where | Effect |
|---|---|---|---|
| B1 | The bridge row carries `start` but not `game_start_utc` | `app_core/qualified_bridge.py`; read by `app_core/candidate_chronology.py` | The app rebuilds best picks after the bridge injects and drops the row as `game_start_missing` |
| B2 | The bridge row carries `line` but not `market_line_used` | `core/live_wager_contract.py` (`_format_best_pick`) | The ticket is rewritten as "(No Line)", never matches the Gemini review, and is held |
| B3 | The market activation does not supply the sport exposure cap | `core/live_wager_contract.py` (`sport_caps` from the legacy wager policy) | $0 unless a second, sport-level legacy policy also validates. Nothing derives that from a market-skill pass. This is deferred decision D23 |

Related risks found on the way:

- The go-live checklist shows 8 of 8 passing while the stake is $0; it has no wager-policy item.
- No test takes a bridge row through `build_best_picks_df` or `finalize_live_wagers`, and no test
  reaches `VALIDATION_PASSED` on a market-skill plan in a real store.
- Staking needs the holdout bet CLV lower 95% bound ≥ 0, stricter than the plan's mean CLV ≥ 0. A
  plan can pass validation and still not stake.
- Plan windows are sized to reach 200 games, not 50 bets.

Not exercised: the full Streamlit run, a real Gemini call, the Drive remote, paid closes and the
legacy wager-policy validation.

## 5. Edge candidates, ranked

Everything measured in the repository, strongest first. Full table with sample sizes and sources is
in the audit notes of this session; the rows below are the ones that matter for a plan.

| Rank | Candidate | Evidence | Bets per day | Status |
|---|---|---|---|---|
| 1 | Novig lag after NBA injury news | After a Pinnacle move a stake book was ≥1% better in 11 of 29 cases against 7 of 51 in the control; 15 of 18 gaps at Novig; gone in about 10 minutes. No bet graded | About 0.6 at 1%, 0.3 at 2% | Capture built today, first slate tonight |
| 2 | Novig against Pinnacle without news | 14% of control snapshots ≥1% in the lag study. Today's replay on once-a-game capture: 0 bets | 0 so far | Needs frequent capture to test (M2) |
| 3 | Research score models (NBA, NCAAB, NHL) | None yet; 0 training games | Unknown | Accrues on its own; no date until teams have history |
| 4 | NCAAF totals against opening lines | 54.7% on 179 picks, 0.6 standard errors above break-even | — | Within noise; needs opening-line capture; football paused |
| 5 | MLB props | 58.6% on 741 graded, never compared with price (the first logged price, −141, breaks even at 58.5%) | — | Benchmark offline before spending anything |
| — | Legacy blend, NCAAB and NCAAF form and ratings, NFL and MLB score models | At or below the market in every test | — | No further work |

Never measured, and cheap to measure from data already stored: the price screen's outcomes by
book, TheOver sharp-money and purple-star signals, and Gemini agree against disagree.

The honest reading: the only signal that beats its control is short-lived and small. At 0.6 bets a
day a 50-bet window takes 12 weeks, so candidate 1 alone cannot validate under the current plan
shape this season. Either the same lag shows up without news when quotes are captured often enough
to see it (candidate 2), or the plan must count bets across both basketball sports (D27).

## 6. Roadmap

Dates are targets in ET and assume owner approval of each PR.

### M1. Make a passing market stake (code; about 2 days; start now)

1. B1 and B2: the bridge row carries `game_start_utc` and `market_line_used`.
2. An end-to-end test built from the rehearsal: synthetic store → `VALIDATION_PASSED` → review →
   activation → bridge → `build_best_picks_df` → `finalize_live_wagers` → non-zero stake. It fails
   today at B1, B2 and B3 in turn.
3. B3, after D23: the recommended fix takes the sport cap from the verified market activation's
   own policy (`sport_exposure_cap` 0.03 in the frozen plan), so one validated, activated market is
   enough.
4. The go-live checklist gains a "stake reachable" item that runs the same sizing code and shows the
   dollar amount, and a staking-CLV item (holdout lower 95% bound).

**Status 2026-11-09: all four items done (owner decided D23 the same day).**

- B1 and B2 are fixed in `app_core/qualified_bridge.py`. The end-to-end test found a third bridge
  defect of the same kind: the pipeline's line guard rebuilds the line from `line_source` and
  `live_spread_line` / `live_total_line`, so the rebuilt row read "line unresolved". The row now
  carries those names too. The bridge label also uses the wager contract's format, so a
  whole-number line ("-3.0") still matches its Gemini review.
- `tests/test_go_live_end_to_end.py` runs the whole chain on a synthetic store (about 50 seconds).
  In the app's own configuration (no legacy wager policy file) the stake is $5.00 on a $1,000
  bankroll.
- B3 (D23, owner 2026-11-09): `finalize_live_wagers` takes a sport's exposure cap from the policy
  of a market activation the market gate verified for a candidate in that run. The legacy wager
  policy still supplies caps for its own sports. Without a verified activation the cap stays 0.
- `app_core/go_live.py` has `stake_checks` (`staking_clv`, `stake_reachable`) and `full_checklist`.
  The go-live script and the app's readiness report use `full_checklist`, which is not ready while
  the sport exposure cap is 0 (it reads the verified activation's policy, as the contract does). The item shows the cap in dollars; it does not size a real pick.

Done when the test passes and the checklist cannot show all-pass with a $0 stake.

### M2. Find out whether the Novig lag has volume (capture; first answer 2026-11-16)

1. Read tonight's news capture (first run 4:47 PM ET) and the Wednesday 11:07 AM replay.
2. Extend the news loop's price poll to a **fixed-interval watch** of games in their last 3 hours,
   NBA and NCAAB, Pinnacle plus the four stake books. This tests candidate 2 and also puts entries
   where close coverage is 90%. Cost estimate before building: about 4 credits per request; a
   15-minute poll over a 6-hour slate is about 100 credits a day per sport (D28).
3. Filter Novig's alternate-line junk prices (two-sided overround above 1.15) from replay and
   prediction inputs (awaiting the owner's answer; D29).
4. After 7 days, report bets per day at 1% and 2%, mean CLV with its standard error, and close
   coverage, split by news and no news.

Go or no-go on 2026-11-16: at least 2 bets a day across NBA and NCAAB with mean CLV above zero.

**Status 2026-11-09 (owner approved D28 for both sports and D29 the same day).**

- Item 2 is built: `app_core/price_watch.py`, `scripts/price_watch_loop.py` and the
  `price-watch.yml` workflow. NBA every 15 minutes, NCAAB every 30 (its slates run about twice as
  long), stored games starting within 3 hours, one 4-credit request per sport per interval. Its
  own allowance is 200 credits a day and 1,400 in all, and the code stops after the night of
  2026-11-16. The schedule runs once this is merged to `main`: 4:17 PM ET each day for 340
  minutes, and 11:30 AM ET on Saturday and Sunday. A game that tips after about 10 PM ET loses the
  last part of its window, and on a full weekend day the daily cap can end the watch early.
- Item 3: `pinnacle_model.without_junk` drops US-book quotes whose two sides sum above 1.15, and
  `replay_estimate` uses it and reports the count. Prediction inputs are **not** filtered yet. A
  running plan's rules are frozen, and the functions it hashes were left untouched so that plan
  v2 keeps predicting. The filter becomes a frozen term of the next plan design (M3).
- Items 1 and 4 are reads of captured data and follow as it arrives.

### M3. A plan design that can pass a real edge (owner decision, then code; about 3 days)

After D26 and D27, add a `market-skill-v2` methodology for new plans only:

- primary gate: bets ≥ the minimum and mean bet CLV at least two standard errors above zero, in each
  window;
- sanity gates: bet ROI above a loss floor, close coverage ≥ 80% of bets, coin-flip bounds;
- Brier and log loss against the market reported, not blocking;
- windows sized by expected bets, not games;
- the plan's CLV gate and the staking CLV bound use the same statistic, so a pass can stake.

`scripts/plan_gate_power.py` becomes the acceptance check: the design must pass a true 1% CLV at
least 90% of the time and no edge at most 5% of the time.

### M4. Freeze, validate, review, activate (calendar time; owner steps)

1. Freeze a plan only when M2's replay shows the bet rate. Do not freeze today's v3 draft
   (hash `ff2d59dc…`): its replay shows 0 bets. Let v2 run out; it costs nothing.
2. Validation window, then holdout. At 2 bets a day and 50 bets a window that is 25 days each; at 4
   a day, 13 days each.
3. Deployment review, exposure ledger, market activation, bridge variables (the 14 owner steps are
   listed in the rehearsal report and go into the go-live checklist in M1).

Earliest realistic first validated market: **late December 2026 to mid January 2027**, and only if
M2 says go. If M2 says no-go, the next candidates are the research score models (M5).

### M5. Second-line candidates (parallel, low cost)

1. Research score models: no work until 120 training games exist; then draft, replay, and treat like
   any candidate. Enable NHL score settlement so NHL accrues too.
2. Offline, from stored data, no credits: price-screen outcomes by book and timing; TheOver signals
   against results; props against their logged prices.
3. Football stays paused (D25) unless a screen shows an edge.

### Order and dependencies

M1 and M2 start now and are independent. M3 needs D26 and D27. M4 needs M1, M2's go, and M3.
M5 never blocks.

## 7. Owner decisions

| ID | Decision | Recommendation |
|---|---|---|
| D23 | What a stake needs besides the market activation | **Owner-chosen 2026-11-09: yes.** The verified market activation supplies the sport cap; no second legacy policy is needed for market-skill markets. Implemented in M1 |
| D26 | Gate design for plans frozen from now on | CLV-led `market-skill-v2` as in M3. Current gates pass a true 2% edge about 1 time in 4 |
| D27 | May one plan count bets across NBA and NCAAB for the same strategy | Yes, for the Pinnacle/Novig lag only, with per-sport results reported. Plans prohibit pooling today; without it the news edge needs most of a season |
| D28 | Credits for the fixed-interval watch | **Owner-chosen 2026-11-09: both sports.** About 100 a day per sport for 7 days (its own allowance: 200 a day, 1,400 in all), then review |
| D29 | Novig junk-price filter | **Owner-chosen 2026-11-09: approve.** In the replay now; in predictions as a frozen term of the next plan (M3) |

## 8. What would change this roadmap

- Tonight's news capture or Wednesday's replay showing bets: M4 can start sooner.
- A week of frequent capture showing mean CLV at or below zero at Novig: the lag is not an edge at
  the prices we can get; stop M2 and wait for M5.
- Novig limiting or repricing: the entry books fall back to three US books, where the screen shows
  −2.6% EV.
