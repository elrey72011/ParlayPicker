# D30: judge a plan's bets against Pinnacle's close, not the stake book's

Drafted 2026-11-12 for the owner's decision on 2026-11-16. **Nothing here is decided or built**,
except the data collection in "Already in place".

## The problem

The running methodology (`market-skill-v2`, D26) passes a plan when, in each window:

* at least `min_bets` bets (75);
* a certified close **at the bet's own book and line** for 80% of them;
* mean closing-line value on those bets at least 2 standard errors above zero;
* bet ROI no worse than -25%.

The stake is placed at Novig, an exchange. Measured on the replay's first-edge bets
(NCAAB, the 7 days to 2026-11-11, 1% edge; the bets fall on 11-09 to 11-11):

| finding | number |
|---|---|
| Bets with a close at Novig's own line | 22 of 36 spreads (61%); 10 of 23 totals (43%) |
| Same, entered in the last 60 minutes | 18 of 24 (75%); 10 of 17 (59%) |
| "Moved" bets whose Novig close was more than 3 points away | 5 of 12 spreads, 2 of 9 totals |
| Examples | Colgate +29.5 bet, Novig close +11.5 forty-five minutes later; Cal Poly +18.5, close +35.5 |

Two conclusions:

1. **No entry rule reaches 80%.** Closes were fetched for every NCAAB bet; the gap is the
   book leaving the line. A plan frozen today fails on coverage whatever its prices are.
2. **Novig's closing snapshot is not a reliable price.** A line does not move 18 points in
   45 minutes. The book thins before tip-off and the snapshot holds whatever orders remain.
   The same-line closes and the one-point "moves" come from the same snapshots, so the
   +1.0% CLV they show is not trustworthy either, in either direction.

Valued instead at a later Pinnacle capture (not a certified close), the same bets:

| market | entry | priced | mean | t | Pinnacle moved its line: for / against |
|---|---|---|---|---|---|
| NCAAB spread | any time | 22 of 36 | +1.4% | 5.4 | 2 / 7 |
| NCAAB spread | last 60 min | 18 of 24 | +1.4% | 5.2 | 0 / 1 |
| NCAAB total | any time | 14 of 23 | +1.5% | 7.2 | 0 / 3 |
| NCAAB total | last 60 min | 10 of 17 | +1.6% | 6.7 | 0 / 2 |
| NBA spread | any time | 6 of 14 | -0.3% | -0.4 | 1 / 6 |

Bets Pinnacle moved away from are not priced there, and most moved against the bet, so
these means overstate the average over all bets.

## Options

**A. Keep `market-skill-v2`.** No plan at Novig can pass. Picks stay research-only.

**B. A new methodology, `market-skill-v3`: closing value against Pinnacle (recommended).**
For each bet, in each window:

* *Closing value* = bet odds x Pinnacle's no-vig closing probability for the bet's side - 1,
  from Pinnacle's **certified** close for the game and market.
* When Pinnacle closed at the bet's line, that probability is read directly.
* When Pinnacle closed at another line, the probability is carried to the bet's line with a
  fixed value per point, frozen in the plan, and only within a frozen limit (3 points).
  Beyond it the bet has no closing value and counts against coverage.
* A bet whose own entry capture is Pinnacle's close is impossible by construction: the
  certified close is the provider's last snapshot before the start, fetched afterwards.

Gates (proposed; each is the owner's to set):

| gate | v2 | v3 proposed |
|---|---|---|
| Bets per window | 75 | 75 (or 50) |
| Close coverage | 80% at the bet's book and line | 80% with a Pinnacle closing value |
| Primary | mean CLV t >= 2 against the book's close | mean closing value t >= 2 against Pinnacle |
| Loss floor | bet ROI >= -25% | unchanged |
| Reported, not gating | Brier and log loss against the market | unchanged, plus the stake book's own close |

**C. Keep v2 but lower the coverage gate** (say to 60%). Rejected: it keeps a yardstick
shown to be unreliable and keeps dropping the bets that moved.

## What option B has to settle before a freeze

1. **The value of a point: measured 2026-11-12.** From 10,725 NCAAB games with an archived
   closing line (2024-25 and 2025-26, CollegeBasketballData; `app_core/ncaab_point_value.py`,
   the "NCAAB line backtest" workflow with `report: point_value`):

   | market | assumed | measured, 1-point move (SE) | measured, 3-point move (SE) |
   |---|---|---|---|
   | spread | 3.6% | 3.72% (0.13%) | 3.59% (0.07%) |
   | total | 2.4% | 2.36% (0.10%) | 2.45% (0.06%) |

   By line size the value runs from 3.4% to 4.0% for spreads and 2.2% to 2.5% for totals, within
   about two standard errors of the overall figure, so one value per market is enough: **3.6%
   for a side and 2.4% for a total can be frozen as measured**, and they hold out to the
   3-point limit. Archived lines have one provider per game and no timestamp: this is how
   results fall around a closing line, not any book's price. NBA (3.3%, 2.2%) is still assumed.
2. **Selection.** The bet rule already compares the book's price with Pinnacle's at entry.
   A positive closing value says Pinnacle did not move against the bet afterwards. It does
   not say the price was available in size. That needs a separate check (below).
3. **Whether to restrict entry to the last 60 minutes.** Late entry had Pinnacle move its
   line on 1 of 24 spread bets, against 9 of 36 at any time. The rule exists
   (`entry_within_minutes`, #2535).
4. **Spreads only, or spreads and totals.** Both show the same value; totals have fewer bets.

## What it does not answer

* **Liquidity.** A price on an exchange is an order, not a market. Nothing captured so far
  says how much could be staked at it. A plan can pass and still not be bettable in size.
  The honest test is small real stakes after a pass, under the provisional staking terms.
* **Durability.** Three nights of bets. A two-month plan may fail.

## Already in place (no decision needed)

* The readiness report prints the closing value against Pinnacle's last capture, the bets
  not priced and why, and each large move (#2540).
* From this change on, each bet's close request also asks for **Pinnacle's certified close**
  for the same game and market, at no extra cost (a request is charged per 10 bookmakers).
  Without it option B has no certified data to be judged on, and every night not collected
  is a night that cannot be used.

## What building option B takes (after the decision)

* `market_skill_validation`: a `market-skill-v3` rule set and its evaluation, reading
  Pinnacle's certified close; new plans only, so frozen plans keep their hash.
* The bound-plan draft: a `market_skill_v3` choice.
* The readiness report: the plan-progress row for v3.
* Roughly two to three days of work and review. Frozen by about 2026-11-19 if decided on
  the 16th, with windows of about 33 days each at 75 bets (about 22 at 50).

## Decisions for the owner on 2026-11-16

1. A, B or C.
2. If B: carry moved lines with a measured value per point, or do not carry.
3. Entry at any time, or the last 60 minutes only.
4. NCAAB spreads only, or spreads and totals.
5. 75 or 50 bets per window.
6. Whether to keep the price watch running after 2026-11-16: a plan cannot find its bets
   without it.
