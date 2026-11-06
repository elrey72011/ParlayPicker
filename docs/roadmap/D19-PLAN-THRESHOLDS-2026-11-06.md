# D19: the frozen probability thresholds cannot be met by a spread or total market

Found 2026-11-06 while preparing go-live (C4). Owner decision needed.

## The problem

Every validation plan (`app_core/prospective_validation_plans.py`, kept by the bound v2 plans)
requires, on both validation and holdout, **Brier <= 0.24** and **log loss <= 0.68**, scored on
the predicted side's decided probability for every game (`P_WIN / (P_WIN + P_LOSS)`).

A spread or total is priced so each side is close to 50%. A 50% forecast scores Brier 0.25
and log loss 0.693. To score lower, a model must give probabilities far from 50% *and* be right.
Even a model that knows each game's true probability exactly (perfectly calibrated):

| spread of true probabilities | hit rate betting the favored side | ROI at -110 | Brier | log loss | passes 0.24 / 0.68 |
|---|---|---|---|---|---|
| 0.03 | 52.4% | 0.0% | 0.2491 | 0.6914 | no / no |
| 0.05 | 53.9% | +3.0% | 0.2476 | 0.6884 | no / no |
| 0.0625 | 55.2% | +5.3% | 0.2459 | 0.6849 | no / no |
| 0.08 | 56.3% | +7.5% | 0.2436 | 0.6801 | no / no |
| 0.10 | 57.9% | +10.5% | 0.2403 | 0.6733 | no / yes |
| 0.125 | 59.9% | +14.4% | 0.2344 | 0.6608 | yes / yes |

(Simulated, 400,000 games per row.) Passing needs about a 60% hit rate against the spread on
every game, roughly +14% ROI, which no public model sustains. A genuinely profitable
model (54-56%, +3% to +7%) **fails**. Our own evidence agrees: the NCAAB line screen
scored Brier 0.250-0.251, and the research models sit within about a point of the line.

So under the current thresholds no spread or total market can qualify, NCAAB spread
included, however good the model or price policy is. C6 (built) would never switch on.

## Options

**A. Keep the thresholds.** Nothing qualifies; picks stay research-only.

**B. A new plan methodology (v3) that measures skill against the market (recommended).**
On the same games, the model must be **no worse than the market's own de-vigged entry price**
(Brier and log loss), and the edge tests become the ones that matter for a bet:
* calibration error <= 0.08 and coverage >= 90% (unchanged);
* verified entry prices and certified-close coverage (unchanged);
* paper ROI >= 0 (unchanged) **and** mean CLV >= 0 against certified closes (new).

This is the same market-anchored standard as the blend-weight study. It is a new plan
version: v2 stays frozen and its records are kept; v3 gets its own windows (sized by game
count, D14), so the first qualification date moves later by about one window.

**C. Coin-flip bounds** (Brier <= 0.25, log loss <= 0.693). Easy to pass by a near-50%
model with no skill; it leaves ROI as the only real test. Not recommended.

## What B would change in code

`evaluate_validation_plan` gains a v3 branch (v1/v2 evaluation unchanged), the plan
drafter (`prospective_calibration.draft_bound_plan`) can draft v3 for review, and the
owner freezes it with the Bound plan workflow as before.
