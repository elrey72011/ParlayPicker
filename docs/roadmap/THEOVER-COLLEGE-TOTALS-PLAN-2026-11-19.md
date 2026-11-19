# TheOver's college totals pick: plan for a formal test

**Approved by the owner on 19 November 2026 ("Yes to all six").** The rules below are fixed in
`docs/roadmap/theover-college-totals-plan-v1.json` (SHA-256
`6f6a2bc21d52881d0abe37e5d0ea2d8220f63ca7fa71c1862826e9f966e0f1ee`), committed before the first
night of the test. A changed rule is a new version of that file, never an edit.

## In short

TheOver's model-backed totals pick is the only input on the board with a winning record, and that
record has not held since the pick went live: 131-82 on college games over twelve nights, 24-29
on the three nights since 16 November. This plan fixes, before any more games are played, what
"it works" will mean: which picks count, how long the test runs, and the bar it must clear on
closing prices. It changes no rating and approves no stake.

## What is tested

**The claim:** on a college basketball total where TheOver lists the recent hit rate of the model
behind its pick, TheOver's side beats the number the sportsbooks close at.

| | |
|---|---|
| Games | College basketball totals, every game the app matches exactly to TheOver's file. A game matched only by similar names is left out, as it is from the blend today. |
| The bet | TheOver's side of the total, at the line and the best price the board shows, in the last run the app saved before the game started. One bet a game, one unit each. It counts whether or not it is the board's pick for the game. |
| The price | The best price at that line among Novig, FanDuel, DraftKings and BetMGM in that run. |
| The close | The same sportsbook's line and price for the same side, read within 30 minutes before the start by the closing-capture job (normally in the last 12; pull request 2578). |
| The result | The final score from the grading job that already grades the saved picks each morning. A push returns the unit and counts as a bet with zero return. |
| College football | Counted and reported the same way, with no bar this season: 46 picks in twelve nights, and too few games left for two test windows. |

## The bar

The same market-skill rules the college basketball spread plan is judged by (decision D26 of
9 November), in each of two windows:

| Rule | Bar | Why |
|---|---|---|
| Bets | At least 100 | Fewer cannot tell an edge from luck |
| Closing prices | Read for at least 80% of the bets | Without them the test below is not measured |
| Beat the close | The average move of the total toward the pick, in points, at least 2 standard errors above zero | The one measure precise enough to settle in weeks, not seasons |
| Loss floor | Units won no worse than 25% below zero | Catches a broken plan, not bad luck |
| Probability sanity | Brier score at most 0.26, calibration error at most 15 points | A coin flip plus room for noise |

Reported beside these, never deciding: wins and losses with a 95% interval against the break-even
rate, units won, Unders and Overs apart (79-44 and 52-38 on college games over the twelve nights),
and the picks where the board's line differs from TheOver's own.

**Both windows must pass.** The first (validation) shows whether the bar is met; the second
(holdout) shows it was not a lucky stretch. No rule is changed after the first window starts.

## Windows

Between 10 and 23 picks a night have fed the blend in college basketball over the last week, so 14
nights should give 200 to 280 bets a window, at least twice the minimum.

| Window | Nights | Slate dates (Eastern) |
|---|---|---|
| Validation | 14 | 21 November to 4 December 2026 |
| Holdout | 14 | 5 to 18 December 2026 |

A window with fewer than 100 bets is extended a night at a time until it has them. The night of
20 November is the closing capture's first night and counts for neither window. If the capture
has not run on main by 21 November, both windows move later by whole nights, as a new version
of the rules file.

## What the record says today

| Period | College totals, TheOver's side | Note |
|---|---|---|
| 7 to 15 November (nine nights) | 107-53 | The nights that made the case for using the pick. A signal always looks best on the nights that suggested it |
| 16 to 18 November (three nights, live) | 24-29 | Since the pick has moved the rating |
| Twelve nights | 131-82 (college basketball 96-71, college football 35-11) | |

None of the twelve nights has a closing price, so none of them can be judged on the bar above.
That is why the test starts fresh.

## What happens at the end

- **Passes both windows:** the result goes to the owner for review. The step after that is the
  route that can fund a stake: TheOver's picks recorded in the research store under a bound plan,
  with the staking terms approved on 8 November. That is a separate decision.
- **Fails the validation window:** TheOver's college pick becomes record only, as its NBA pick
  did on 17 November and its NHL pick on 19 November. Those games then rest on the price alone
  and read "No lean".
- **Passes validation, fails holdout:** the same as a failure.

One limit, known when this was approved: the bar asks the market to move toward TheOver's side by
the close. An edge the market never notices would fail it even while winning. The report shows
wins and losses as well, so that case would be seen, but 200 to 280 bets cannot prove a 57% pick
on results alone: its interval would still reach below the break-even rate.

## What is built

- **Built in pull request 2578:** the closing-capture job reads a close for TheOver's side on every
  such game, also where it is not the board's pick; the rules file above.
- **Built in pull request 2579:** the read-only report, printed by the morning grading job with the
  paper track record (`core/theover_plan.py`): the plan's tally (bets, record, units, closes read,
  average move toward the pick and its standard errors) from the saved runs, with the rules file's
  fingerprint each day so a changed rule cannot pass unnoticed. It changes no rating, pick or stake,
  and needs no Odds API credits beyond the closing capture (about 80 a day on a 46-game night).
  Two readings it had to make, stated here because the rules file does not spell them out: a game
  that starts before 5 AM Eastern is counted with the night before; and a close is matched to a bet
  by provider event, side and the book the pick was bound to, whichever saved run the capture
  recorded it against, with the move counted from the bet's own line.
- **Also in pull request 2579:** the closing-capture job picks up a run saved after it started, so
  the last Refresh before a game is the one whose picks get their closes.

## Decided by the owner, 19 November 2026

1. **Scope.** College basketball totals now; college football counted with no bar this season.
2. **Which pick is the bet.** The pick in the last run saved before the game: the one the closing
   capture and the paper track record already use.
3. **The bar.** The market-skill rules above, with 100 bets a window.
4. **Windows.** 14 nights and 14 nights.
5. **On failure.** TheOver's college pick becomes record only.
6. **Hockey.** TheOver's NHL pick is record only from the first Refresh after pull request 2578 is
   merged: 30-28 over twelve nights and 2-4 since it went live, and its line often sat a full goal
   from the board's. This changes NHL totals ratings: those games rest on the price, or on Kalshi
   where it counts.

## Where the numbers come from

Twelve graded nights, 7 to 18 November 2026: the owner's audit exports, each pick at the line of
the last export before its game, graded from final scores. Picks with an unresolved line are left
out. The 24-29 is the forward record kept since 16 November against the 57% review line agreed
that day; this plan replaces that informal line.
