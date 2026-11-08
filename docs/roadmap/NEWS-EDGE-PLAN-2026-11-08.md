# News edge: NBA injury reports (option 3, owner 2026-11-08)

## Why

The NCAAB Pinnacle plan cannot get bets at the books a pick can be staked at (BetMGM,
DraftKings, FanDuel, Novig). Over 41 plan games (2.7 days), no game was 1% or more better
than Pinnacle's no-vig price at either the earliest shared-line capture or any later one.
Today's board agreed: 11 of 104 rows were at least 1% better than Pinnacle and 1 was at least
2%, across all books. Its calibration was also broken (slope -13.2; fixed by `guarded_fit`).
Copying Pinnacle has nothing left to beat.

The next edge source is **news**. A player's status changes (for example Questionable to Out
at 6:30 PM), and for a few minutes some books may not have moved.

**Hypothesis to test first:** after a material status change, Pinnacle moves before the four
stake books. If so, the existing Pinnacle model with the `FIRST_EDGE_CAPTURE` entry rule finds
bets, provided prices are captured right after the news. If not, an injury-adjusted model is
needed (a bigger build).

## Source

The NBA's **official injury report**, a PDF published every 15 minutes at a fixed URL per slot
(`Injury-Report_<date>_<hh>_<mm><AM|PM>.pdf`). This season's slots stay downloadable, so the
archive can be backfilled and a missed run catches up. Last season's return 403.

NCAAB has no official report (ESPN's NCAAB injury feed is empty), so this path is **NBA only**
(NBA SPREAD and TOTAL).

On 2026-11-07, between 10 AM and 11 PM ET, there were 48 reports and 104 status changes, 77 of
them material (into or out of Out or Doubtful). Most were teams' first submissions. A few were
late moves, for example Davis, Anthony: Questionable to Out at 6:30 PM; Dennis, RayJ:
Questionable to Out at 5:30 PM; Thornton, Bruce: Out to Available at 1:45 PM.

## Steps

1. **Archive (built, `app_core/nba_injury_reports.py`, workflow "NBA injury reports").**
   - Each completed ET day is written once to Drive under `parlaypicker/nba-injury-reports-v1/<day>/`.
   - The manifest holds every slot's URL, HTTP status, SHA-256, parsed rows (game, team, player,
     status, reason) and the status changes. The PDF of every report whose rows changed is kept,
     so each row replays from retained bytes.
   - No Odds API credits. Runs daily at 7:17 AM EST; backfill with the `start` input
     (season start 2026-10-21).
2. **Lag study (next; needs an owner credit decision).** For each material change of a player who
   matters, compare Pinnacle and the four stake books before and after:
   - Weight changes by minutes played (`nba_api` is installed).
   - Measure how far the spread or total moved, which book moved first, and how long the stake
     books lagged.
   - Prices: the Odds API historical snapshots around each change (5-minute granularity; about
     10 credits per region per market per snapshot), capped by a one-time budget the owner sets.
     The alternative is live captures from now on (slower to collect, cheaper).
   - Output: the share of material changes with a stake book still 1% or more better than
     Pinnacle's new price, and for how many minutes.
3. **If books lag: news-triggered capture.** When a report shows a material change for a
   rotation player, capture that game's prices at once. Decision capture already runs every
   15 minutes and has Drive write access. The NBA Pinnacle plan then uses
   `FIRST_EDGE_CAPTURE`. It needs the NBA Pinnacle calibration (9 of 40 games on
   2026-11-08), the guarded fit, and a frozen plan.
4. **If books do not lag: an injury-adjusted model.** The model's line moves by the missing
   players' impact (minutes-weighted rating), scored against Pinnacle's price. This is a new model
   and needs its own frozen plan.

## Decisions needed (owner)

- **Lag study prices:** a one-time historical-odds budget (recommended: up to about
  5,000 credits of the 20,000 a month, about 60 changes with 4 snapshots each), or live captures only.
- Then, depending on the study: news-triggered capture credits (step 3), or the model build (step 4).
