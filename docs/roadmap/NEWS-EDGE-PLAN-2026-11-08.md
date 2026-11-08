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
2. **Lag study (built, `app_core/news_lag_study.py`, workflow "News lag study (one-time research)";
   owner approved 5,000 credits on 2026-11-08).** It runs once per label (a write-once marker on
   Drive) and stops at the 5,000 cap or when the account has fewer than 4,000 monthly credits left.
   Each piece of news costs 41 credits (the events list plus 4 spread snapshots: 30 minutes before,
   at, 10 and 30 after), so about 120 pieces of news. Flips come first, then first listings within
   3 hours of tip-off.
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

## Decisions

- **Lag study prices: decided 2026-11-08.** A one-time 5,000 credits.
- **Open:** depending on the study, news-triggered capture credits (step 3) or the model build (step 4).

## Run order

1. Merge, then run **NBA injury reports** with `start` 2026-10-21 (the backfill).
2. Run **News lag study** with `start` 2026-10-21 and `end` set to the last recorded day.

## Lag study result (2026-11-08, run 42504314711, 4,625 of 5,000 credits)

All 115 pieces of pregame news from 2026-10-21 to 2026-11-07 were studied (114 matched an event).
100 were status flips; the rest were first listings within 3 hours of tip-off.

| | Count |
|---|---|
| News studied | 114 |
| Pinnacle moved (≥ 0.5 pt or 2 points of probability) | 39 (34%) |
| After those moves, a stake book ≥ 1% better than Pinnacle's new price | **11 of 29** with a comparable price (38%) |
| ... ≥ 2% better | 5 of 29 (17%) |
| Control, 30 minutes before the news: ≥ 1% | 7 of 51 (14%) |

- **Books do lag after news, but almost only Novig.** Of the 18 price gaps of ≥ 1% after news,
  15 were at Novig and 1 each at DraftKings, FanDuel and BetMGM.
- **The gaps are short.** Each of the 3 present at the report was gone 10 minutes later.
  Others opened 10 to 30 minutes after it.
- **Volume is too low on its own.** About 0.6 a day at ≥ 1% and 0.3 a day at ≥ 2%, against
  about 2.3 a day for 50 bets in a 22-day window.
- Only 692 of 1,777 stake-book quotes were at Pinnacle's exact line, so line matching also
  limits the comparisons.

**Next (read-only, no credits):** replay `FIRST_EDGE_CAPTURE` on NBA spreads from the stored
captures. Even without news, Novig's NBA prices were ≥ 1% better than Pinnacle's in 14% of
the control snapshots. The question is whether an NBA plan entering at Novig, with frequent
captures, reaches about 2.3 bets a day.
