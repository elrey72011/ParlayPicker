# Football Stage 2: first run, 2026-11-05

**Source.** [Football Stage 2 run 42065772785](https://github.com/elrey72011/ParlayPicker/actions/runs/42065772785) on `main` (owner-approved).

- **Stage 1 refresh.** `PARTIAL_FAILURE`, with backup read-back verified. It made the first successful football backup in days: the restore cache made it possible.
- **Earlier attempt.** The first attempt (run 42051035520) was refused (`STAGE1_AUTHENTICATED_REFRESH_NOT_VERIFIED`). Its backup uploaded about 6,200 of 6,864 backlog records, then the Drive sync failed.

## Result: no football model can be trained yet

| Scope | Unique games | Settled games | Independent games selected | Development / selection / calibration / holdout | Gate (minimum) |
|---|---|---|---|---|---|
| NCAAF Spread | 375 | 376 | 156 | 86 / 24 / 22 / 24 | 200 / 60 / 60 / 60, two seasons |
| NCAAF Total | 375 | 376 | 157 | 86 / 24 / 23 / 24 | same |
| NFL Spread | 102 | 92 | 47 | 26 / 7 / 10 / 4 | same |
| NFL Total | 102 | 92 | 47 | 26 / 7 / 10 / 4 | same |

**Blockers.** Every scope shows the same blockers:

- `DEVELOPMENT_INDEPENDENT_N_BELOW_200`, and each later partition below 60;
- `TWO_DISTINCT_SEASONS_REQUIRED`;
- `RESULT_AVAILABILITY_CROSSES_*`;
- `DEVELOPMENT_CLASS_SUPPORT_INSUFFICIENT`.

The status is `INSUFFICIENT_TRAINING_EVIDENCE`. There is no model, no calibration and no research prediction.

**Rows excluded from NCAAF Spread training** (one independent observation per game and market, `football-one-observation-v1`):

| Reason | Rows |
|---|---|
| Other books and reprices (`NONSELECTED_BOOK_OR_REPRICE`) | 9,618 |
| Results not verified (`RESULT_UNVERIFIED`) | 2,230 |
| Settlements not reproducible (`SETTLEMENT_UNREPRODUCIBLE`) | 673 |

## What this means

1. **Two seasons are required, and all captured data is 2026.** No football scope can train before the 2027 season, however many more 2026 games are captured.
2. **NCAAF, even within the season, has 156 of the roughly 380 independent games the gates need.** With about two-thirds of the regular season played, the rest of the season will not close that gap alone.
3. **The v3 plan proposal (D12) has no measured inputs yet.** Without a trained model there is no Brier or log-loss variance, push rate or dependence estimate to size a sample from, so any v3 minimum would be invented. The honest v3 step is to capture the full 2026 season cleanly, now that the restore cache is in place, and design v3 from the first trained model.
4. **The faster path is the long-season sports.** NBA, NHL and NCAAB research models already train and record predictions in the canonical store. They need captured events, now unblocked, plus plan binding (C2) and certified closes (C1, NBA spread first under D13).
