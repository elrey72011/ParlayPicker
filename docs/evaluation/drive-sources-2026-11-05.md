# Drive evidence by sport, and why NBA/NCAAB/NHL are empty (2026-11-05)

Sources: Market readiness run 42077461084 (`with_sources=true`, read-only restore,
`writes_refused=0`) and Research capture and grading run 42071186506 (cycle audit).
Nothing here is a model-quality claim.

## Per-sport research source stores in Drive

| sport | store | Drive records | market | captured events | quote rows | scores observed | settled | replay-verified quotes |
|---|---|---|---|---|---|---|---|---|
| NFL | nfl-market.sqlite3 | 25 | SPREAD / TOTAL | 15 / 15 | 326 / 328 | 14 / 14 | 0 / 0 | 0 |
| NCAAF | ncaaf-prospective.sqlite3 | 11 | SPREAD / TOTAL | 4 / 3 | 8 / 4 | 4 / 3 | 4 / 3 | 0 |
| NBA | nba-market.sqlite3 | **0** | SPREAD / TOTAL | 0 | 0 | 0 | 0 | 0 |
| NCAAB | ncaab-market.sqlite3 | **0** | SPREAD / TOTAL | 0 | 0 | 0 | 0 | 0 |
| NHL | nhl-market.sqlite3 | **0** | PUCK_LINE / TOTAL | 0 | 0 | 0 | 0 | 0 |
| MLB | mlb-pregame-receipts.sqlite3 | 271 (+1,810 receipts) | RUN_LINE / TOTAL | 127 / 127 | 254 / 254 | 127 / 127 | 127 / 127 | 254 / 254 |

Canonical store: 82,101 records, all football except MLB reconciled facts
(1,468) and the 16 frozen plans. `prospective_event`, `prospective_quote`,
`prospective_model`, `prospective_prediction`, `prospective_calibration` and
`prospective_close` are empty for every sport.

## Finding 1: NBA, NCAAB and NHL have never been captured (G5)

There is no NBA/NCAAB/NHL data stuck outside the canonical store: their Drive stores
are empty. The scheduler reads `RESEARCH_SPORTS` from the repository variable, which
is set to `MLB,NCAAF,NFL` (run 42071186506 log), overriding the workflow default
`NFL,NCAAF,NBA,NCAAB,MLB,NHL`. **Owner action:** set the repository variable
`RESEARCH_SPORTS` to `NFL,NCAAF,NBA,NCAAB,MLB,NHL` (or delete it to use the default).
Until then the C1 closes and the C2 calibration chain have nothing to work on.

## Finding 2: NCAAF capture has failed since 2026-10-03 (D15)

The cycle audit's only error is `NCAAF:stale_frozen_model`. The frozen NCAAF model is
pinned to a hash of five source files (`ncaaf_prospective.runtime_hash`). Two later
commits changed files in that set without touching the model, feature or history code:

| commit | date | change | runtime hash |
|---|---|---|---|
| 0d36bd74 | 2026-09-13 | NFL aliases (already reviewed) | d5acdd2c… |
| 93a7f236 | 2026-10-03 | NCAAF identity aliases: McNeese, LSU, "gardnerwebb" | 6f314e5f… |
| 1ccf0a28 | 2026-10-06 | NHL New York names in core/team_mapper.py | 8283b60a… (current) |

PR #2462 adds this exact transition to the reviewed recovery
(`research_model_recovery.REVIEWED_TRANSITIONS`); anything else still refuses. The
aliases can change which LSU, McNeese and Gardner-Webb games match, so the owner
reviews it before merge.

## Finding 3: NFL discovers events but captures no quotes

NFL discovered 28 events this cycle and captured 0 quote rows (Drive has 15 captured
events in total). To be investigated separately.
