# MLB prospective paired evaluation

Under Data Maintenance, expand **MLB Prospective Evaluation**. This compares margin/total forecasts, not bets or win probabilities.

1. Extract checkpoint.json from the MLB history ZIP and pitcher-checkpoint.json from the pitcher development ZIP. Upload those two files and click **Freeze MLB comparison models**. Their source hashes must match. The two models reproduce the fixed ridge alpha-10 team-only and starter-enhanced fits using 2023 matched rows only. They do not refit on 2024 or 2025.
2. Click **Restore / back up MLB evidence** and verify success. Subsequent sessions can restore the frozen models instead of uploading and fitting again.
3. Click **Refresh upcoming MLB games**, choose a game, and **Capture paired MLB predictions** before its scheduled start. Capture makes one season schedule request plus two pitcher game-log requests. Both probable pitchers and minimum history must be available. Calls are manual; a capture taking over five minutes is rejected. No API key is required.
4. Back up again after capture. After games finish, **Grade pending MLB predictions** processes up to six games per click, with at most two API requests per game. Back up after grading and download the report/records.

Snapshots preserve the observed season schedule, pitcher logs, starter IDs, computed features, paired forecasts and cohort. All contributing games were already marked Final in the observed pregame schedule. Both models are evaluated on the same first capture per game/cohort. Starter changes are flagged and retained in the primary comparison; final actual start time provides an additional timing check. Reports keep cohorts separate. Captures rejected by the final storage-time start gate can save an empty attempt; check captured-game counts.

The API lists probable pitchers, not a guaranteed final lineup or original announcement timestamp. This workflow proves the listing was observed pregame; it does not call it confirmed. A late starter change is part of prospective performance. Retrospective actual-starter development remains a different evidence standard. Historical corrections and previously excluded games remain limitations.

Storage is isolated in mlb-prospective.sqlite3, with append-only hashed records and a separate `parlaypicker/mlb-prospective-v1/` Drive prefix. Remote read-back verification is manual. This PR does not automate captures, activate wagering, capture odds or claim an accuracy improvement. Re-freeze only when model/runtime inputs change; identical freezes reuse the cohort.
