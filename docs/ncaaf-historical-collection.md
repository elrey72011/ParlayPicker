# Historical NCAAF collection

In Streamlit, open **Data Maintenance → NCAAF Historical Collection** and click **Collect next NCAAF batch**. The first click starts the three prior calendar seasons (2023–2025 when run in 2026). Continue until the pending request count is zero. Each click makes at most six requests using the existing CFBD key. Ordinary page reruns and downloads make no CFBD requests.

The collector fetches the full regular/postseason FBS schedule for each season, then team box scores for each season/week/type discovered in those schedules. The weekly team endpoint may also return games outside the FBS slate; those are excluded from feature construction. Week zero and postseason weeks are kept distinct. It stops at the first HTTP, connection or response-shape failure, preserves successful requests, and retries only the unfinished request when clicked again. Empty responses are retained and reported as zero coverage, not treated as proof of usable data. Request completion cannot detect games entirely absent from the provider's schedule.

## Save and resume

Download the checkpoint before leaving the session. **Download NCAAF collection ZIP** includes:

- `checkpoint.json`: allowlisted result/stat records, request parameters and retrieval timestamps; usable for resuming.
- `coverage-audit.json`: per-season game/stat/feature counts, pending requests, invalid or duplicate records and missing/conflicting statistics.
- `research-features.csv`: same-season prior-game scoring and optional total-yard averages, game IDs used, counts and cutoff timestamps.
- `targets.csv`: final scores, home margin and game total, linked by game ID.

**Back up NCAAF checkpoint to Drive** uses the existing Shared Drive configuration and verifies a read-back. It writes an immutable content-addressed JSON object under `parlaypicker/ncaaf-history-v1/`, separately from prediction evidence. Backup failures leave the session data and downloads available. To resume after restart, upload the downloaded checkpoint or the historical JSON downloaded from Drive and click **Load NCAAF checkpoint**. Import replaces the active collection only after validation; it never changes live predictions. Keep the prior checkpoint if switching collections.

## Feature interpretation

Features include only valid completed games from the same season whose kickoff was strictly more than seven days before the target kickoff. The target, simultaneous games, recent games and future games are excluded. No aggregate season statistics, Elo fields or postgame win probabilities are imported. Conflicting duplicate results are excluded; conflicting or score-mismatched team statistics cannot supply yardage features. Scoring feature availability requires at least three prior games for both teams; missing yardage is left missing with its own count. This availability flag is not a model approval.

These are **research features**. CFBD historical responses retrieved today do not establish original publication or correction timestamps. The seven-day lag is a conservative working assumption, not verified historical availability. All rows mark historical publication verification and production eligibility false. A subsequent historical evaluation must disclose this limitation, and prospective frozen snapshots are still needed. Historical odds, betting returns, model fitting, calibration, and production activation are outside this collection step.

After collection, share the coverage audit and ZIP for review. Before fitting, resolve material coverage gaps and fix chronological training/calibration/holdout periods without selecting them based on results.

API reference: [CFBD games and team box scores](https://api.collegefootballdata.com/api/games).
