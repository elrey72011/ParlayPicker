# Prospective NCAAF evaluation

This workflow is paper research and never feeds production stake approval. It compares the frozen ridge model and baselines at observed sportsbook quotes. The evaluated historical holdout is not reused for tuning.

## Run in Streamlit

Open **Data Maintenance → NCAAF Prospective Evaluation**.

1. First setup: load the completed historical checkpoint, run the fixed research evaluation, then click **Freeze research models for prospective capture**. After a restart, use **Restore / back up prospective evidence** to recover an existing frozen cohort instead. Re-freezing an unchanged artifact is idempotent.
2. Click **Refresh current NCAAF inputs**. If needed, click **Continue current NCAAF inputs** until ready. Each click makes at most six CFBD requests. Start a fresh refresh each day; capture requires all inputs to have been retrieved within 24 hours.
3. Before kickoff, click **Capture prospective NCAAF predictions**. It makes one The Odds API request for US moneyline, spread and total markets. Provider billing is by markets/regions, not simply HTTP-call count.
4. Click **Restore / back up prospective evidence** to upload and read-back verify the new records. Local saves alone do not establish restart durability. Downloads provide a portable audit copy; restore is from Drive.
5. After games finish, click **Grade pending NCAAF predictions**, which checks at most six exact CFBD game IDs. Repeat as needed. Unfinished games rotate behind less recently checked games; they are never assigned fabricated results. Back up again after grading.
6. Download the prospective evaluation report for review.

These are manual controls, not a scheduled job. Capture again on the appropriate pregame days. At least three same-season prior games, all strictly more than seven days before kickoff, are required for both scoring and yardage features. Early-season games may therefore have no predictions. Skipped reasons are retained in the records download.

## Integrity and selection policy

The feature builder uses real earlier completed games and the upcoming schedule without supplying fake future scores. Raw allowlisted feature inputs and their retrieval timestamps are frozen with each capture. Original historical publication timestamps remain unknown, but these actual prospective inputs are observed before the predicted event.

An event requires an unambiguous canonical home/away match and kickoff agreement within 60 seconds across CFBD and The Odds API. No fuzzy matching is used here. Already-started games, uncertain starts, missing features, future quote timestamps and quotes older than 15 minutes are excluded. Models and feature/identity implementation hashes define the cohort; a changed implementation requires a new freeze.

Every model saves its candidate probabilities and the exact source book, market, line, American/decimal price and quote timestamp. Win/push/loss probabilities sum to one. Moneyline ties are treated as void/push in this paper policy. No-vig implied probability is included only when a unique fresh opposing quote exists at the same book and matching line; it is conditional on a decided outcome where pushes are possible.

Before outcomes exist, each model selects the highest-EV candidate per game, with a deterministic tie-break. A hypothetical one-unit wager is counted only when EV is positive; all live stakes remain zero. Report selection is the **first eligible capture per game and cohort**, independent of outcome, not the most favorable later capture. Different models may select different markets. Cohorts are reported separately.

Scores are joined by exact CFBD game and home/away IDs. The initial grading workflow does not automatically revisit already graded games for provider corrections. Append-only capture and score records are stored separately from production evidence in `ncaaf-prospective.sqlite3`, with immutable Drive objects under `parlaypicker/ncaaf-prospective-v1/`. Drive restore merges records and verifies content hashes; backup verifies read-back.

The report shows graded counts, paper hit rate excluding pushes, paper return per staked unit, three-way Brier score, log loss and calibration bins. Observed quotes do not prove actual execution. No market-independent significance claim, automatic scheduling, or production approval is provided by this workflow. Closing proxies are described below.

References: [The Odds API v4](https://the-odds-api.com/liveapi/guides/v4/) and [CFBD games](https://api.collegefootballdata.com/api/games).

## Exclusion diagnostics

The report and sidebar summarize exclusions from the latest capture. Matching exclusions distinguish already-started events, the seven-day window, missing start/team information, absent or reversed team pairs, kickoff disagreement, uncertain/completed schedule entries, invalid IDs and ambiguous matches. The JSON includes source team names, normalized names and up to ten matching schedule candidates. These diagnostics preserve existing capture gates. Older captures retain their original combined reason; a new capture is needed for detailed reasons.

After deploying this change, freeze the research models again to create a cohort with the updated implementation hash before capturing. Existing evidence remains preserved.

## NCAAF provider identities

Prospective matching uses a scoped exact alias table in `app_core/ncaaf_identity.py` before the generic name mapper. This handles provider abbreviations, mascot names and accented spellings without fuzzy matching or changing other sports' aliases. The identity module is part of the frozen runtime fingerprint, so deploying alias changes requires a new cohort freeze. Matching a game still does not waive minimum history, freshness, ambiguity or kickoff checks.

## Closing proxies

Click **Capture NCAAF closing proxies** in the final 30 minutes before kickoff. The action is manual, uses one three-market odds request, and can save observations even before model history is sufficient. Both collection time and source quote time must precede kickoff; source quotes must be at most 15 minutes old and within the final 30-minute window. Back up prospective evidence afterward.

Download **NCAAF closing-line report** for comparisons against the first eligible prediction per cohort. It uses the latest available observation of the selected book/market, with exact event ID, team orientation, kickoff and line equality. Changed lines, ambiguity and missing comparable quotes produce no price CLV. Price CLV is entry decimal odds divided by closing decimal odds minus one; positive means a better entry payout. This is raw price CLV, not no-vig value or proof of execution. These manual observations are closing proxies, not guaranteed final closes. No historical or in-play quotes are substituted.

Closing records use the existing append-only store and verified Drive backup. This reporting addition does not change the frozen prediction runtime and does not require a new model freeze.
