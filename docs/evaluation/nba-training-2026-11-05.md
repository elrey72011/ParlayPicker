# NBA ESPN benchmark: first training run (2026-11-05)

Owner decision D9 approved one bounded download of NBA game history, with no Odds API credits.

> **Status after the repository audit (2026-11-05): offline benchmark only.** The NBA model in the validated chain already exists: `app_core/prospective_research_models.py` (`pregame-score-ridge-normal-v1`). The research scheduler retrains it after each NBA capture, from Odds API events and final scores bound to captured quotes. This ESPN model uses the same features and ridge method on a longer history. Its games have no quote lineage, so the evidence store cannot accept it as a model. It is kept only to test whether more history helps.

## Data

- **Source.** ESPN's public NBA scoreboard, one request per day, the same feed `app_core/espn_results.py` uses. The free `nba_api` endpoint (`stats.nba.com`) does not answer from the cloud container: it returned an empty reply after 31 s.
- **Window.** 2024-25 and 2025-26 regular seasons, plus 2026-27 through 2026-11-04. That came to 2,565 completed, non-neutral regular-season games.
- **Excluded: 7 games.** These were All-Star games ("Team Shaq", "World", …) that ESPN labels as regular season, including 1 duplicate. Training now keeps only games between the 30 NBA franchises (`NBA_FRANCHISES`). The excluded games never had enough team history to become training examples, so the fit was unchanged.
- **Checks.**
  - 30 franchises.
  - 1,228 / 1,229 / 108 games per season.
  - Mean total: 228.9 points.
  - Home teams won 54.8%, by a mean margin of +1.69.

## Model and honest holdout

- **Model.** `nba-espn-benchmark-v1` (`app_core/nba_espn_benchmark.py`; first run named `nba-score-ridge-v1`): ridge regressions for margin and total. Features are each team's last 2–8 games within 240 days.
- **Split.** Time-ordered: trained on 2,020 games through 2026-02-20, then scored on the 505 later games. Ridge strength was chosen inside the training window.

| Holdout, 505 later games | Model RMSE | Always predict the training average | Gain |
|---|---|---|---|
| Margin | 15.55 | 17.56 | −2.01 |
| Total | 19.66 | 20.46 | −0.81 |

## What this does and does not show

- **It does show** that the model learns real signal: it beats a constant on later, unseen games, for both margin and total.
- **It does not show** that the model beats the betting market. That is the bar that matters, and it is untested.
  - ESPN keeps no lines for completed games, and historical Odds API lines cost credits.
  - Published analyses commonly put the market's closing NBA spread error at around 12 points. **That figure was not measured here.** If it holds, this model's 15.5-point margin error is far less accurate than the market.
- **Known limits.**
  - No injury or lineup inputs (`NO_AUTHENTIC_INJURY_LINEUP_FEATURE`).
  - Early-season form reaches back into the previous season.
  - No pace, travel or back-to-back features beyond rest days.
- **Measurement path.** NBA markets qualify only through the existing chain: the frozen NBA SPREAD and TOTAL validation plans (≥ 200 independent events, Brier ≤ 0.24, log loss ≤ 0.68, calibration error ≤ 0.08, ≥ 80% certified closes, paper ROI ≥ 0), then a deployment review, then owner activation.

## Artifact

- **Not committed.** It briefly was, as research-only. It was withdrawn after the audit because nothing in the validated chain can read it.
- **Not wired in.** A shadow-pipeline hookup (`PARLAYPICKER_PIPELINE_NBA`) was withdrawn for the same reason, before it was ever pushed.
- **Reproduce with** `python scripts/train_nba_model.py --espn --seasons 2024-25 2025-26 2026-27 --through 2026-11-04`. That makes about 560 free ESPN requests.
- **Next useful step.** Compare this holdout with the research model's own chronological diagnostics for the same games. Only a clear gain would justify proposing a longer history window, as a new frozen plan version, to the research model.
