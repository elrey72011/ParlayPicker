# NFL spread and total model validation

The September 13 candidate audit contained 52 NFL candidates and 13 selected
NFL games. All 52 candidates reported no configured market-specific NFL model.
The existing home-win classifier is not a spread-cover or total-over model.
Research locking and wager eligibility are separate: permitting a named
sportsbook quote for a research lock does not establish predictive value.

## Decision: do not install this candidate

A new scoring regression was evaluated on 544 regular-season games from the
2024 and 2025 seasons. It failed the existing promotion gate for both targets.
The regression and fitted parameters remain offline research. No NFL probability
was added to the live blend; no confidence, EV, Gemini, or wager threshold was
relaxed. The absence of a validated NFL model remains unresolved.

| Target | Settled, non-push games | Candidate log loss | Market log loss | Candidate Brier | Market Brier | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Home spread cover | 539 | 0.727065 | 0.693798 | 0.265711 | 0.250325 | Reject |
| Total over | 541 | 0.700456 | 0.693336 | 0.253321 | 0.250094 | Reject |

Lower log loss and Brier scores are better. These metrics assess predicted
probabilities, not a bet selection win rate. Pushes are excluded (five spreads,
three totals); no paired prices were missing in this evaluation. The gate
requires improvement on both proper scoring rules. Passing this gate alone
would still not prove an executable edge or authorize automatic installation.

## Fixed evaluation protocol

Before inspecting the evaluation metrics, the split was fixed as follows:

- Training: 2015–2022, 2,079 completed regular-season games.
- Residual-scale calibration: 2023, 272 games.
- Evaluation: 2024–2025, 544 games. No parameter selection used these outcomes.
- Features: each team's points scored/allowed and win fraction across its last
  16 completed regular-season games, requiring at least four previous games;
  home-field indicator. Relocated franchises retain history.
- All games on the same calendar date are featurized before any of that day's
  results enter history. The current game's score is never an input.
- Separate standardized ridge regressions predict home-minus-away margin and
  combined score, with a fixed penalty of 10. The intercept is unpenalized.
- Calibration residual RMS determines each normal residual scale. Integer score
  thresholds reserve a push interval; probabilities condition on no push.
- Compare one outcome per game and target (home cover, over) against paired
  implied probabilities normalized to remove the margin. Opposite sides do not
  double the evaluation sample. Prices and external predictions are not features.

The data source is the public [nflverse schedule CSV](https://raw.githubusercontent.com/nflverse/nfldata/66dc458eee31fa768bce0b6f58edc3c769e752c3/data/games.csv).
Its [official dictionary](https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_schedules.csv)
defines a positive `spread_line` as the number of points by which the home team
is favored; home cover therefore requires `home_score - away_score > spread_line`.
This differs from the app's signed home handicap convention.

The source snapshot SHA-256 is
`566f092836accf6d93362ac9d30ce3f9c99a47bc13d6f0cd4a24520f85942e01`.
The [machine-readable report](research/nfl-market-model-2026-09-13.json) includes
all fitted coefficients, scale parameters, aggregate/per-season metrics, and
the explicit failed promotion decisions. The source CSV is intentionally kept
outside version control; the linked source is pinned to a commit. Verify the hash when reproducing.

## Reproduce

From the repository root with project dependencies installed:

```text
python -m scripts.validate_nfl_market_model --games /path/to/games.csv --report outputs/nfl-model-validation.json
python -m pytest tests/test_nfl_model_validation.py tests/test_market_probability_model.py
```

The script accepts a local dataset and does not call an API, install a model,
write publication history, or change any locked pick. The fitted artifact in
this documentation is not in the live `models/` directory.

## Limits and next development step

This is a simple scoring-history baseline, without quarterback availability,
injuries, roster changes, or opponent-strength adjustments. Its rolling history
also differs from the live NFL season-average enrichment, so even a passing
result would require feature parity before deployment. Historical schedule
prices have no recorded observation timestamp here; they are a probability
benchmark, not a proof that those odds were available at a particular lock time.
No ROI or profitable-wager claim is made.

The 2024–2025 evaluation is now used evidence. Future iterations must identify
it as a reused benchmark and obtain a fresh chronological or prospective test
before claiming an untouched evaluation. A stronger model should first add
pregame quarterback and opponent-adjusted features with known availability times,
then use the same feature builder offline and live. Candidate predictions should
be recorded prospectively without stake until their calibration and value are
supported. More permissive lock rules alone cannot solve the model gap.
