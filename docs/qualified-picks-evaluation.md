# Qualified picks and evaluation

New public boards default to Qualified picks. A qualified selection must retain final wager authorization and a positive stake from the private analysis, verified team statistics, an exact supported fresh quote, and positive estimated value at its actual price. These checks do not establish that the model has a profitable edge. Unknown or fallback data stays in the full research board as PASS. Existing locked selections and their original outcomes remain unchanged.

New parlays require two qualified legs, each with positive estimated value and a fresh exact Novig quote. No eligible pair produces no parlay. Combined probabilities assume independence; displayed parlay prices remain estimates requiring confirmation on an actual ticket. Previous publications retain their earlier policy for historical validation and grading.

Net units and ROI on the results table assume one unit risked on each priced settled single at its recorded odds. Pushes return the unit, pending and unpriced records are excluded. No parlay ROI is inferred from separate leg prices. These are hypothetical results, not the user's actual returns.

## Comparing forecasts before changing weights

Use original saved forecasts and outcomes, one selected pick per game/category. Do not replace old prices or probabilities with a later run. Compare each league and market family separately, and retain records predating the public display cutoff for research. A market baseline must be for the same outcome and line, preferably the de-vigged two-way market probability.

Example command (use the column names in the export):

```powershell
python scripts/evaluate_walk_forward.py evaluation.csv game_date calibrated_probability outcome --market-probability-column market_probability --league-column league --min-train-rows 100
```

The report uses identical model/market rows from later complete calendar slates and reports Brier score, log loss, calibration bins, coverage and date boundaries separately by league. It does not fit models or change weights. Negative model-minus-market scores mean better performance in that holdout, not proof of a durable edge. One day's data reports insufficient history. Ordering alone does not prove training provenance; verify it with scripts/validate_selector.py before calling a result out of sample. Test proposed weights on future untouched dates, not yesterday's winners. No live model weights were changed by this update.
