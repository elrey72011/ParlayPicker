# NCAAF frozen research model vs archived 2025 lines (2026-11-05)

Source: workflow run 42095167820 ("NCAAF line backtest", roadmap D18 step 1). Research
only; nothing was written to Drive or the evidence store. Archived CFBD lines carry no
quote timestamp, so this is a screen, not validation evidence.

- The rebuilt artifact hash `9334f2a4…` **matches** the frozen cohort artifact, so this
  is the same model the scheduler runs.
- 614 holdout games (2025), all with an archived line (Bovada, DraftKings, ESPN Bet).
- Break-even at -110 is 52.4%. The 95% interval on a hit rate is about ±4 points at
  600 picks and about ±5 points at 400.

| model | market | min confidence | picks | hit rate (±95%) | ROI/pick | Brier (coin flip 0.250) |
|---|---|---|---|---|---|---|
| ridge | SPREAD | 0.50 | 614 | 49.7% ± 4.0 | -0.052 | 0.271 |
| ridge | SPREAD | 0.60 | 302 | 50.0% ± 5.6 | -0.045 | 0.271 |
| ridge | TOTAL | 0.50 | 614 | 50.5% ± 4.0 | -0.036 | 0.260 |
| ridge | TOTAL | 0.60 | 213 | 51.6% ± 6.7 | -0.014 | 0.260 |
| scoring_blend | SPREAD | 0.55 | 451 | 51.4% ± 4.6 | -0.018 | 0.270 |
| scoring_blend | TOTAL | 0.50 | 614 | 48.5% ± 4.0 | -0.073 | 0.260 |
| constant | SPREAD | 0.50 | 614 | 51.0% ± 4.0 | -0.027 | 0.293 |
| constant | TOTAL | 0.55 | 468 | 53.6% ± 4.5 | +0.024 | 0.268 |
| constant | TOTAL | 0.60 | 352 | 53.7% ± 5.2 | +0.025 | 0.268 |

(Full table: the run's `ncaaf-line-backtest` artifact.)

## Finding

**No variant shows an edge over the line.**

- Every spread variant is at or below 51.4%, and every ridge and blend total is at or
  below 51.6%. All lose money at -110.
- Every Brier score is **worse than a coin flip** (0.260–0.293 vs 0.250). The model's
  probabilities are overconfident relative to the line: it disagrees with the market
  more than it should.
- The only positive row, `constant` TOTAL at about 53.6%, is not skill.
  - It is a fixed league-average total, so its picks are "under when the line is above
    average, over when below".
  - It sits well inside its ±4.5-point interval.
  - It is the best of 18 rows tested, so a positive row was likely by chance alone.

## Decision

D18 step 2 (registering this model and binding a v3 plan) is **not recommended**.
Validating it would most likely confirm it has no edge.

The next NCAAF step is **C5, model improvement**. The model has to start from the market
line, not ignore it. One option is to predict the deviation from the line. Another is to
blend the model with the no-vig market probability and shrink it toward the market. This
backtest is the screen for any new variant, which must beat break-even here first. The
2025 season has now been used for screening, so a variant that passes still needs a fresh
prospective validation.
