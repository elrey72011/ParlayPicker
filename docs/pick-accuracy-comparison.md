# Pick accuracy comparison

Results → Candidate Ranking Backtest → **Show pick accuracy comparison** now
compares the saved current ranking, probability-first selection, and a sportsbook
baseline. It also scores the independent model, AI Analysis/TheOver, and Kalshi
on identical original selected tickets. Gemini remains a separate qualitative
review comparison. Opening Results does not compute the new report until its
checkbox is selected; the report makes no network requests.

## Owner workflow

1. Run analysis before games start to capture the complete candidate snapshot.
2. After games finish, grade results using the existing Results workflow. Keep
   the full candidate ledger, not only the selected-picks download.
3. Open Candidate Ranking Backtest and enable Show pick accuracy comparison.
4. Select an evaluation start date that follows the model's training cutoff.
5. Review game counts, win rates, simulated returns, source coverage, and
   exclusions. Download the Markdown report or detailed JSON.

Earlier candidate ledgers can be uploaded through the existing prior-ledger
control. Imported metadata are declarations, not independently attested evidence.
A missing timestamp must not be filled with a convenient pregame time. The model
training cutoff covers fitted/calibrated/tuned components, not just base training.

The comparison does **not** change selection rules, source weights, wager approval,
locked picks, or publication history. A higher observed rate in a small/reused
sample is not a promotion decision.

## Comparison contract

- Reuse `core.selector_validation.build_report` for complete candidate pools,
  same-run identities, duplicate conflicts, latest pregame snapshots, declared
  training/availability times, exact prices, outcomes, and push semantics.
- Exclude an incomplete candidate pool entirely; never choose an older snapshot
  because its grades or probabilities are more convenient.
- Current ranking is the selection originally saved by each run. It does not
  mean today's code replayed on historical games.
- Probability-first ranks `calibrated_probability`, conditional on a decision;
  composite scores and `selection_probability_used` are not probabilities.
- Each ranking chooses exactly one candidate per eligible game. Ties use market
  type and pick identity, never outcomes. The JSON lists each original and
  alternative ticket, plus per-league results and paired changes.
- Win rates exclude pushes; simulated one-unit returns include pushes in turnover
  and use original odds. They are not actual bets, fills, fees, or account profit.
- Probability-source diagnostics use only saved, oriented `blend_in_*` inputs
  on the original selected ticket. Missing/0/1 sentinel inputs are not replaced
  with 50% or ambiguous raw columns. ML target metadata must match the market;
  a home-win classifier is not a spread-cover or total model.
- Each source is scored against the sportsbook probability on exactly the same
  decided tickets. Different source cohorts cannot be compared as if identical;
  a separate all-sources-common cohort is included. League/market breakdowns and
  calibration bins are in the JSON.
- These are source-alone diagnostics, not a measurement of incremental value in
  the live blend. Gemini agreement is qualitative, not a numeric probability or
  causal improvement claim.

## Reproduce from downloaded ledgers

Run from the repository root with project dependencies installed:

```text
python -m scripts.compare_pick_accuracy --audits "path/to/candidate_results_ledger*.csv" --evaluation-start 2026-09-03 --output outputs/pick-accuracy.json
```

This writes JSON plus Markdown and records the SHA-256 of every exact input file.
Choose the evaluation date before inspecting results. Models must declare a
training cutoff earlier than that date and earlier than their prediction time.
Repeatedly inspecting historical results does not create a new untouched test.

## Initial local run, September 13, 2026

The 11 available candidate-ledger exports contained 1,176 rows. An evaluation
beginning September 3 had **zero fully verified eligible games**. Primary
row-level exclusions (including development slates) were:

| Reason | Candidate rows |
| --- | ---: |
| Development slate | 366 |
| Export not verified pregame | 168 |
| Missing identity | 128 |
| Missing or invalid prediction timestamp | 502 |
| Training cutoff unverified or leaking into evaluation | 12 |

These counts describe the submitted historical files, not every game played.
Zero eligible games means insufficient evidence, not a 0% win rate. This run
cannot establish whether changing ranking or increasing a source's weight helps.
Future pregame captures and completed grades should be evaluated through the same
checks before a live ranking/weight change is proposed.
