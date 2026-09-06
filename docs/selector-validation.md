# Selector validation

Generate Markdown and companion JSON with `scripts/validate_selector.py`.
It evaluates the exported selector without fitting weights or changing production.

```powershell
python scripts/validate_selector.py report --audits "downloads/graded_candidate_audit*.csv" --selections "downloads/best_picks_export*.csv" --train-through 2026-09-02 --output output/selector-validation
```

`--selections` is optional. Without matching final exports, approval is unknown.
Audit approval flags are ignored because the pipeline initializes them as
backtest-only placeholders. Joins require the exact `export_run_id`, `matchup_id`,
`market_type` and `best_pick`. Compact exports without matchup IDs are matched
only through an unambiguous exact run/date/league/canonical-team/market/pick
identity using `Home`, `Away`, and `Local Date`. Ambiguous identities remain
unknown. Conflicting approvals fail with an error. Final
exports must preserve the approval decision made for that run.

## Required evidence

Existing graded audits supply identities, selection flags, outcomes and prices.
Each candidate also needs these recorded fields for strict evaluation:

| Column | Contract |
|---|---|
| `best_available_candidate_count` | Expected complete candidate count for this snapshot; must match observed rows |
| `game_start_utc` | Aware scheduled start; existing `game_time_est` is accepted as an Eastern fallback |
| `prediction_generated_at` | Aware prediction time before start and no later than export |
| `model_version` | Version of the complete model/calibration/selector bundle |
| `model_trained_through` | Aware latest data cutoff across all fitted/tuned components, before evaluation and prediction |
| `model_available_at` | Aware bundle availability time after training cutoff and no later than prediction |
| `calibrated_probability` | Oriented win probability conditional on a decided result; configurable with `--probability-column` |
| `probability_semantics` | Must be `win_conditional_on_decision` |
| `market_probability` | Oriented market probability with the same semantics; configurable with `--market-column` |
| `odds_recorded_at` | Aware quote timestamp no later than prediction |
| `odds_american`, `odds_source` | Recorded offered price and source; synthetic/inferred/default/unpriced/fallback labels fail |

Aware timestamps must include a timezone or UTC offset. The live pipeline now
captures these fields automatically when their source evidence is available. Capture
them at prediction generation and retain them through grading. Never reconstruct
missing historical provenance from today's model or a download timestamp. A
pipeline build is not a model version. Ranking scores cannot serve as probabilities.

Old exports lacking evidence produce **insufficient_verified_data** with exclusion
counts. Missing values are never guessed. Each row shows its first failed check;
resolving one gap may reveal another. This behavior is intentional.

## Evaluation rules

- Separate whole Eastern calendar slates at the cutoff. Start times distinguish
  doubleheaders; repository team aliases collapse duplicate feed identities.
- Count exact duplicate downloads once. Conflicting candidates invalidate their
  snapshot. Exclude ambiguous canonical event identities.
- Choose the latest pregame export before examining grades or model eligibility.
  Never substitute an older snapshot because the latest lacks evidence.
- Require every candidate in a snapshot to pass and have WIN, LOSS or PUSH.
  Exclude incomplete pools as a whole; coverage refers to supplied eligible games,
  not the entire league schedule or every possible sportsbook wager.
- Compare the exported selector to the highest market probability on the same
  games. Tie-break by market type then pick text, never by outcome or model rank.
- Separate all selections, approved wagers, explicit passes and unknown approvals.
  League/market segments pair the same games; the baseline may choose another market.
- Simulate one unit per selection at recorded prices. Pushes return zero and
  remain in turnover. Hit rate and probability scoring use decided outcomes only.
  Drawdown sums each slate first. These returns are not actual account P&L.
- Report counts, coverage, calibration bins, Brier score, log loss and ROI.
  Small or selectively observed samples do not establish an advantage.

## Freeze a future evaluation

Once the full model/configuration bundle has a real version, freeze a specification
before the first evaluation slate. Choose a development cutoff of today or later:

```powershell
python scripts/validate_selector.py freeze --train-through YYYY-MM-DD --model-version YOUR-REAL-BUNDLE-VERSION --output output/selector-specification.json
python scripts/validate_selector.py report --audits "downloads/graded_candidate_audit*.csv" --selections "downloads/best_picks_export*.csv" --train-through YYYY-MM-DD --specification output/selector-specification.json --output output/selector-validation
```

Freeze records the actual UTC time, configuration and evaluator code hash. It
refuses to overwrite a specification or backdate a freeze. Save it with the
versioned model bundle. The report checks timing, configuration, code hash and
matching versions. Declarations cannot independently prove no prior inspection,
so the report never claims independently verified out-of-sample performance or
automatically promotes a model. Input SHA-256 hashes are in the companion JSON.

The legacy `evaluate_walk_forward.py` now keeps UTC calendar days together and
labels its output as a chronological diagnostic, not verified out-of-sample evidence.


## Live evidence workflow

1. Run **Master Analysis**. Before analysis, the app fingerprints model artifacts,
   calibration inputs, source code, loaded weight settings and relevant controls.
   The first observation freezes that bundle; unchanged runs reuse its version.
2. After Gemini review, portfolio allocation, recovery and final classification,
   one transaction stores the input candidates, candidate audit and final card.
   Each run gets a unique snapshot ID and a microsecond export run ID. Audit and
   final-card CSVs retain those identifiers. A save failure appears as a warning.
3. Open **Performance Recap** or use **Refresh / Backfill Final Scores**. Saved
   decisions from yesterday load automatically, even without a downloaded CSV.
   Only resolved grades are appended to the evidence store. Exact snapshot and
   matchup IDs link scores to the original prediction. Ambiguous doubleheaders
   and wrong dates remain ungraded.
4. When new score revisions arrive, the app regenerates a separate report for
   each frozen bundle. Performance Recap offers the generated Markdown report
   for download. Its evaluation starts on the Eastern calendar day after that
   bundle was first frozen. Pending and unverifiable candidates remain excluded.

Historical models do not gain an invented historical training date. The captured
`model_trained_through` is a conservative information-cutoff upper bound: the
first time the complete artifact bundle was observed and frozen. The explicit
`training_cutoff_basis` is `frozen_artifact_information_upper_bound`. This excludes
that first day from evaluation. It does not independently certify the training
dataset, feature quality or calibration. Changes create a new version and a new
evaluation window; they do not overwrite the older version's evidence.

Quote timestamps come from the provider's market/book update field, never the
fetch time. A quote must match the exact book, market, side, line and price.
Integer spread/total lines remain `push_semantics_unverified` until explicit
win/push/loss modeling supports conditional probabilities. This may exclude
otherwise valid descriptive picks from the strict probability report.

The default store is `data/prediction_evidence/evidence.sqlite3`. Predictions,
frozen manifests and score revisions are append-only, with database triggers
blocking updates/deletes and SHA-256 checks on saved prediction payloads. Score
corrections append another revision. Derived Markdown/JSON reports live alongside
the database under `reports/` and can be regenerated. Runtime evidence is ignored
by Git. For deployment, set `PARLAYPICKER_EVIDENCE_DIR` to a persistent writable
volume; a disposable container filesystem does not preserve evidence on redeploy.
Keep that directory backed up. The implementation does not provision cloud storage.

Headless commands:

```powershell
# Fetch and grade yesterday's saved decisions; regenerate per-bundle reports.
python scripts/prediction_evidence.py refresh

# Attach explicitly final scores using snapshot_id, matchup_id,
# actual_home_score and actual_away_score (no identity inference).
python scripts/prediction_evidence.py import-scores --scores final-scores.csv

# Export joined evidence and a report for an explicit development cutoff.
python scripts/prediction_evidence.py report --train-through YYYY-MM-DD --output output/live-validation
```

All commands accept `--database PATH` for an alternate evidence database. Scheduled
execution is not installed: refreshing Performance Recap or running the command
triggers the work. Original predictions are never rewritten by grading or reports.


## Development threshold comparisons

In Performance Recap, expand **Development Threshold Comparison**, load saved
evidence, choose the training cutoff and development end date, and generate the
comparison. Approved wagers are the default scope; coverage picks are a separate
exploratory scope. The fixed probability grid reports volume, coverage, odds, hit
rate, flat-stake ROI and a market-only comparison on the same games, by league and
market. Markdown and JSON downloads include verification exclusions; JSON also
includes calibration and uncertainty metrics. No production thresholds change.

```powershell
python scripts/compare_thresholds.py --train-through 2026-09-02 --development-through 2026-09-05 --output output/threshold-comparison
python scripts/prediction_evidence.py status
```

The threshold command also accepts `--audits` CSV paths/globs and `--selections`
final decision exports, or an alternate `--database`. It removes later slates
before validation and snapshot selection. Compare rules only on development data,
freeze a chosen rule, then evaluate untouched future slates. Exploring many
thresholds increases selection bias; displayed intervals are pointwise, not a
multiple-comparison adjustment. A small sample reaching 75% proves neither a
sustainable win rate nor profitability.

**Prediction Evidence Status** is visible before a new analysis run. Download its
JSON to inspect snapshot counts, latest ID/time, payload health and quote binding.
Access to snapshots written by a previous process provides a restart observation;
it does not prove survival of a hosting redeployment. Older snapshots without
process identifiers cannot supply this observation. Hosted durability remains
unverified until separately tested with genuinely persistent storage.


For Streamlit Community Cloud, use the [Google Shared Drive setup guide](evidence-storage-setup.md)
for immutable remote backup and automatic restore. The default local directory
alone remains ephemeral. Evidence Status distinguishes local health from remote
backup status; a successful read-back is required for `synced`.
