# MLB live model connection (research only)

The live Odds API collector now returns this run's verified pregame receipts by
market, in addition to retaining the immutable first receipt used for training.
The receipts travel through live ingestion and candidate expansion into the
existing opt-in challenger adapter. Repeated refreshes use the current observed
quote for inference without rewriting the first training receipt.

The adapter requires a verified MLB/Odds API identity mapping, matching team IDs,
start, market, line, sportsbook and decimal price, a pregame receipt and a quote
observed within 30 minutes. Feed-labeled candidates use the existing exact quote
binder to resolve the book. Missing, conflicting or ambiguous facts fail closed.
The adapter saves the actual input receipt inside mlb_challenger_result alongside
its hash, model provenance and probability, which the existing private snapshot
projection retains. Baseline probabilities and wager authority are unchanged.

## Deployment

1. Collect and retain real pregame receipts with the existing collector; reconcile
   final results and export settled records. Historical reconstructed observations
   and synthetic test fixtures are not training evidence.
2. Use scripts/train_mlb_spread_total_model.py with the real dataset, a source label,
   and predeclared chronological train-through and validation-through dates.
   Existing minimum sample and holdout requirements remain unchanged.
3. Deploy the resulting version directory containing manifest.json and
   estimators.json to the application filesystem. Set the environment variable
   PARLAYPICKER_MLB_CHALLENGER_MODEL to that exact directory. Restart the app.
4. Refresh picks before kickoff. Inspect mlb_challenger_status_counts in pipeline
   diagnostics and mlb_challenger_result in the saved private candidate snapshot.
   RESEARCH means verified research inference; NOT_CONFIGURED means no model
   path; ARTIFACT_UNVERIFIED means artifact validation failed;
   PREGAME_EVIDENCE_UNAVAILABLE means the required receipt/binding was unavailable.
5. Compare future predictions against appended outcomes. Model-specific calibration,
   sport validation, owner exposure configuration and explicit activation are
   separate requirements before a non-zero wager recommendation.

No real model manifest was present in the local models directory during this
implementation. No trained artifact or validation result was fabricated and no
production model configuration was changed. The pregame receipt database still
requires persistent storage/backup as documented by its collector; passing a
receipt to inference does not establish deployment durability for that database.

Verification covers live ingestion/expansion to exact receipt binding, actual
fixture-trained inference, mismatched providers/teams/markets/lines/books/prices,
stale observations, immutable first receipts and retained $0 research authority.
