# Run readiness

Open **Run Readiness Report** beneath Prediction Evidence Status. Inspect the
current run or load and select an earlier saved snapshot. The report exports
Markdown, JSON, game-level CSV and candidate-level CSV.

Each game has three separate dimensions:

- **Evidence readiness:** identity, complete candidate pool, one selected pick,
  final-card match, pregame timing, model provenance, quote evidence, executable
  price/source and probability semantics. One invalid candidate blocks the pool.
- **Recorded wager decision:** approved, pass, or unknown. A ready evidence pool
  is not a recommendation to place a wager. Conflicting or missing decisions are
  not converted into approvals. Wager reasons retain the exported gate messages.
- **Data warnings:** missing independent model probability, unavailable feature
  timestamps, quote age at capture and whether the game is after the model freeze
  day. These warnings do not change the production thresholds.

Displayed, production, independent-model and market probabilities are separate.
An unavailable production probability is never filled with the displayed one.
The 15-minute quote-age warning is a report diagnostic, not a newly adopted
trading policy. Feature freshness cannot be established without source timestamps.
Run-level stale-schedule warnings are not incorrectly attributed to every game.

This is a pre-grading checklist, not the formal selector validation report. It
does not use outcomes, choose a historical snapshot based on results, train a
model, change any stakes, or prove a win rate. Formal validation still checks
settlement, canonical identity, latest eligible snapshot selection and cutoff
rules. Missing evidence remains visible instead of silently dropping the game.

```powershell
python scripts/run_readiness.py --audit candidate-audit.csv --final best-picks.csv --output output/readiness
python scripts/run_readiness.py --database data/prediction_evidence/evidence.sqlite3 --snapshot-id SNAPSHOT_ID --output output/readiness
```

Without CSV arguments, the command uses the latest saved snapshot, or the explicit
snapshot ID. CSV input hashes accompany the JSON report. No source file is edited.
