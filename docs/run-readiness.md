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


### Quote binding and final line decisions

`quote_verified` reports an exact provider quote match. `line_eligible` separately
reports whether the final line was rejected; it does not approve a wager. A row
can have a verified raw quote and an unresolved final pick after the line/event
safety check. Capture preserves this rejection as `final_line_rejected`, and
readiness and formal validation block it. Historical snapshots are unchanged;
readiness also recognizes their unresolved labels and rejected line sources.

### Push-capable probabilities

Whole-number spreads and totals may settle as PUSH. The report exports
`settlement_rule` and `probability_semantics`. Existing unverified records remain
blocked, including when the problematic candidate was not selected.

The validator accepts explicitly recorded `win_unconditional_with_push` inputs
only with both `push_probability` and `market_push_probability`. Each must be
finite, nonnegative and below one, and win plus push probability must not exceed
one. For scoring it converts each probability using `P(win) / (1 - P(push))`.
Original forecasts remain unchanged. PUSH settles at zero profit and remains in
turnover, but is excluded from binary scoring and decided-wager hit rate.

This support does not estimate push probabilities, relabel historical forecasts,
change production probabilities, or loosen wager approval. The current model
must supply verified push-aware forecasts before its integer-line records can
use this path.
