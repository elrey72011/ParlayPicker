# NFL, NCAAF, and MLB canonical reconciliation

`scripts/reconcile_prospective_legacy.py` reads the restored NFL, NCAAF, MLB
receipt, and older MLB prospective SQLite stores in read-only mode. It appends
their source artifacts and research facts to `prospective-evidence.sqlite3`
without changing the legacy stores.
Each source artifact retains its exact bytes, immutable source ID, and SHA-256
hash. Each projected fact retains a source key, source hash, market identity,
and integrity-checked payload. A second run with unchanged sources adds no
rows. Changed bytes under the same immutable source ID raise a conflict.

Historical capture facts are stored in `prospective_reconciled_source` and
`prospective_reconciled_fact`. They are **research only**. The real-time
`prospective_event`, `prospective_quote`, `prospective_prediction`, model,
result, and validation tables are not backdated. A historical model name,
probability, score, or close proxy does not satisfy the prospective wager gate.
The dashboard's reconciled counts are separate from canonical prospective
predictions and have a $0 stake.

The JSON report separates source records, receipts, games, market rows,
duplicate market identities, settled and pending games, older unpriced MLB
score forecasts, and identity, chronology, model, price, result, and close
blockers. MLB independent settled
event/line units are computed from validated receipt/outcome pairs, without
counting opposite market sides as separate games. A missing local source is
reported as `LOCAL_ABSENT_REMOTE_UNKNOWN`; only an authenticated remote restore
can establish the remote inventory.

For a local dry run after restoring source stores:

```bash
python scripts/reconcile_prospective_legacy.py \
  --source-dir /path/to/restored/evidence \
  --canonical /path/to/restored/evidence/prospective-evidence.sqlite3
```

The canonical database must be backed up and read back after reconciliation.
The scheduled and authenticated readiness flows perform that remote sync.
