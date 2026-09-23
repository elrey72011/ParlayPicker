# MLB receipt readiness, 2026-09-23

The owner's `mlb-receipt-store-audit.json` and `mlb-settled-training-records.json`
exports were checked with `prepare_rows`, which verifies the saved pregame
receipt hashes, identity, quote and feature timing against each result. All 180
settled market records passed. The store contains 308 receipts across 77 games;
45 games (180 market records) are settled and 128 market receipts are pending.

| Eastern slate | Settled games | Spread rows | Total rows |
| --- | ---: | ---: | ---: |
| Sep 17 | 2 | 4 | 4 |
| Sep 18 | 15 | 30 | 30 |
| Sep 19 | 13 | 26 | 26 |
| Sep 20 | 15 | 30 | 30 |

The trainer requires at least 20 **independent event/line units** per family
for fitting and 20 decided market rows per family in each later validation and
holdout period. Whole Eastern slates cannot be split. With Sep 18 as the last
training slate, only 17 independent units per family are available. With Sep
19 as the last training slate, there is only one later settled slate and no
separate holdout. Thus no training cutoff pair meets even the count floors;
outcome-availability checks can further narrow eligibility. The old audit's
empty `blockers` list was misleading because it counted 90 spread and 90 total
rows in aggregate without testing partitions. The revised audit reports
`no_chronological_split_meets_minimums` and the maximum training capacity with
later evaluation floors.

Normal Streamlit analysis calls receipt collection with
`reconcile_history=False` to avoid delaying the page. It backs up new pregame
receipts but does not fetch finals. A separate daily `MLB receipt
reconciliation` workflow now restores the immutable Shared Drive receipts,
checks up to 100 saved games against fresh MLB final feeds, verifies the new
Drive backup, and uploads a count-only audit. It also has a manual run option.
The workflow needs the existing Drive secrets and
`RESEARCH_SCHEDULER_ENABLED=true`; it starts only after this change reaches the
default branch. A successful run may still leave future or unverifiable games
pending; the audit identifies whether enough settled slates have accrued.

Once a count-feasible split exists, declare the two cutoffs before inspecting
performance and run the existing research trainer on a fresh settled export.
Its output remains a $0 research challenger and historical out-of-sample
diagnostic. A deployed, model-specific calibration, prospective predictions
made after the model exists, closing observations, validated frozen policy and
owner bankroll/exposure settings are still required for approved wagers.
