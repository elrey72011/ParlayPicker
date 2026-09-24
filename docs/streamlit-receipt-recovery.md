# Streamlit receipt recovery crash, 24 September 2026

The owner-supplied Streamlit log starts a game analysis and then reports two
full MLB receipt-cache reads: 1,483 record objects and 71 manifests. The next
entry is a failed `/healthz` request with a connection reset. There is no Python
traceback in that log, so the exact process-exit mechanism is unconfirmed.
Full-store recovery in the interactive request is the likely resource trigger.

The interactive pipeline now reads live odds without restoring or writing the
entire MLB receipt archive. It labels receipt capture
`DEFERRED_TO_SCHEDULED_CAPTURE`, reports zero newly created receipts, and grants
no remote-backup or wager authority. The readiness page likewise reads local
market evidence only and labels remote counts unknown. Its catch-up and backup
restore controls no longer execute full-store operations in Streamlit.

The dedicated [MLB receipt reconciliation workflow](../.github/workflows/mlb-receipt-reconciliation.yml)
runs at 12:30 and 18:30 UTC when `RESEARCH_SCHEDULER_ENABLED` is enabled. With
the repository Actions secrets `PARLAYPICKER_DRIVE_FOLDER_ID`,
`PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`, and `ODDS_API_KEY`, it restores the
authenticated receipt store, fetches the live MLB slate, captures up to 20
pregame feeds, verifies a backup before reconciling results, then verifies the
final backup and publishes its audit. Missing credentials or a failed
capture/backup cause a classified job failure; they never authorize a wager.

This change keeps the app responsive by deferring receipt collection. New
interactive MLB rows do not inherit the scheduled runner's receipts; any
receipt-dependent approval stays blocked until a separately verified binding
path exists. The scheduled workflow remains the durable research capture path.
