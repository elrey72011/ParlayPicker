# Research synchronization timeout follow-up

Run 35252831602 used the merged scheduler-recovery commit. Research exceeded
its 14-minute job limit without a final report. Activation-evidence completed
and uploaded the expected latest_producer_health diagnostics. The absent
research progress logs prevent assigning every minute to a specific operation.

Inspection found full archive restore plus per-record existence queries and
read-back on every research backup, including backups after individual events.
This repeats growing remote work many times in one invocation.

Research stores now share a synchronization helper. A fresh scheduler invocation
restores once per sport, using the existing bounded bulk Drive reader. It verifies
remote hashes and local canonical IDs. Subsequent backups upload only records
not already verified in this invocation; each upload still receives read-back
verification. Failed reads/writes are not marked verified. Independent manual
syncs and new scheduler runs restore afresh. No cache is persisted across runs.

Unbuffered stage messages report restore/backup start and completion, record
counts, sport capture/grade stages, MLB capture starts and public grading start.
No credentials or provider exception text are logged. The timeout is unchanged.
This removes a concrete repeated-I/O bottleneck; only a deployed workflow can
establish actual end-to-end duration. Capture rules, frozen model fingerprints,
historical records and wager authority are unchanged.
