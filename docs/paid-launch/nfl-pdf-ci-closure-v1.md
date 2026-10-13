# PDF CI closure: fixture isolation and measured scheduling

The original PDF implementation is retained at `f2ea50b77e63554abf0f6aeed55845b326ce6e9b`. Its sealed head `0596b245e5832c7b776808bd1f508f42ef5758fe`, raw logs, artifacts and failures remain retained. Main is `9aceef16d9f79fdb95c9a910b5411e3cd449e5ce`; neither release was merged or rerun for this correction.

## Repository-copy failure

Run 38401018393 shard 5 raised `shutil.Error` at `tests/test_mlb_nhl_research_scope.py:65`: enumerated loose Git objects disappeared before copy. It was a setup error, not an assertion failure. An offline synthetic reproduction used the unchanged session fixture, actual `copytree`/`copy2` and real interleaved `git gc --prune=now`; the same missing-object error occurred. Quiescent copies before and after GC retained HEAD and passed `git fsck --full`. This reproduces a mechanism, not proof that background GC caused the historical run.

Only that fixture's Git helper now uses per-command `gc.autoDetach=false` and `maintenance.auto=false`. Git housekeeping completes before copying; no global setting is changed. Each test still copies the full repository into its own mutable location. Copy errors propagate. All existing dirty, staged, committed, resealed, policy and ancestry assertions remain unchanged. Two added synthetic regressions cover isolated copies, history/integrity, unchanged configuration and propagated copy errors.

## Timeout diagnosis and correction

Shard 2 spent 44 seconds installing dependencies, then 1,762 seconds in the combined collection/test step before cancellation. It continued passing tests at 93%; no hang is established. Reporting/upload did not complete. The NCAAF compatibility progress block spanned 764.247 seconds, but this is an incomplete observation, not completed JUnit timing.

The single authorized diagnostic run 38429225374 compared main and the original PR under pinned Ubuntu/Python 3.12.15 dependencies, identical 16 tests and isolated fixtures. Both completed, without retries, and uploaded raw profiles, XML and progress receipts. Main took 101.744 seconds; the PR took 76.902 seconds. Collection/setup/call were respectively 9.391/11.103/80.857 and 6.957/9.326/60.264 seconds. Repository copying was 0.825/1.051 seconds. The inherited reconstruction stack dominated the bounded comparison; these overlapping cumulative categories must not be added. Different runners and profiler overhead prevent attribution of the 24.842-second difference to the PDF implementation. This evidence does not show a PDF regression.

The existing profile underestimates inherited scope work. The successful #2420 run's completed 118-case compatibility XML sums to 583.735 seconds, versus the current 354.260-second advisory entry. Twenty-two existing scope-file costs with at least 30 seconds of demonstrated underestimate are refreshed from exact-count, completed Ubuntu XML. The existing 15% advisory margin remains an estimate. No cancelled span is promoted to measured timing. Installation, collection, reporting and upload are excluded from testcase sums. Frozen revision 3 and all previous provenance remain unchanged; the new evidence is separate successor metadata.

The scheduler, whole-file assignment, within-file order, five jobs, required workflows, network blocking, assertions and 30-minute limit are unchanged. Redistribution changes assignment only; it does not reduce total test work or guarantee future runtime. The preview has to be reconciled against fresh final-head execution and artifact uploads. Prior passes under another assignment cannot satisfy that gate.

Two frozen metadata-transition tests receive the exact hash-verified predecessor profile in their own synthetic repository. Their original assertions remain intact. The actual scheduler continues reading the current profile. A separate successor regression verifies its changed costs and completed provenance, while the exact reconstruction returns the original historical metadata bytes. This fixture isolation provides no claimed runtime savings and does not rewrite frozen revision 3.

An experimental reconstruction memo extension was not adopted. No new mutable validation cache or guard optimization is included. The PDF custody/extraction/interpretation/owner-review contract, old readers, model mathematics, saved Rams result and all wagering gates are unchanged. No acquisition or authentic inference occurred.

## Exact successor

The refreshed binding records the ordered original implementation and fixture commit as immutable ancestors, the precise final implementation, all changed blobs and reversible predecessor source reconstructions. The next commit contains only the policy seal. Extra commits, mutations, altered ancestry or resealing continue to reject. Raw receipts and private evidence stay local; final check identities, counts, durations, artifacts and readiness are reported in the PR validation receipt.
