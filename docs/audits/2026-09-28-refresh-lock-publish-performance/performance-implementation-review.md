# Refresh, lock and publish performance implementation review

## Release status

```text
implementation_tests: PASS
integrity_equivalence: PASS
remote_benchmark: NOT_RUN_EXTERNAL_BLOCKER
hosted_content_verification: NOT_RUN
production_rollout: READY_FOR_REVIEW
```

This work changes storage discovery, immutable-byte reuse, telemetry and recovery behavior. It does not change model features, probability or EV formulas, validation tiers, wager approval, stakes, quote freshness, kickoff rules, exposure controls, or sportsbook behavior. No production lock, publication, wagering, or authenticated cloud benchmark was executed.

## Revisions and scope

- PRD audited revision and actual branch base: `ed0bd408a05064ae1172c5f569a520826789e122`. The two revisions were identical when work began.
- Verified implementation code commit: `b79264c7`.
- Artifact commit: the pull request head containing this report.
- GitHub CI verified commit: `a7daf3ef90cae854ea854f50da0a4e0f6856f092`, run `36477150153`.
- Primary files: `app_core/evidence_drive.py`, `app_core/evidence_remote.py`, `app_core/performance_spans.py`, `app_core/public_history.py`, `app/ui/lock_picks.py`, and `app_core/stage_timing.py`.
- Isolated P2 warning fix: `app_core/prediction_evidence.py`.
- Tests/benchmark: `tests/test_refresh_lock_performance.py`, existing storage/re-lock tests, and `scripts/benchmark_refresh_lock_storage.py`.

## Root cause and implementation

The old evidence restore invoked a complete folder listing independently for six table prefixes. Lock state did the same separately for original locks and removals, then repeated both after saving. The supplied timers defaulted some record counts to zero even when the generic timer did not know a count, and the supplied child restore times exceeded the enclosing stage by 6.271 seconds. That makes the 686.319-second stage sum useful as a bottleneck indicator, but not a correlated end-to-end trace.

`DriveStore` now creates an immutable, operation-scoped complete inventory. One inventory can serve multiple prefixes without downloading unrelated namespaces. Pagination and incomplete-search failures remain fail closed. A fresh inventory is required at each authority boundary; it is not a global TTL.

Immutable bytes use the existing checksum pattern: fresh Drive membership/checksum metadata plus a local SHA-256 match permits reuse. Missing checksums, explicit full verification, local corruption, and new checksums force media reads. Duplicate remote names are all checked; different bytes fail. Cache files are content-addressed, written through a random temporary file and atomically replaced. Cache paths are scoped by database/storage identity and can be deleted and rebuilt without losing authority.

Evidence restore now requests all six table prefixes from one inventory, then preserves table-specific decoding, identity checks, dependency order, conflict handling and the transactional local merge. `restore_once` includes the database file generation, so replacing a database at the same path invalidates the process guard.

Lock state now has a combined `ActiveLockSnapshot` for originals and removals. The pre-write snapshot validates active membership and re-lock tokens. Writes remain create-only and get an exact read-back receipt. Partial batches expose verified and unresolved keys. The UI performs a distinct fresh post-write snapshot and never substitutes only the newly written locks for full history.

If publication fails after lock verification, the UI retains `LOCKED_PUBLICATION_PENDING` and exposes **Retry publication**. Retry obtains another fresh lock/removal snapshot, rebuilds results, and calls publication without calling `lock_picks`. The original package is deep-copied, preserving quote, analysis, decision and lock timestamps.

Structured spans use monotonic duration, UTC evidence time, correlation IDs, nullable counters, scope hashes, cache/download/retry counters and sanitized reason codes. Provider URLs, headers, credentials and raw exception messages are not logged.

## Controlled before/after results

These are three matched fake-provider trials on Python 3.12.14, not authenticated cloud measurements. The fixture had 60 objects on five pages: 36 evidence, 12 lock/removal and 12 unrelated.

| Phase/cache | Before median | After median | Reduction | Listing pages | Status |
|---|---:|---:|---:|---:|---|
| Refresh, cold empty cache | 364.716 ms | 153.733 ms | 57.85% | 30 → 5 | PASS |
| Refresh, warm unchanged | 275.200 ms | 61.995 ms | 77.47% | 30 → 5 | PASS |
| Refresh, warm plus one new object | 278.818 ms | 66.678 ms | 76.09% | 30 → 5 | PASS |
| Lock membership, cold empty cache | 121.427 ms | 80.315 ms | 33.86% | 10 → 5 | PERFORMANCE_TARGET_NOT_MET |
| Lock membership, warm unchanged | 98.765 ms | 55.366 ms | 43.94% | 10 → 5 | PERFORMANCE_TARGET_NOT_MET |
| Lock membership, warm plus one new object | 100.223 ms | 58.735 ms | 41.40% | 10 → 5 | PERFORMANCE_TARGET_NOT_MET |
| Publication control | 20.000 ms | 20.000 ms | 0.00% regression | n/a | PASS |

Refresh request amplification fell from six traversals to one (83.33% fewer listing pages). Each lock membership phase fell from two traversals to one (50% fewer listing pages). Warm unchanged media reads were zero; the new-object trial downloaded exactly one object. Result hashes were identical on both paths and unrelated objects were never downloaded.

The lock wall-time target was not met in this controlled fixture because cold media verification is unchanged and one of two complete listings is still required. Verification was not weakened to manufacture a passing duration. Production latency status remains `NOT_RUN_EXTERNAL_BLOCKER` pending authorized credentials and a read-only matched exercise.

## Verification

- Compilation: PASS for every changed Python module.
- `git diff --check`: PASS.
- Focused storage, evidence, lock, re-lock, recovery, telemetry and warning suite: **87 passed**.
- GitHub CI: production-safety, both full-suite shards and the full-suite aggregate all passed on commit `a7daf3ef90cae854ea854f50da0a4e0f6856f092`.
- Full local suite: **2,976 passed, 22 failed, 38 subtests passed** on the first run. One failure was the intentionally changed lock-discovery expectation and was updated, then passed in the focused run. The remaining 21 local failures reproduce outside the changed paths: 20 Windows `TemporaryDirectory` cleanup failures from open SQLite handles and one assertion affected by locally configured Streamlit secrets. Clean Linux pull-request CI is the release gate.
- Authenticated remote benchmark: `NOT_RUN_EXTERNAL_BLOCKER` (no production credentials used).
- Hosted content verification: `NOT_RUN`.

The acceptance mapping for P01–P34 is in `integrity-equivalence-report.json`. The implementation preserves unresolved-identity, invalid-target, missing-model, stale-quote, empty-board and no-approved-wager blockers; faster storage does not imply a validated betting model.

## Rollout and rollback

The optimized path is enabled by default. Set `PARLAYPICKER_SHARED_INVENTORY=0` to return evidence and public-history reads to the prior per-prefix discovery path. This changes read behavior only; it does not remove or rewrite immutable records.

To rebuild local caches, stop the application and remove only the rebuildable cache directories beneath the configured `PARLAYPICKER_EVIDENCE_DIR`: `evidence.sqlite3.remote-cache/` and `remote-cache/public-history/`. Do not delete the SQLite evidence database or any Drive object. The next read performs fresh discovery and downloads required bytes.

## Audit checklist

- [x] Branch base matched the PRD-audited revision.
- [x] Parent/child spans are correlated and unknown counts remain null.
- [x] Six-table restore uses one complete phase inventory; empty prefixes do not relist.
- [x] Verified bytes are reusable without hiding new, removed, corrupt or conflicting records.
- [x] Database replacement and storage scope changes invalidate process/cache authority.
- [x] Locks and removals share one fresh pre-write snapshot and one distinct post-write snapshot.
- [x] Exact read-back, first-write behavior, re-lock tokens and concurrent changes are preserved.
- [x] Publication retry never writes locks and rebuilds from fresh membership.
- [x] No mutable odds/news or activation/exposure authority received a blanket cache.
- [x] Cold/warm request counts, timings, result hashes and unmet targets are reported.
- [x] Rollback flag is tested; immutable evidence is untouched.
- [x] No wager approval, stake policy or validation threshold changed.
- [ ] Authorized production remote benchmark.
- [ ] Authorized hosted-content verification.
