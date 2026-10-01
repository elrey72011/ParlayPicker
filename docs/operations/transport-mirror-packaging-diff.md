# Transport-mirror source diff and packaging

Base main: 702e9788b61980737921428666d90844b0906b0b, containing merged OAuth repair #2368.
Application dependency remains 7c4fe71c7b9bd1a7ae73f8ba04b5e9a79d720eea.

The historical v2 driver is retained unchanged as previous_oauth_snapshot_acquire.py:
53,084 bytes, SHA256 8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93.
The current driver is a new byte identity, listed in the artifact hash records.
Both baseline and current reusable source use LF; no claim that new code has the old hash.

Changes:
- Unique fsynced same-directory publication, serialized local replacement, finite classified retries.
- Cooperative Windows read handles with delete sharing; non-mutating sharing classification probe.
- Strict journal schema/links/counters, read-only exact-prefix validation (zero/one event only).
- First-error diagnostics independent of mirror loading and safe stderr fallback.
- Explicit original/effective specification and ancestry validation, accepted-payload rehash.
- New-only projection/reconciliation; cumulative initiated-slice, wall and disk bookkeeping.

The original v2 hash assertion is unchanged and now targets its immutable fixture. Functional runtime defaults to the repaired driver. The runner still guards collection of exactly the prior 82 cases, then adds 15 new cases; no prior assertion or acceptance count is removed. Authentication, scientific and existing protected workflow files are unchanged. The existing Windows/Linux workflow invokes the expanded runner without a workflow edit.

Source segments verified byte-equivalent after parsing (line positions change): source_check, anchor, capture, materialize, assess, guarded_session_factory, CaptureAuthState, accepted_objects, load_state, persist_state, run_approved_workers. The new verified_record reader changes handle sharing only; digest/content checks remain.

No private specification/addendum, credentials, raw corpus, real database, incident logs/journals or owner files are tracked. Authentic stopped-state verification and before/after file registers remain private. The public recovery-v3 template is unusable and grants no execution authority.
