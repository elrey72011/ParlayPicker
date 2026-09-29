# Current wagers operator summary

Evidence captured on 2026-09-29 against repository base `fd81b534dce193efa95ce910a447a9d6e27feb92`. This is a local implementation and a read-only replay of the recovered public package. It is not a provider refresh, model qualification, staging publication, market activation, or wager.

## Operator answer

The saved board had no wagers before browser expiry. Its 11 Overall rows were all saved as `PASS` research rows, not `APPROVED` or `TRIAL`, and every saved EV was negative. The recorded public-row reasons were:

- 7 `model EV is not positive`;
- 2 `Alternative selection; has not passed final wager and portfolio checks; estimated EV is not positive`;
- 2 `missing push probability`.

The older frozen diagnostic schema separately recorded `STRICT_TRACE_UNAVAILABLE` for all 11 selected rows. The replay preserves that immutable diagnostic instead of rewriting it under current code. It also reports the public row's recorded qualification reason so the original economic/semantic rejection is not hidden by later expiry.

At the screenshot observation time (`2026-09-29T16:01:29Z`), all 11 selected Overall rows had `QUOTE_EXPIRED` as the current primary reason and all 11 also had `ANALYSIS_EXPIRED`. No saved actionable row existed to withdraw. The supported conclusion is therefore a combination of original non-qualification and later expiry after publication—not a claim that stale timestamps caused otherwise eligible wagers to disappear.

Whether a fully qualifying upstream alternative was omitted is **UNKNOWN**. The public package retained 11 selected outputs but not the complete candidate audit. No September 29 candidate-audit file was available in the supplied downloads. `all_market_candidate_count` remains `null`, every unavailable funnel stage is labeled `UNAVAILABLE`, and the empty candidate trace is not presented as proof that no alternative existed.

## Recovered identity and timing

| Evidence | Value |
|---|---|
| Public package SHA-256 / board hash | `6b273154a814c9a6b369f69bd5527c4ce74bf553c7984ce14b4f8c8d6b6f79e6` |
| Publication-version SHA-256 | `0f398bd6bbac0be6c31b8575ae710f0674d7a731d21a140386ffc76c655e68a3` |
| Deployed source revision | `7600b449fea5936d143717a2e32a0c4363dabace` (`source_git_dirty=false`) |
| Current implementation base | `fd81b534dce193efa95ce910a447a9d6e27feb92` |
| Package built | `2026-09-29T15:36:20.938909Z` |
| Publication marker | `2026-09-29T15:38:32.370774Z` |
| Screenshot observation | `2026-09-29T16:01:29Z` |
| Quote timestamps | `15:20:28Z`, `15:20:33Z`, or `15:20:44Z` |
| Analysis timestamp | `2026-09-29T15:21:45.277641Z` |
| Quote age at publication | 1,068.371–1,084.371 seconds (17:48.371–18:04.371) |
| Analysis age at publication | 1,007.093 seconds (16:47.093) |
| Remaining validity at publication | 715.629–731.629 seconds (11:55.629–12:11.629) |
| Build-to-publication-marker interval | 131.432 seconds; this is an interval, not a measured compute duration |
| Publication-to-observation interval | 1,376.629 seconds (22:56.629) |
| Quote age at observation | 2,445–2,461 seconds (40:45–41:01) |
| Analysis age at observation | 2,383.722 seconds (39:43.722) |
| Remaining validity at observation | -661 to -645 seconds; expiry occurred after publication |
| Review/upload/host-verification/notification durations | Unavailable; not inferred from cohort ages |

Diagnostic outcomes: `EXPIRED_AFTER_PUBLICATION` and `INSUFFICIENT_SOURCE_EVIDENCE_TO_RECONSTRUCT`. `NO_QUALIFYING_PRICE_IN_EVALUATED_CANDIDATES` is established only for the 11 selected rows, not the unavailable full candidate set.

## Implementation

- Added a private, deterministic candidate-to-output trace with exact run/event/market/selection/line/book/quote/price identity, exact-duplicate removal, separate candidate/game counts, pass/block/unknown/not-executed stages, original/current blockers, and a local replay CLI. It performs no provider calls or authority writes and never enters the public package.
- Added modern release preflight at local, Netlify, and SFTP boundaries plus hosted observation. Current schema-v5/frozen-authority content rechecks start, quote/analysis TTL, exact selection/market/line/price/book/timestamps, saved authority/allocation, and optional current authority. Legacy schema-v1/v2 packages remain history-only under existing browser usability behavior.
- Preserved saved producer reasons before later expiry and retained immutable selected-diagnostic v1 packages through an explicit legacy validator path.
- Reused `core.price_value`, `decimal_price`, and probability-semantic conversion for probability-first research display. Only a provable half-point/no-push legacy row may use the compatibility path; unsupported semantics remain non-actionable.
- Added owner-only trace display/download to the existing publish panel and added elapsed-expiry states to hosted reconciliation. No storage engine, lock, billing, subscriber, calibration, threshold, or market-activation architecture was changed.

## Local verification

- Focused current-wagers/publication suite: **108 passed**.
- Dedicated probability/pricing/calibration/subscriber-contract/trial/publication suite: **269 passed**.
- Production-safety workflow command: **563 passed** after rerunning with a repository-local pytest temp directory. The first attempt recorded 475 passes and 88 Windows system-temp ACL setup errors; it had no assertion failures and is superseded by the clean run.
- Public browser journey: **PASS** (`live updates, fallback/retry, state, pagination, mobile layouts and stale approval`).
- Subscriber browser journey: **PASS** with one checkout, portal, cancellation, logout, and alert call; 11 current-feed reads; five results reads; all seven result states; late-response suppression; open-page expiry clearing; and no premium leakage.
- Two local full-suite shards were attempted concurrently and are not final evidence. They exposed two branch regressions, both corrected and covered above. Remaining failures were existing Windows-only SQLite cleanup handles, Windows path-separator assumptions, configured local Streamlit secrets, and an isolated launcher whose CI dependency installation is absent from the bundled interpreter. Final Linux PR workflows are the authoritative full-suite, protected-scope, and PostgreSQL evidence.

## Protected scope and effects

Protected performance/storage/lock paths changed: **none**.

`app_core/evidence_drive.py`, `app_core/evidence_remote.py`, `app_core/performance_spans.py`, `app_core/public_history.py`, `app_core/stage_timing.py`, `app_core/prediction_evidence.py`, `app/ui/lock_picks.py`, their protected tests, and the benchmark script are unchanged. Existing test expectations are unchanged; the only test file added is `tests/test_current_wagers_trace_and_release.py`.

The local scope guard reports no protected changes, no protected local changes, no existing-test changes, no scope-guard edits, and no runtime shadowing. Its only local failure is the already documented Windows CRLF working-tree mismatch for the guard/workflow byte hashes. Linux PR CI is authoritative.

No sales, live billing, markets, trials, production authority, production records, calibration replacement, provider spend, remote mutation, public publication, or wager was enabled or performed.

## Remaining external dependencies

1. **Historical evidence owner:** provide the exact September 29 full candidate-audit artifact, identified by run/package/hash, if omission of a qualifying alternative must be answered historically.
2. **Provider/run owner:** explicitly authorize a fresh provider-backed run (and quota/spend) if a current real candidate funnel is required. No such run was performed.
3. **Model owner:** supply current model/calibration qualification and exact market-scope evidence. A passing fixture is not qualification.
4. **Census owner:** execute or reference a current authenticated six-sport/twelve-market census. This replay does not reuse old counts as current readiness.
5. **Staging owner:** provide authorized staging credentials/target and permission to deploy if hosted staging verification is required. No staging or production publication was attempted.
6. **Repository CI:** final PR-head application shards, production safety, paid/subscriber PostgreSQL, completion PostgreSQL, and protected-scope checks must pass before merge.

Completion label: `LOCAL_IMPLEMENTED_AND_TESTED` plus `BLOCKED_WITH_REPRODUCIBLE_DEPENDENCY` for historical full-funnel reconstruction and external operational evidence.
