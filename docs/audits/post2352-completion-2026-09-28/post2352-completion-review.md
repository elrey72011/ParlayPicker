# Post-#2352 completion review

## Revision and scope

- Audit/current-main base: `fc43d6492986874df996f8ecb744c73933432681`.
- PR A: [#2353](https://github.com/elrey72011/ParlayPicker/pull/2353), branch `codex/probability-integrity-closure`, head `f4f87cd7ef14fbf47e652dbace7933dd42fad64d`.
- PR A code evidence revision: `51f66aaf81e1302f44b207956a96c7025e1c8acf`.
- PR B: [#2354](https://github.com/elrey72011/ParlayPicker/pull/2354), stacked on PR A, branch `codex/subscriber-journey-completion`, code revision `2d6d6dfcb3cae8e6734450b5464ac6628fc15d75`.
- Open-PR overlap: no active recent PR changes the probability or subscriber files in this handoff. Older open PRs #2141 and #2133 affect Streamlit export/precision areas only.
- Owner uncommitted files were not staged, changed, reset, or cleaned.

## PR A closure

F01 now treats batch compression as telemetry. The unchanged candidate retains its per-row model/healing probability across an isolated call, additions, ordering, and repeated calls; the hash epsilon no longer enters probability or EV. Actual model exceptions remain explicit unavailable values.

F02 uses an explicit schema-v3 production policy. Unknown, malformed, missing, and legacy versions cannot become production accepted. Explicit paths are research-only. Current production acceptance additionally requires matching consumer predictor and exact scope, a valid digest/status/method/chronology, and a complete manifest.

F03 reads bytes once and inspects the parsed in-memory payload. Raw-byte and canonical identities are distinct. A binding over knots, payload, acceptance, and raw hash detects post-load mutation before trusted provenance reuse.

F04 makes the isotonic target conditional on a decided outcome and converts it to unconditional win/push/loss only with supported per-candidate push mass. The specified `.54/.10/.36`, `.18`, `-.09`, and `.09` examples pass through the shared price helper and subscriber v2 contract.

F05 records every included/excluded observation, source/file hash, event, scope, predictor, probability semantics, outcome/availability, weight, and reason. Cutoffs are derived from the rows passed to the fitter, including `0` and `1`; independent events are counted separately from repeated rows.

F06 now includes deterministic execution of the actual batch inference method, calibration/conversion, shared pricing, production gate, and subscriber projection for all 12 required scopes. Static AST inventory remains separately labeled. Fixture execution does not claim model quality or production authority.

## PR B completion

The base-revision reproduction found no checkout, portal, cancellation, logout, alert, result filter, or current-release refresh controls. It also confirmed the result-count-only shell, absent open-page revalidation, and placeholder support link.

The completed browser uses the existing OIDC/session, billing, cancellation, alert, current-release, and result endpoints. A narrow authenticated `/api/v1/offer` adapter exposes only server-owned public price and policy versions plus fail-closed gate reasons. `/api/v1/me` now returns the already-stored alert preference. Entitled results are restricted to the entitlement's product code and joined to the original immutable recommendation payload; JavaScript never recalculates settlement or return.

Current-release cards revalidate every 60 seconds, at expiry/start, on focus, visibility restoration, and reconnect. Offline or failed checks clear actionability. Monotonic request generations and publication times prevent a delayed older response from replacing a newer revision. No premium payload is stored in browser storage.

The Playwright journey exercised sign-in, a duplicate-click-resistant hosted checkout, manual success-URL navigation without entitlement, server-confirmed entitlement, all seven result statuses, corrections, alert update, expiry, offline state, response races, sales-disabled portal/cancellation, mobile layout, keyboard logout, and expired session. Its machine-readable evidence is `subscriber-browser-evidence.json`.

## Verification checkpoint

- Focused probability/provenance suite: `68 passed` on the code evidence revision.
- Runtime scope trace: `12/12` scopes executed; every production route blocked by research-only authority.
- PR A GitHub CI: both full-suite shards, production safety, aggregate full-suite, protected scope, and subscriber PostgreSQL all passed.
- PR B focused local suite: `33 passed`; subscriber asset/collection suite: `4 passed`.
- PR B Playwright journey: `PASS` with one checkout, portal, cancellation, logout, and alert mutation; all seven result statuses rendered; expiry and late-response checks passed.
- PR B application CI at code revision: production safety `563 passed`; shard 1 `1,374 passed, 1,678 deselected, 38 subtests passed`; shard 2 `1,678 passed, 1,374 deselected`; aggregate full-suite passed.
- PR B paid-service CI at code revision: `23 passed`; protected scope passed.
- New PostgreSQL completion cases: `3 collected`, execution `NOT_RUN_EXTERNAL_BLOCKER`. This Windows host has no Docker, PostgreSQL, or WSL. The protected paid workflow lists only its existing cases and was not changed without a separately approved narrow CI exception.
- Full Windows shard 1 checkpoint before the final combined revision: `1607 passed`, `21 failed`, `1420 deselected`. Two task-related provenance failures were corrected and then passed. The remaining failures were Windows-only SQLite temporary-file handle cleanup plus a CLI platform case; authoritative Linux PR CI is required and is not yet reported in this checkpoint.
- Protected dependency diff: none.

## Status separation

```text
probability_implementation: COMPLETE_VERIFIED
artifact_acceptance_integrity: PASS
runtime_probability_path: VERIFIED_FOR_LISTED_PATHS
subscriber_journeys: INCOMPLETE (local browser verified; additive PostgreSQL completion cases not run)
protected_scope: PASS
combined_revision_ci: PASS
hosted_verification: NOT_RUN_EXTERNAL_BLOCKER
empirical_market_qualification: unchanged; no market qualified by this task
live_calibration_replacement: NOT_PERFORMED
market_activation: NOT_PERFORMED
live_sales_and_billing_enablement: NOT_PERFORMED
```
