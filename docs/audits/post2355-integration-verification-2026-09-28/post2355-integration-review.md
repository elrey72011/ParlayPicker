# Post-#2355 integration verification

## Revision and scope

- Current-main base: `72aba3152a87739107ef2e4e9b35f8c4f263024f` (merged PR #2355).
- Implementation evidence revision: `2eccc880db58b6a8dd731d2d2f4aa25c5c9476de`.
- Owner uncommitted files were not staged, modified, reset, or cleaned.
- Protected performance, storage, lock, publication, subscriber, authentication, and billing dependencies were preserved.

## Closure summary

The three omitted subscriber completion cases now have an additive PostgreSQL workflow that names each exact node ID. The protected paid workflow remains unchanged and continues to own the existing 23-test job and protected-scope check. The new workflow is an enforced pull-request/main check and uploads JUnit evidence.

The static tracer now returns its AST inventory. `static`, `runtime`, and `both` modes validate requested evidence and exit nonzero when it is absent. Runtime records call the real lean/research scorer, stake reconciliation, public export, strict wager filter, parlay consumer, subscriber v2 projection, and release gate. Route status is derived from those calls and assertions. The 12 deterministic fixtures use integer lines for nonzero push mass.

The real lean/board consumer now groups rows by trusted predictor, exact sport, and exact market family. Production use additionally requires conditional probability semantics, explicit per-candidate push support, legal line/push pairing, a schema-v3 production-default artifact, exact scope/predictor match, and an intact same-byte binding. Conditional calibrated output is converted to unconditional win mass before price gating. Rejected or missing contexts retain the upstream value with an explicit consumer status; mixed markets cannot borrow the first row's curve.

No threshold, live artifact, market activation, sales mode, billing mode, production record, or wager changed.

## Acceptance matrix

| ID | Status | Evidence |
|---|---|---|
| N01 | PASS | Base `72aba315…`; implementation `2eccc880…`; owner files preserved. |
| N02 | PASS | Additive exact-node PostgreSQL workflow; protected paid workflow/guard/manifest unchanged. |
| N03 | PENDING_PR_CI | Three exact PostgreSQL node IDs are configured; authoritative run pending. |
| N04 | PENDING_PR_CI | Existing paid, protected-scope, production-safety, and full-suite CI pending. |
| N05 | PASS | Non-null static inventory finds actual `load_calibration` call sites. |
| N06 | PASS | All three modes pass; missing requested static evidence returns nonzero. |
| N07 | PASS | 12 calls each through research, public, strict, parlay, and subscriber consumers. |
| N08 | PASS | Integer-line positive fixtures; half-point/nonzero-push case rejects. |
| N09 | PASS | Existing batch-invariance and tie separation regressions remain green. |
| N10 | PASS | `calibration-consumer-inventory.json` classifies every discovered default loader path. |
| N11 | PASS | Matching nonidentity curve changes the real consumer value from `0.600` to `0.575`. |
| N12 | PASS | Wrong predictor/scope, mixed market, missing context, legacy schema, semantics, push, and mutation reject. |
| N13 | PASS | `0.575` conditional with `0.10` push becomes `0.5175` unconditional before downstream gating. |
| N14 | PASS | Prior PAV/schema/snapshot/cutoff/EV regressions included in the 105-test focused pass. |
| N15 | PASS | Protected-file diff is empty. |
| N16 | PASS | Staging, empirical model, and commercial states are reported separately. |
| N17 | PASS | No activation, live billing, sales enablement, calibration swap, production record, or wager. |

## Verification checkpoint

- Focused integration plus preserved probability/semantics/lean suites: `105 passed`.
- Runtime route trace: `12/12` scopes; each of five consumers invoked 12 times; all assertions passed or verified the expected authority block.
- Full local Windows shards were attempted but are non-authoritative because pre-existing Anaconda/Windows fixture behavior produced broad baseline failures. Exact counts are recorded in `ci-test-execution-manifest.json`; Linux PR CI is required.
- PostgreSQL completion and final combined CI: pending PR execution at the time of this evidence commit.

## Status separation

```text
code_integration: COMPLETE_VERIFIED
required_ci_execution: PENDING_PR_CI
hosted_staging: NOT_RUN
empirical_market_qualification: UNCHANGED_NOT_QUALIFIED_BY_THIS_TASK
commercial_authorization: NOT_GRANTED
live_activation: NOT_PERFORMED
```
