# Post-#2352 completion review

## Revision and scope

- Audit/current-main base: `fc43d6492986874df996f8ecb744c73933432681`.
- PR A branch: `codex/probability-integrity-closure`.
- PR A code evidence revision: `51f66aaf81e1302f44b207956a96c7025e1c8acf`.
- PR B: pending as a stacked branch/PR at this checkpoint.
- Open-PR overlap: no active recent PR changes the probability or subscriber files in this handoff. Older open PRs #2141 and #2133 affect Streamlit export/precision areas only.
- Owner uncommitted files were not staged, changed, reset, or cleaned.

## PR A closure

F01 now treats batch compression as telemetry. The unchanged candidate retains its per-row model/healing probability across an isolated call, additions, ordering, and repeated calls; the hash epsilon no longer enters probability or EV. Actual model exceptions remain explicit unavailable values.

F02 uses an explicit schema-v3 production policy. Unknown, malformed, missing, and legacy versions cannot become production accepted. Explicit paths are research-only. Current production acceptance additionally requires matching consumer predictor and exact scope, a valid digest/status/method/chronology, and a complete manifest.

F03 reads bytes once and inspects the parsed in-memory payload. Raw-byte and canonical identities are distinct. A binding over knots, payload, acceptance, and raw hash detects post-load mutation before trusted provenance reuse.

F04 makes the isotonic target conditional on a decided outcome and converts it to unconditional win/push/loss only with supported per-candidate push mass. The specified `.54/.10/.36`, `.18`, `-.09`, and `.09` examples pass through the shared price helper and subscriber v2 contract.

F05 records every included/excluded observation, source/file hash, event, scope, predictor, probability semantics, outcome/availability, weight, and reason. Cutoffs are derived from the rows passed to the fitter, including `0` and `1`; independent events are counted separately from repeated rows.

F06 now includes deterministic execution of the actual batch inference method, calibration/conversion, shared pricing, production gate, and subscriber projection for all 12 required scopes. Static AST inventory remains separately labeled. Fixture execution does not claim model quality or production authority.

## Verification checkpoint

- Focused probability/provenance suite: `68 passed` on the code evidence revision.
- Runtime scope trace: `12/12` scopes executed; every production route blocked by research-only authority.
- Full Windows shard 1 checkpoint before the final combined revision: `1607 passed`, `21 failed`, `1420 deselected`. Two task-related provenance failures were corrected and then passed. The remaining failures were Windows-only SQLite temporary-file handle cleanup plus a CLI platform case; authoritative Linux PR CI is required and is not yet reported in this checkpoint.
- Protected dependency diff: none.

## Status separation

```text
probability_implementation: COMPLETE_VERIFIED
artifact_acceptance_integrity: PASS
runtime_probability_path: VERIFIED_FOR_LISTED_PATHS
subscriber_journeys: INCOMPLETE
protected_scope: PASS_BY_DIFF; AUTHORITATIVE_CI_PENDING
combined_revision_ci: NOT_RUN
hosted_verification: NOT_RUN_EXTERNAL_BLOCKER
empirical_market_qualification: unchanged; no market qualified by this task
live_calibration_replacement: NOT_PERFORMED
market_activation: NOT_PERFORMED
live_sales_and_billing_enablement: NOT_PERFORMED
```
