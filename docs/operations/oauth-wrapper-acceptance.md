# Offline acceptance mapping

The 82 cases are collected by `tests/qualification_ops/run_offline.py --collect-only`; CI executes the complete runner on the selected PR revision. Windows provides 82 passing cases when successful; Linux explicitly skips the single Windows process-tree case. All transport lifecycle tests use real Google Auth objects with fake HTTP transport and socket denial.

| ID | Executable or preservation evidence |
|---|---|
| A01 | Byte-preserved v1/v2 hash/envelope case; independent private-file preservation report outside commits |
| A02 | `test_A02_old_auth_churn_actual_reader`: real constructor lifecycle fails fifth batch at20 |
| A03 | `test_A03_5000_actual_objects_625_batches`: 5,000 objects,625 batches,one OAuth |
| A04 | `test_A04_actual_supervisor_eight_slices_child`: actual capture-block/assembly/assessment children |
| A05 | `test_A05_27580_objects_virtual_full_duration`; both duration-suite cases |
| A06 | All four `test_A06_*`: expiry,401,generation,retry/failure coordination |
| A07 | `test_A07_nineteen_concurrent_limit`: one new POST,three denials,total20 |
| A08 | `test_A08_A09_recovery_keeps_usage_and_orphans`: 32 accepted/three unledgered; durable69/20/9929120 |
| A09 | A08/A09 case and `test_A09_actual_linked_recovery_supervisor`: zero unamended capacity; linked synthetic recovery |
| A10 | `test_A10_recovery_faults_no_requests`: journal/cache/mirror/commit/source/scope/membership faults |
| A11 | All 64 `FunctionalTests`; incremental-cap/startup-time cases; unchanged real envelope |
| A12 | `test_A12_end_to_end_real_auth_empty_science` plus V07/V08/V10/V11: legitimate empty tables,byte replay,read-only input hashes |
| A13 | 64 functional cases preserved; platform skip recorded rather than counted as pass |
| A14 | Explicit migration diff/artifact hashes; original private preservation outside commits |
| A15 | `test_A15_failure_record_is_sanitized_and_retains_progress`; no key/token/header leakage |
| A16 | Tracked runbook/templates and separately unapproved recovery proposal |

| ID | Repository handoff evidence |
|---|---|
| G01 | Base/current-main verification and byte-identical existing v2 driver |
| G02 | Narrow tools/tests/docs/new-offline-workflow paths only |
| G03 | Collection82; final-revision Linux/Windows execution summaries/JUnit |
| G04 | A03–A07 and duration fault cases |
| G05 | A08–A10 synthetic predecessor and linked successor |
| G06 | Separate clean pinned application checkout; runtime SHA/tree/hash/identity guards unchanged |
| G07 | No authentic config,payload,DB,journal or owner files staged; sanitized summaries only uploaded |
| G08 | Local-to-tracked diff,new portable artifact hashes,executed runbook command |
| G09 | One pushed branch/PR against actual main; final-head workflow checks |
| G10 | No real recovery executed; proposed cumulative40 amendment separately unapproved |
