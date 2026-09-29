# Post-#2362 focused closure ledger

- Ledger date: 2026-09-29 UTC
- Audited main/base: `2c20d42350449bc4d1ecdd888256a893461026a1`
- PR A implementation revision: `17aa3492fee748eca784159f8f970040a9483463`
- PR B implementation revision: `12d7193037f04cc863841eb694a12ee4b1a2049c`
- PR C implementation revision: `45df8f645e71f573f73d103dd85b4cbc5eb956f5`
- Evidence boundary: local tests and GitHub pull-request CI only. No authenticated census, provider refresh, hosted execution, production publication, market/trial activation, live billing/sales enablement, production record, or wager was performed.

## Review packages

| Package | Pull request | Base | Implementation head | State at ledger creation |
|---|---|---|---|---|
| A — census workflow/schema | [#2363](https://github.com/elrey72011/ParlayPicker/pull/2363) | `2c20d42350449bc4d1ecdd888256a893461026a1` | `17aa3492fee748eca784159f8f970040a9483463` | Open, mergeable, all reported checks successful |
| B — trace/probability semantics | [#2364](https://github.com/elrey72011/ParlayPicker/pull/2364) | `2c20d42350449bc4d1ecdd888256a893461026a1` | `12d7193037f04cc863841eb694a12ee4b1a2049c` | Open, mergeable, all reported checks successful |
| C — external verification | [#2365](https://github.com/elrey72011/ParlayPicker/pull/2365) | `2c20d42350449bc4d1ecdd888256a893461026a1` | `45df8f645e71f573f73d103dd85b4cbc5eb956f5` | Open, mergeable, CI running at ledger creation |

None of these pull requests has been merged. Actual landing therefore remains an owner-controlled follow-up.

## Reproductions and corrections

### PR A

- Before: the census workflow used `runner.temp` at job-level `env`, an invalid GitHub context boundary; the canonical reader rejected writer-emitted `prospective_reconciled_source` and `prospective_reconciled_fact` records.
- After: `CENSUS_DIR` is initialized from `RUNNER_TEMP` in an early runner step, workflow definitions are checked with an Actions-aware validator, and a shared canonical registry accepts every actual writer-emitted table while still rejecting unknown/corrupt records.
- Consumer proof: actual canonical writer fixtures round-trip through the census reader; module-form clean-subprocess startup, checkpoint identity, resume, interruption, budget and corruption tests pass.

### PR B

- Before: equal sport/market/selection/price values could associate one game's candidate with another game's output; unsupported explicit probability semantics could enter absent-legacy compatibility; a half-point integer-score market could retain nonzero push mass.
- After: selected candidate ID and diagnostic position are bound to exact available event/run/line/book/quote identity; explicit conflicts never fall back to display matching; unresolved legacy mappings remain `UNKNOWN/UNRESOLVED`; explicit semantics and half-point push contradictions fail closed.
- Consumer proof: collision, doubleheader, reprice, book, quote, reorder and partition cases execute through `build_private_candidate_trace`; probability cases execute through `per_game_board` and `build_package`.
- Numeric control: `.575` conditional win with `.10` push and decimal odds `2.0` remains `.5175/.10/.3825`, break-even `.45`, edge `.0675`, mean EV `.135`, conservative EV `.09`.

### PR C

- Before: local, Netlify and SFTP publication called saved-package preflight without a current-authority map; the paid-launch verifier defaulted to no independent resolver.
- After: each actual publication boundary reads existing activation/exposure or trial authority without mutation, requires exact saved ticket binding, applies the earliest provider/review/authority deadline, and blocks missing, expired, withdrawn, mismatched or unverifiable authority. The canonical production activation verifier remains the final policy boundary, including rejection of synthetic/test identities.
- Hosted evidence now requires a separately configured, out-of-band HMAC-authenticated attestation registry. Missing, in-tree, malformed, duplicate, mismatched or tampered attestations remain blocked; report fields cannot attest to themselves.
- Provider policy remains explicit: a still-valid frozen quote may be used until its existing deadline. The adapter does not fetch, reprice or restamp it.

## A/B/R/Q acceptance matrix

`PASS-LOCAL` proves implemented behavior with local fixtures. It is not production qualification or external authorization.

| ID | Status | Evidence / remaining condition |
|---|---|---|
| A01 | PARTIAL | All PRs pin actual main and preserve owner files; #2363 is open against main. Actual landing is not done and must be checked after owner merge. |
| A02 | PASS-CI | Runtime directory initialization and Actions-aware validation are in #2363; its `workflow-validation` check passed. |
| A03 | PASS-LOCAL | Module-form clean-subprocess startup/configuration/error tests passed. |
| A04 | PASS-LOCAL | Fixtures are emitted by the actual canonical writer for every registered table, including both reconciled tables. |
| A05 | PASS-LOCAL | Legacy reconciled records remain research-only; unknown schema, wrong key/hash and corrupt records remain blocked. |
| A06 | PASS-LOCAL | Source/storage/checksum resume identity, interruption/budget retention and duplicate-read prevention passed. |
| A07 | NOT-RUN-AUTH | No authenticated 12-scope census was dispatched. Requires explicit owner authorization and configured census secrets. |
| B01 | PASS-LOCAL | Different games with identical total/price cannot share output. |
| B02 | PASS-LOCAL | Selected ID binds to its diagnostic position. |
| B03 | PASS-LOCAL | Run/book/quote/line and explicit-ID conflicts block fallback. |
| B04 | PASS-LOCAL | Ambiguous historical rows stay `UNKNOWN/UNRESOLVED` with a reason. |
| B05 | PASS-LOCAL | Actual private-trace consumer keeps per-candidate and package-level status distinct. |
| B06 | PASS-LOCAL | Truly absent legacy half-point data remains compatible only in the existing research route. |
| B07 | PASS-LOCAL | Unsupported explicit probability semantics fail closed. |
| B08 | PASS-LOCAL | Actual public-board path rejects nonzero push on half-point integer-score markets. |
| B09 | PASS-LOCAL | Push-aware numerical control matches the values recorded above. |
| B10 | PASS-LOCAL | Singleton, mixed, reordered and partitioned inputs retain decisions, values and IDs. |
| B11 | PASS-LOCAL | Assertions execute the `per_game_board` to `build_package` consumer route. |
| R01 | PASS-LOCAL | Local, Netlify and SFTP publishers resolve current trusted authority. Missing/expired/wrong binding and synthetic authority are fault-tested. No real activation was created. |
| R02 | PASS-LOCAL | Earliest quote/analysis/start/review/provider/authority deadline controls the release report; source timestamps are not changed. |
| R03 | PASS-LOCAL | The real verifier loads a signed out-of-band registry; tampering and self-declaration remain blocked. No hosted run was performed. |
| R04 | NOT-RUN-AUTH | No fresh provider-backed candidate run or complete private production audit was executed. Requires provider-spend/remote-execution authorization. |
| Q01 | PARTIAL | #2363 and #2364 report green application, production-safety, protected-scope and both PostgreSQL jobs. #2365 CI was running when this ledger was written. No authenticated browser/hosted run occurred. |
| Q02 | PASS | Protected-file diff is empty. No activation, calibration swap, provider spend, publication, billing change or wager occurred. |
| Q03 | PASS | Remaining local, census, model, hosted, commercial, pilot and owner decisions are separated below. |

## Test evidence

- PR A local: 32 focused tests passed; nine workflow definitions validated. GitHub: workflow validation, production safety, protected scope, subscriber PostgreSQL, completion PostgreSQL and both full-suite shards passed.
- PR B local: 91 focused and 139 broader regression tests passed. GitHub: production safety, protected scope, subscriber PostgreSQL, completion PostgreSQL and both full-suite shards passed.
- PR C local final focused suite: 52 passed, covering real publishers, release preflight, hosted attestations, Netlify and public assets.
- PR C local paid-launch/evidence discovery: 20 passed. This directory-level command did not collect the `case_*.py` PostgreSQL files and is not PostgreSQL evidence.
- Exact local `case_*.py` attempt: collection stopped because the local interpreter lacked `authlib`; `PAID_TEST_DATABASE_URL` was not configured. Local PostgreSQL execution is therefore unperformed. The GitHub `subscriber-postgres` and `completion-postgres` service jobs are the required PostgreSQL evidence.
- Pinned local production-safety collection before the final authority-hardening assertion: 563 passed. The final focused suite covering that hardening passed.
- Pinned local broad application attempt: 3,081 passed, 21 failed, 38 subtests passed. The 21 failures were local Windows/environment issues: SQLite files held open during temporary-directory cleanup, a configured developer Streamlit secret, and a Python `tests` package collision. Isolated reruns confirmed the SFTP/speculative suites pass; one pre-existing Windows path-separator assertion remains platform-specific. Linux GitHub full-suite checks are authoritative.
- Local protected-scope execution found zero protected changes, zero existing-test changes and zero runtime shadowing. It returned `SCOPE_GUARD_TOOLING_HASH_MISMATCH` for unchanged guarded files because of Windows line-ending hashing; Linux GitHub protected-scope checks for #2363/#2364 passed, and #2365 was pending at ledger creation.

## Protected-file diff

No files in the protected set changed:

- `app_core/evidence_drive.py`
- `app_core/evidence_remote.py`
- `app_core/performance_spans.py`
- `app_core/public_history.py`
- `app_core/stage_timing.py`
- `app_core/prediction_evidence.py`
- `app/ui/lock_picks.py`
- `tests/test_refresh_lock_performance.py`
- `tests/test_lock_storage_performance.py`
- `tests/test_prediction_evidence.py`
- `scripts/benchmark_refresh_lock_storage.py`

Existing test expectations were not edited; PR A, PR B and PR C add focused tests.

## Remaining blockers by authority domain

- Local implementation: no known PR A/B/C code blocker after final CI; merge conflicts and actual landing must still be checked at merge time.
- Authenticated census: A07 needs explicit workflow-dispatch authorization and the repository's configured read-only secrets. No secrets should be sent in chat.
- Model qualification: no authentic twelve-scope census, frozen cohort evaluation or first-market qualification was produced here; fixture authority is not model evidence.
- Hosted verification: configure the trusted registry and HMAC secret on the authorized execution surface, deploy staging, and run the real subscriber/browser/recovery/load/backup evidence. None was claimed.
- Provider/current run: R04 requires explicit authorization for provider spend and remote execution. A correct zero-wager result is acceptable; full candidate evidence is still required.
- Commercial: sales and live billing remain disabled. Pricing, terms, jurisdictions, support and launch timing remain owner decisions.
- Pilot: the required observed pilot and operational acceptance evidence were not run.
- Owner authorization: the owner must decide whether and in what order to merge #2363, #2364 and #2365, then separately authorize any census dispatch, market/trial activation, staging publication or commercial action.
