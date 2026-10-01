# Qualification snapshot tooling: offline validation and separate recovery approval

This utility packages the completed operations-wrapper repair. It does not change application prediction, calibration, publication, subscriber, evidence-storage or qualification policy. A PR, template, hash or execution flag is not execution authorization.

## Revision and evidence identities

The TOOLING revision is the selected PR head/commit containing `tools/qualification/snapshot_acquire_and_assess.py`. Its APPLICATION dependency remains the clean separate checkout at `7c4fe71c7b9bd1a7ae73f8ba04b5e9a79d720eea`, tree `e9d684d0cdf695e3e6207641ab7a1e28eb00e5ea`. Never substitute the tooling SHA, update the historical source fields, copy application files into the tooling tree, or bypass the source/tree/clean-state guard.

The tracked driver is byte-identical to tested local v2: 53,084 bytes; SHA-256 `8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93`. Scoped Git attributes preserve its exact bytes on Windows and Linux. The original failed driver is retained as an offline baseline fixture, 37,141 bytes; SHA-256 `6c01b00da684956f4319017c3b7f08b78eafc5133b74de98697d69fad287de67`. Never invoke that fixture for a real operation.

Runtime evidence identities remain independent: exact census run/artifact, captured as-of, namespace, membership hash, storage scope, checkpoint/report/archive hashes, blocked-operation ID, state/journal/file-register hashes and accepted ledgers. The example templates intentionally contain unusable placeholders; no authentic private specification, incident state or payload is committed.

## Offline validation

Use Python 3.12 and Git with existing dependencies, or install `tests/qualification_ops/requirements.txt` in an isolated test environment. Dependency installation is setup; no dependencies are downloaded by the actual tests. CI separately checks out the pinned application and the PR tooling. It has no Google secrets.

From the tooling repository on Windows, select a clean existing application checkout and a NEW test-output directory:

```powershell
$env:PARLAYPICKER_QUALIFICATION_APPLICATION_CHECKOUT = '<clean separate pinned application checkout>'
python -B -X utf8 tests/qualification_ops/run_offline.py --output-directory '<new private test-output directory>' --collect-only
python -B -X utf8 tests/qualification_ops/run_offline.py --output-directory '<different new private test-output directory>'
$offlineExitCode = $LASTEXITCODE
Write-Host "OFFLINE TEST EXIT CODE: $offlineExitCode"
```

On Linux, set the same environment variable to the separate checkout and run the same Python command. Collection requires exactly 82 cases: 64 functional, 16 actual-auth/recovery and two full-envelope/fault cases. Windows executes the process-tree timeout case; Linux records its explicit platform skip. The output must show actual collection, execution, case results, driver/application/tooling identities and zero real socket attempts. A passing count without those records is insufficient.

Fixtures use the actual application writer/codec, verified parallel reader and Google Auth credentials/AuthorizedSession lifecycle. Signing keys exist only in synthetic process memory. Fake transport replaces HTTP adapter send; socket connections and DNS are denied in parent and children. Tests never require owner credentials, real Drive data, incident files or a production database. The runner strips inherited Google settings from suite child environments. Synthetic acceptance uses lower test-only slice/disk limits where necessary; the example's real envelope is tested separately without widening it.

Only `collection.json`, `combined.json` and `combined.xml` are uploaded by CI. Raw fixtures, SQLite files, synthetic journals, logs and private paths stay outside CI artifacts. Keep real private evidence outside the repository, OneDrive and public reports. No broad ignore exception or force-add is needed.

## Repair behavior

A single supervised capture-block child spans every permitted internal slice. Per-thread request sessions share locked authentication state. Refresh generations coordinate expiry and concurrent 401 responses, including unchanged issuer token text. Bounded Google Auth retries and each OAuth POST/Drive GET remain durably charged. Tokens, private keys and credential JSON are never persisted. Separate assembly and read-only assessment workers make no HTTP requests.

The historical fault reproduced four accepted batch OAuth totals 5/9/13/17 and failure in batch five at the original 20-attempt cap. This mechanism is retained as an executable offline regression. The v2 5,000-object case measures one OAuth POST across 625 accepted batches. Tests also cover 27,580 synthetic objects, all eight slice boundaries, full virtual authentication time, expiry/401 faults, concurrent cap exhaustion and retained failure progress. These simulations do not prove provider throughput or promise real completion.

## Unchanged envelope and soft bounds

One separately approved invocation; at most eight capture slices in the operation chain. Each slice permits 5,000 new objects, 500,000,000 media bytes, 2,700 processing seconds and 3,000 wall seconds. Batch size eight, at most four media workers, 60,000,000 bytes per object. Before another batch, aggregate observed response bodies are checked against 4,000,000,000 bytes; safety stop at 5,000,000,000. At most 90,000 Drive GET attempts and originally 20 OAuth POST attempts. Capture wall 24,000 seconds; assembly 900 and assessment 900; whole operation 27,000 seconds. Working-disk monitored stop 12,000,000,000 bytes; initial free disk prerequisite 14,000,000,000; accepted standalone database at most 2,000,000,000 bytes.

Batch/request/deadline and periodically sampled disk stops are soft monitored bounds, not reservations or OS quotas. In-flight requests may overshoot thresholds; retain actual counters, timings and overruns. Raw-cache size is not cumulative wire transfer. Never reset durable attempt/body usage to an older accepted-state counter.

## Private linked-recovery preparation (no authorization conferred)

1. Preserve the blocked predecessor, original driver/spec, logs, attempt markers, all 51 incident files and incurred usage unchanged. Confirm no matching worker remains; do not delete/reset/overwrite or silently restart.
2. Independently verify original driver/spec hashes, source SHA/tree/clean-state, census archive/file/canonical digests, namespace/membership/storage binding, every state commit, accepted batch seal/content/cache hash and complete transport journal/mirror. Any missing/conflicting evidence blocks recovery.
3. Verify the historical incident: 32 accepted objects in four batches; three cached files without accepted ledgers; 69 GET attempts, 20 OAuth POST attempts, 9,929,120 observed response-body bytes. The lower accepted-state counters are not incurred usage. The three files remain unaccepted; retain them in private provenance and re-download their pinned logical objects under future approval before ledger admission.
4. The unchanged 20-OAuth cap leaves ZERO further OAuth attempts. A separate explicit owner amendment may propose up to 20 additional requests, 40 cumulative. This repository handoff does not approve that amendment. Keep historical counters and the original operation ID in the linked chain.
5. Prepare a hash-bound private addendum for ONE absent exact successor destination. At most seven successor slices plus the prior failed slice; charge the prior 54.171 seconds and predecessor disk bytes against unchanged chain limits. Copy full provenance privately; only 32 accepted cache files enter the active successor cache. Do not materialize a successor or inspect raw payloads into public reports during repository handoff.
6. Select the tested tooling commit and verify the actual driver, unchanged private original spec and separately proposed private addendum hashes. Obtain explicit approval of the selected tooling commit/path, original application/evidence/storage identities, exact private destination, cumulative resource caps and permitted effects. A template is not the approved addendum.
7. Only an authorized owner runs the command in the privately configured process-scoped Windows shell. The Codex process does not inherit that environment. Do not reveal credentials, put them in command history, export GitHub secrets or create new credentials. Closing a parent shell does not guarantee termination of children; verify the matching process tree separately.

The separately hash-approved command form is:

```powershell
& '<reviewed Python>' -B -X utf8 '<selected tooling checkout>/tools/qualification/snapshot_acquire_and_assess.py' `
  --spec '<unchanged private original specification>' `
  --approved-spec-sha256 '<verified original spec SHA256>' `
  --recovery-spec '<separately approved private linked-recovery addendum>' `
  --approved-recovery-sha256 '<verified addendum SHA256>' `
  --approved-driver-sha256 '8d8c629d45626fe64260593ba1a22795d962ba9574ff074043f3b403a240ec93' `
  --execute-approved-operation
$recoveryExitCode = $LASTEXITCODE
Write-Host "RECOVERY EXIT CODE: $recoveryExitCode"
```

Do not execute this placeholder command. No automatic second invocation, retry, census or budget increase is allowed. Stop on rejected prior evidence, identity/hash/source conflict, unexpected failure, no progress, retention failure or resource exhaustion.

## Result interpretation and next tasks

Keep acquisition completion, snapshot acceptance and readiness separate. Exit 0 requires the operation's retained completion evidence, not a model qualification claim. Exit 2 is bounded PARTIAL with retained progress; no automatic continuation. Exit 3 records BLOCKED worker failure; establish the first unsuccessful stage and its sanitized reason from retained records. Some startup guard errors raise before those files exist; never invent a recorded cause from exit code alone.

A complete canonical snapshot can legitimately have zero canonical model/calibration/prediction/validation/review rows and produce blocked readiness. Verify standalone database integrity, all pinned re-encoded bytes and input hash before/after read-only assessment. Do not manufacture absent rows or fit/register/freeze models/plans to complete capture. Keep eight unsupported non-football manifest calculations UNKNOWN. Canonical absence describes only the captured corpus; native MLB/NCAAF research and separate receipt inputs are not canonical production registrations.

If exact-target model lineage, compatible calibration or evaluated-plan bindings are absent, return those specific missing identities/cohorts and a separate bounded scientific work order. Do not infer product-to-plan bindings, matured holdouts or launch authority. No activation, staging, publication, sales/billing, reservations or wagers follows automatically from either PR merge or snapshot acceptance.
