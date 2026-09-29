# Post-#2360 main carry-forward and launch ledger

Date: 2026-09-29 UTC

Actual base main: `7600b449fea5936d143717a2e32a0c4363dabace`

Main already contains #2358 through merge
`7600b449fea5936d143717a2e32a0c4363dabace`. The carry-forward branch was
created from that exact revision. It applies only the reviewed #2359 verifier
commit and #2360 census commit before the narrow corrections described below;
it does not replay #2358 or reset main. The final head revision and GitHub CI
are authoritative on the base-main pull request.

## Carry-forward content

- Evidence-v2 schema, validator, documentation and tests from #2359.
- Read-only census module, launcher, workflow, documentation and tests from
  #2360.
- Module-form census launch from the repository root.
- Clean Python 3.12 subprocess coverage with `PYTHONPATH` absent.
- Explicit batch-level byte/deadline semantics and controlled interruption,
  checkpoint-retention and resume coverage.
- Independent hosted-proof boundary: document consistency cannot create
  execution provenance. A trusted out-of-band resolver must provide a bound
  `IndependentVerification` before `hosted_status` can pass.

## Launcher and budget evidence

The audited file-form command reproduced the defect before correction:

```text
python -u scripts/run_read_only_census.py --help
exit 1: ModuleNotFoundError: No module named 'app_core'
```

The corrected workflow command is:

```text
python -u -m scripts.run_read_only_census ...
```

From the repository root under Python 3.12 with `PYTHONPATH` removed, `--help`
exits 0. A fully parsed invocation with missing Drive configuration exits 1
and writes a sanitized blocked report without attempting a provider write.
Invalid arguments exit 2 through argparse.

Object selection is a hard limit. Byte and deadline limits are intentionally
batch-level soft limits and are identified that way in every report. Reads use
eight-object batches. Given the existing 60 MB canonical object maximum, the
maximum documented byte-limit overrun is 480 MB. The deadline is checked
before each batch and relies on existing transport timeouts for in-flight
work. The 2,700-second census deadline leaves a 300-second margin inside the
50-minute workflow job for terminal reporting and artifact upload.

A checkpoint and report are written after every completed batch. Controlled
deadline termination and read failure retain usable sanitized state; resume
reuses only fresh-metadata/checksum-compatible objects and does not duplicate
reads. The upload step now attempts to run under `always()`. Artifact retention
after an infrastructure hard kill remains explicitly `NOT_GUARANTEED` and is
not claimed as tested.

## Evidence semantics

Evidence-v2 still rejects unsupported/status-only, fixture, synthetic, stale,
wrong-build/environment, skipped-scenario, hash-mismatched and tampered
documents. A locally consistent document now reports these independent states:

```text
EVIDENCE_STRUCTURE_VALID
EXECUTION_PROVENANCE_UNVERIFIED
HOSTED_SCENARIOS_UNVERIFIED
```

It remains blocked as hosted proof until an out-of-band resolver binds the
evidence kind, environment, source revision, provider/execution identity,
artifact hash set, scenario set, verification time and attestation identity.
The default command-line verifier has no resolver and therefore cannot convert
self-declared provider/reviewer fields into hosted proof.

## Acceptance ledger

| ID | State | Evidence or remaining action |
|---|---|---|
| M01 | PASS | Actual main and PR destinations checked; owner checkout with six untracked files was not modified. |
| M02 | PASS — branch | #2359/#2360 content is on a branch created from actual main; #2358 is inherited once from main. |
| M03 | PASS — local | Changes are confined to reviewed verifier/census paths, new launcher tests and this new report; protected diff is zero. |
| M04 | PENDING FINAL PR CI | Local focused and production-safety results plus final GitHub checks are recorded on the PR. |
| M05 | PENDING OWNER MERGE | Main inclusion and post-merge CI cannot be claimed before owner review/merge. |
| L01 | PASS — local | Exact module-form help launch succeeds in a clean Python 3.12 subprocess without `PYTHONPATH`. |
| L02 | PASS — local | Help, invalid argument and missing-configuration outputs are tested; no provider write path is invoked. |
| C01 | PASS — local | Eight-object byte overrun and deadline boundaries are explicit and exercised. |
| C02 | PASS — controlled / hard kill unverified | Controlled termination retains checkpoint/report; hard-kill artifact survival remains unverified. |
| C03 | PASS — local | Scope, revision, membership, checksum, deletion, conflict, full-verify and no-duplication resume tests pass. |
| V01 | PASS — local | Evidence-v2 rejection coverage is retained. |
| V02 | PASS — local contract / external unverified | Hosted PASS requires a separately supplied independent resolver result. No real hosted attestation was retrieved. |
| O01 | NOT RUN — authorization required | No current authenticated twelve-scope census was dispatched. |
| O02 | IMPLEMENTED / current values UNKNOWN | All scopes report explicit states and keep objects, events and eligible games separate. |
| O03 | PASS — local | Mutation methods remain trapped; census calls only inventory and verified reads. |
| R01 | PASS — state separation | Hosted, model, commercial, pilot and owner authorization gates remain independent and blocked where unperformed. |

## Current launch ledger

- Authenticated twelve-scope census: `NOT_RUN`; all current model,
  calibration, cohort and metric values remain `UNKNOWN` until an authorized
  pinned-revision dispatch completes.
- Hosted subscriber staging: `NOT_RUN_EXTERNAL_BLOCKER`; actual staging host,
  served revision and customer journeys are not evidenced.
- Model qualification: no first market selected or activated.
- Commercial approval: not granted; sales and live billing remain disabled.
- Pilot: `NOT_STARTED`; no observed pilot interval is claimed.
- Owner go-live authorization: not granted.

## Exact main-landing proof boundary

Before merge, proof consists of a PR whose base is the exact main revision
above, a merge base equal to that revision, a reviewed expected-path diff, and
green final-revision checks. After owner merge, separately fetch main, record
the PR merge commit, prove that merge commit is an ancestor of `origin/main`,
verify the expected path content is present, and inspect the new post-merge CI
runs. A merged flag on a feature-branch PR is not sufficient.

No merge, live billing/sales enablement, market activation, calibration
replacement, production test record, wager, or qualification-threshold change
is performed by this work.
