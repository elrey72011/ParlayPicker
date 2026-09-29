# Post-#2357 closure implementation and operations report

Date: 2026-09-29 UTC

Audited main: `117597cae1786838da450bcbbf28b51c11e49b30`

PR A head: `ee3e51bc13806179c6edf27422a31ca0bda619ff`

PR B head / PR C base: `b85c70202a6ebe7dec603dffe2d53f9b9b3ac529`

This report is new evidence for the post-#2357 work. It does not replace or
rewrite a historical release, calibration, launch report, or prior audit.

## Reviewable package order

1. PR #2358, `codex/post2357-row-contracts` into `main`: row-isolated price
   contracts. All GitHub application, production-safety, paid, protected-scope,
   and subscriber-completion checks passed at the head revision above.
2. PR #2359, `codex/post2357-evidence-verifier` into
   `codex/post2357-row-contracts`: evidence-v2 verifier hardening. All GitHub
   application, production-safety, paid, protected-scope, and
   subscriber-completion checks passed at the head revision above.
3. PR C, `codex/post2357-readonly-census` into
   `codex/post2357-evidence-verifier`: bounded read-only census and workflow.
   Its exact head and final GitHub results are recorded on the PR rather than
   predicted in this pre-push report.

Merge in that order. Nothing in these packages auto-merges, activates a
market, changes a live calibration, enables billing or sales, creates a
production record, or places a wager.

## Authenticated run diagnosis

### Research run 36505631987

- Source revision: `72aba3152a87739107ef2e4e9b35f8c4f263024f`.
- Requested sports: MLB, NCAAF and NFL, not all six sports.
- Complete canonical inventory: 31.265 seconds, 32 pages, 31,519 metadata
  items and one listing traversal.
- Canonical verified reads: 3,006.322 seconds, 27,262 objects and
  1,516,591,084 bytes, with zero verified reuse/cache hits.
- The approximately 60-minute job was cancelled after reaching
  MLB/RECEIPT_RECOVERY. The measured dominant cost was the roughly 50-minute
  cold canonical read before that stage; the evidence does not support calling
  this an MLB processing defect.

### Football run 36543473792 and artifact 11024560282

- Source revision: `263cab0e1e4c4e7dc49d4906b43dea857f5f0347`.
- Complete canonical inventory: 52.672 seconds and 31,562 metadata items.
- Canonical verified reads: 3,581.654 seconds, 27,305 objects and
  1,517,299,361 bytes, with zero verified reuse/cache hits. Two read retries
  added two seconds of retry wait.
- The run verified all 263 pending uploads, then correctly ended
  `PARTIAL_FAILURE`; Stage 2 was skipped.
- The retained cycle audit identifies a strict coverage failure, not a
  credential, provider-integrity or timeout claim: NCAAF had 56 quote-due
  games and two without a verified pregame price,
  `ncaaf:cfbd:401864513` and `ncaaf:cfbd:401866433`, producing
  `NCAAF_NO_VERIFIED_PREGAME_PRICE`. NFL completed its requested slate.
- NCAAF provider reconciliation recorded 54 `MATCHED_TARGET`, two
  `OUTSIDE_TARGET_WINDOW`, two `TEAM_IDENTITY_MISMATCH`, and zero unexplained
  provider events or target games. NCAAF training readiness reported 1,679
  ready spread rows, 1,682 ready total rows, and 38 independent manifest
  events for each market. The 77 blocked games per market comprised six
  `NO_VERIFIED_PREGAME_PRICE`, 54 `RESULT_PENDING`, and 17
  `TRAINING_ELIGIBILITY_BLOCKED`.
- NFL reported 16 independent manifest events for spread and total, 812 and
  814 ready rows respectively, and 16 result-pending blocked games per market.
  These historical counts are diagnostic inputs, not a current qualification
  census and not permission to activate either market.

## Read-only census implementation

`app_core/read_only_census.py` scans exactly the current six producer
namespaces plus the canonical prospective namespace. It performs one complete
Drive inventory per run boundary and only verified object reads after that.
It does not import or call database restore, reconciliation, capture, receipt
recovery, plan creation, backup, publication, activation, billing, or wagering
code.

The checkpoint is bound to the storage-scope hash and the exact twelve-scope
contract. Each reusable item is bound to its provider metadata token and
content SHA-256 from a fresh complete inventory. A bad checkpoint digest,
different storage scope, changed metadata/checksum, missing provider checksum,
deleted object, conflicting duplicate, or explicit full-verification request
invalidates reuse. Object parsing verifies content-addressed source keys,
canonical encoding, canonical primary-key identity, embedded source/payload
hashes, and duplicate byte consistency through the existing Drive reader.

Each run is bounded by object count, downloaded bytes and its own deadline.
Round-robin namespace selection prevents one large namespace from silently
starving every sport. A checkpoint and sanitized report are written after
each batch. Partial and blocked states remain explicit, and the process exits
successfully only for a complete twelve-scope census. The new manual workflow
can resume an access-controlled prior Actions artifact; it has only
`actions: read` and `contents: read` permissions and contains no provider API
keys or mutation step.

## Current twelve-scope readiness ledger

No post-#2357 authenticated census has been run from the PR C revision. The
workflow is new on this branch and has not been merged or owner-dispatched.
Therefore real current object, game, model, calibration, cohort, plan and
evaluation values must remain unknown rather than borrowing fixture or old
workflow counts.

| Scope | Current evidence state | Qualification / activation | Exact next blocker |
|---|---|---|---|
| NFL/SPREAD | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NFL/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NCAAF/SPREAD | UNKNOWN — current authenticated census not run | Not qualified / not activated | Complete price/settlement evidence and census |
| NCAAF/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Complete price/settlement evidence and census |
| NBA/SPREAD | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NBA/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NCAAB/SPREAD | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NCAAB/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| MLB/RUN_LINE | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| MLB/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NHL/PUCK_LINE | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |
| NHL/TOTAL | UNKNOWN — current authenticated census not run | Not qualified / not activated | Model owner runs and reviews complete census |

The first launch market cannot honestly be selected yet. Selection follows a
complete census and the already frozen scientific plan; no threshold is
lowered and no holdout is reused to make a market qualify.

## Acceptance matrix

| ID | Status | Evidence |
|---|---|---|
| A01 | PASS | Main anchor verified; owner checkout stayed untouched; PRs are additive and stacked. |
| A02 | PASS | Existing nonzero-push regression retains `.5175/.10/.3825`, BE `.45`, edge `.0675`, EV `.135/.09`. |
| A03 | PASS | Actual lean-consumer singleton/mixed legacy invariance test. |
| A04 | PASS | Explicit contract remains row-local beside legacy and invalid rows. |
| A05 | PASS | Conservative-bound presence and policy are dispatched per row. |
| A06 | PASS | Reorder, partition, repeat and artifact-identity coverage. |
| A07 | PASS | Partial inputs, impossible push, unknown semantics and quote conflicts fail closed. |
| A08 | PASS — code/CI | Export/subscriber projections use the same final values and authority checks; no hosted claim is made. |
| A09 | PASS | Non-default and duplicate index restoration is deterministic. |
| V01 | PASS | Status-only, missing-field and unsupported-schema fixtures are rejected. |
| V02 | PASS | Wrong build/environment, stale data and hash/reference mismatches are rejected. |
| V03 | PASS | Fixture/skipped journeys cannot satisfy hosted success. |
| V04 | PASS | Pilot interval and load evidence require actual execution and valid observation times. |
| V05 | PASS | Valid isolated evidence-v2 fixtures pass only structural validation. |
| V06 | PASS | Verifier is read-only and grants no live authority. |
| O01 | PASS | Both run diagnoses above are tied to logs and the nine retained diagnostics. |
| O02 | PASS — local | Tests expose mutation methods as traps; census invokes only inventory and verified reads. |
| O03 | PASS — implementation / EXTERNAL NOT RUN | All twelve scopes always receive complete, partial or blocked output. Current authenticated output is pending. |
| O04 | PASS — implementation / EXTERNAL NOT RUN | Raw objects, records, unique events and direct independent eligible games are separate fields. |
| O05 | PASS — local | Resume, full verify, tamper, deletion, checksum/metadata, duplicate and scope invalidation tests. |
| O06 | PASS — local | Batch checkpoints resume without duplicate reads; sanitized terminal reports survive controlled failure. |
| O07 | NOT RUN — external | No owner-dispatched authenticated PR C measurement exists yet. Old cold measurements are retained above, not relabeled. |
| O08 | PASS — contract / EXTERNAL UNKNOWN | Output records actual IDs/bindings/plans/metrics when present and a precise UNKNOWN reason otherwise. |
| R01 | PARTIAL | PR A and B final GitHub suites passed. PR C combined GitHub CI is pending push; local combined regression is recorded in the PR. |
| R02 | PASS | No protected file changed. New module, script, workflow, test and this new report are additive. |
| R03 | PASS | Staging, empirical, commercial, pilot and owner authority remain separate below. |

## Protected-file diff

Zero protected files are changed by PR C. There is no scope exception. On
Windows, the local guard can report its known CRLF self-hash mismatch for the
guard script and paid workflow even while reporting zero protected changes;
the authoritative Linux protected-scope job remains the required result.

## Remaining blockers by owner

- Deployment owner / Robert: identify the authorized staging host/domain and
  configuration location, deploy the existing API, worker, PostgreSQL and
  subscriber shell, then record source/deployed/served identities.
- Identity, payment-sandbox, database and email owners: configure their test
  resources and run the real OIDC, Stripe sandbox, entitlement, portal,
  cancellation, expiry, results, alert and denial journeys. Live billing and
  sales remain disabled.
- Model owner: after the stacked PRs merge, dispatch and resume the read-only
  census until all twelve scopes are terminal; review actual model,
  calibration, cohort, plan and untouched-evaluation evidence; select a first
  market only if it qualifies under the existing plan.
- Operations owner: execute isolated restore, worker restart, rollback,
  notification recovery, publication retry and hosted load/performance tests.
- Commercial owner: decide selling entity, offer, pricing/no-pick handling,
  data/source permissions, processor support, reviewed policies/claims,
  support and incident ownership.
- Pilot owner: run a genuinely observed pilot; current state is `NOT_STARTED`
  with zero evidenced pilot days/slates.
- Repository owner: issue a separate go-live decision for the exact reviewed
  build, environment, product, market and cohort. No such approval exists.
