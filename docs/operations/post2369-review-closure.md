# Post-2369 two-comment closure

Merged application-tooling base: 98015f64c6ca3f5df6e68ce0acbb27a8d0ecbd36.
Separate application dependency: 7c4fe71c7b9bd1a7ae73f8ba04b5e9a79d720eea.
This is a new closure record. Earlier reports/proposals remain historical.

## Corrections

- P1 4158597559: remove redundant finally publication; close the response,
  preserve the primary exception and journal details, retain sanitized secondary
  close errors, and propagate standalone close failures.
- P2 4158597568: validate the current original-driver SHA256 and bind it to the
  verified historical v2 addendum before successor creation or remote activity.

Only guarded_session_factory and inspect_linked_predecessor change in the
operations driver. Publication, journal/prefix, auth/session lifecycle, source
checks, accepted payload verification, allocation/scientific code and cumulative
policies remain unchanged. No workflow/ignore/application edit is needed.

## Executable acceptance mapping

| ID | Evidence |
|---|---|
| C01 | Actual-main/overlap inspection, private preservation ledger and pinned application source/tree |
| C02 | test_C02_actual_merged_adapter_restarts_publication: actual baseline adapter + real writer + fake HTTP; 16 replacement attempts, one GET/body charge |
| C03 | Two test_C03 cases: eight attempts once; primary fields reach reporter; response closes even when close itself fails |
| C04 | Three test_C04 cases: ordinary success, unrelated primary error and standalone close error; no cleanup publication/request/charge replay or batch admission |
| C05 | test_C05_C06: valid actual-writer v1/v2 ancestry and v3 link; otherwise valid conflicting v3 hash rejected with its recalculated outer approval |
| C06 | Same test: missing, null, short, nonhex and wrong-type fields rejected explicitly before creation/copy/network |
| C07 | Runner asserts original 82 and mirror 97 collections, then collects 104; Windows all cases, Linux three existing Windows-only skips |
| C08 | Final-revision application/subscriber/offline CI metadata and downloaded sanitized artifacts, not expected outcomes |
| C09 | Original review-thread replies link correction commit, regression lines and final run; resolution requires substantive correction |
| C10 | Scoped follow-up PR with exact base/head and artifact hash registers |
| C11 | Both stopped private incidents, original inputs/proposals and owner files hashed privately; no private runtime data committed |
| C12 | New private proposal binds corrected driver/tooling, preserves prior proposal and all cumulative limits; remains UNAPPROVED |

## Offline commands

Use a separate clean checkout of the pinned application and existing test dependencies.

python -B -X utf8 tests/qualification_ops/review_closure_suite.py --run-directory <new-private-test-directory>
python -B -X utf8 tests/qualification_ops/run_offline.py --output-directory <different-new-private-test-directory>

Set PARLAYPICKER_QUALIFICATION_APPLICATION_CHECKOUT to that clean pinned checkout.
The runner removes real secure environment inputs and audit hooks deny sockets.
CI retains collection.json, combined.json and combined.xml only.

The new correct-lineage test pauses before v3 workers; it does not claim real
capture or model readiness. Existing M12 covers the full synthetic pipeline.
All synthetic measurements remain distinct from real acquisition evidence.

## Resource and identity boundaries

Keep 22/40 OAuth (18 remain), 15,204/90,000 GET, 140,811,321 observed body bytes
and 3,695.718 charged seconds. Five initiated slices leave at most three under
the original eight. Preserve per-slice 5,000 objects/500,000,000 media bytes/
2,700 processing seconds/3,000 wall seconds; aggregate 4,000,000,000 before-batch
threshold/5,000,000,000 safety stop; 24,000 capture/27,000 whole wall; 900 seconds
each assembly/assessment; 12,000,000,000 working disk/14,000,000,000 initial free
disk/2,000,000,000 accepted database. Object/batch/request limits are unchanged.
Existing soft stopping/OS-call limitations remain; no runtime forecast follows
from the remaining object count.

No real requests, incident edits, recovery, models, activation, deployment,
publication, billing, wagers or merge are authorized by this record.
