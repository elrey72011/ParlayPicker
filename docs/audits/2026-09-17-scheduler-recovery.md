# Scheduler run 35247011938 recovery

## Verified failures

The research job failed with MLB:ReadTimeout and NCAAF:stale_frozen_model.
The activation-evidence job succeeded and uploaded ranking-evidence-rebuild.
Public grading reported no errors. Pending games were not the job failure.
The prior error report cannot establish which MLB network request timed out.

## Changes

Drive metadata, listing and content GETs retry transient network/429/5xx failures
at most three times, with one- and two-second backoff. Auth failures, integrity
errors and uncertain uploads are not retried. Scheduler failures now identify
restore, capture/grade, MLB schedule, or backup. Persistent failures still fail.

The saved NCAAF model from September 7 has runtime bd837b0b...220e57.
Comparing its source commit 61838bc2 with current code shows only NFL aliases
added by 0d36bd74 in core/team_mapper.py. The four NCAAF model/feature/identity
files are unchanged. A recovery allowlist pins BOTH exact runtime hashes AND
the original artifact hash. Only that transition appends a new research cohort
with its real creation time, source model ID and review reason. The new cohort
must be backed up before capture; old predictions and model IDs are untouched.
Any other runtime or artifact still fails closed. This is not sport validation
or activation. The allowlist targets the verified Linux checkout bytes used by
Actions; a different checkout fingerprint remains blocked.

## Evidence remains unvalidated

The latest available immutable Drive snapshot, 2026-09-17T16:21:45.254176+00:00,
contains 64 candidates: 36 MLB, 20 WNBA, 4 NFL and 4 NCAAF. All have verified
quote binding, but all lack model_version, model_trained_through,
calibration_version, selection_policy_version, identity_verified,
conservative_probability and evidence_version. Football rows also lack slate_id.
These are not merely legacy exclusions. A successful collector cannot manufacture
these missing facts or turn the configured blend into a validated trained model.

The report now separates latest producer exclusions by sport from cumulative
legacy exclusions. Existing strict admission gates and active ranking artifacts
are unchanged. Remaining requirements are a real trained Spread/Total producer,
its chronological calibration artifact, verified event identity, provider season/
week for football, and a frozen sport-specific evidence plan. Prospective model
recovery uses a separate paper-research store and does not supply those production
facts. No non-zero authority is enabled by this patch.

## Verification and rollout

Focused scheduler, Drive integrity, model recovery and ranking-report tests cover
transient recovery, persistent failure, auth rejection, no upload retry, unknown
model rejection, idempotence and immutable history. Run the workflow on the
merged commit to verify actual provider/storage availability and see the new
latest_producer_health report. No production run has been claimed successful
based on these local tests.
