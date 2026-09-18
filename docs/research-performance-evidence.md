# Research performance eligibility

The scheduled ranking-evidence report now includes a separate `research_performance` section. The same read-only report is available in Workspace > Diagnostics > Run Readiness Report > Prepare research performance report. The UI uses locally restored evidence; it does not initiate another full remote restore.

This descriptive cohort does not require trained model/calibration versions, conservative probability, or sport wager policy. It does require:
- Hash-verified original snapshots and score revisions, supplied by materialize.
- Explicit saved best-pick selection and both prediction/capture timestamps before game start.
- Sport and event identity, exact quote binding, an original valid line, and quote time no later than prediction.
- A provider-backed settlement with recorded event ID and valid result time.
- Original consensus, with no inference from today's signals.
- Latest saved pregame selection per sport/event chosen before examining results. Ambiguous latest selections are rejected; an unresolved latest selection cannot fall back to an earlier winner.

Pushes are reported separately. Buckets remain sport-specific. Existing records are never rewritten. Missing fields have explicit exclusions.

This report is descriptive, not a calibrated model or an active ranking artifact. It is intentionally not loaded by load_bucket_stats or activation_validation, and cannot alter suggested stakes. Promotion into ranking requires a separately reviewed prospective evaluation; the stale legacy overlay remains excluded until a justified replacement exists. No automatic freshness reset occurs.

Local verification: the local database produced no eligible selected research rows. That is not evidence about the newer deployed store. Download the new report after deployment to inspect its cohort and exclusions.
