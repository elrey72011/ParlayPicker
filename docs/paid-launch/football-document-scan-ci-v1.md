Document scanner diagnostic CI timing successor
==============================================

This change is timing/provenance only. Scanner v4 remains diagnostic only;
credential rejection is unchanged. The authentic guide trigger remains UNKNOWN.

Baseline 6e31a802: run 38764258973 completed all 7,838 application identities
(7,836 passed, two inherited skips, zero failed/setup-error tests). Shard 2 also
uploaded its final artifact but reached the job deadline: cancelled job and failed
aggregate remain failures. This is not evidence of a hung test or missing coverage.

Completed Ubuntu XML and whole-file assignment manifests from runs 38684830215
and 38764258973 supply the advisory costs. They use the same test-source blobs,
requirements and application workflow; the newer guard/runtime is attributable
to the successor and the older run is not represented as the same revision.
Each source records actual tested checkout separately from PR head, XML and
manifest hashes, completion and job-gate outcome. Cancelled progress intervals
are never converted into completed measurements. The cancelled scan job's final
XML is usable because all selected identities, outcomes and upload completed.
Retained install logs show 86 matching package versions and one difference:
Pluggy 1.6.0 in the prior run versus 1.7.0 in the scanner run. No causal effect is
established. These are compatible completed observations, not an identical-environment
performance benchmark; the maximum retained whole-file measurement is used.

The fixed selection rule populates all 36 absent compatible whole-file entries
(including the two new scanner files), and refreshes nine existing entries whose
scheduling cost differs by at least 60 seconds from the completed maximum plus
the existing 15% variance. Both upward and downward corrections are included.
No adjustment is selected to force a desired assignment. Every other entry and
all frozen revision-3 metadata remain unchanged. New metadata is a successor.

The two new files formerly cost 168 fallback seconds; their completed sums are
0.855 and 1.499 seconds, giving 0.98325 and 1.72385 advisory seconds. The older
underestimated slate/NCAAF/NFL scope costs are also corrected. This does not
speed up tests, Git, fixture construction, reconstruction, hashing or validation.
Those subphases are not separately instrumented by these XMLs. No fixture/runtime
optimization or mutable validation cache is introduced.

Completed scan job time outside JUnit testcase sums is 64.01–88.04 seconds,
including installation, collection, reporting, upload and cleanup. The preview
reserves 120 seconds per job separately; it is an estimate, not a guarantee.
New follow-up files use the unchanged scheduler fallback until measured.
The 7,838-identity preview is 1,584.904 advisory test seconds per shard plus
120 seconds overhead, leaving 95.096 seconds estimated headroom. Follow-up
regressions must be included in the final collection/preview before dispatch;
the earlier two-entry-only 36-second forecast is not the release forecast.

Historical twelve-row profile assertions receive an isolated, exact-hash
4ed86b6f3577ba1a9874cc3632889afec45650e857326bda5ffcb3ffd421bc49
predecessor input, without Git history. All original test function/assertion ASTs
are unchanged. Live scheduling and substitution/dirty/staged/committed/resealed
checks continue using current evidence. Closure provenance peels only the exact
successor bytes before its existing exact predecessor check. Unknown inputs fail.

Six whole-file shards, scheduler, workflows, timeouts, network blocking, test
identities, acquisition/model behavior and operational scheduled task pins are
unchanged. The one new final-head cycle must pass all 16 required checks and
reconcile six complete artifacts exactly once before owner review readiness.
