# CI test execution

The CI workflow keeps `production-safety` as a focused check and `full-suite` as the required gate for all application tests. The complete suite runs in five separate GitHub jobs, each with its own checkout and runner filesystem. `scripts/run_ci_tests.py` partitions whole collected files deterministically using the versioned advisory timing profile and retains collection order within each file. Every collected test belongs to exactly one shard. All five jobs must succeed; a failure, cancellation, missing result or unexpectedly empty shard prevents the existing `full-suite` gate from passing. The 30-minute application-job limit and production-safety selection remain unchanged.

New commits cancel superseded runs on the same pull request. Pushes to main retain separate runs. The scheduled research/grade workflow is independent and unchanged.

Each test job prints test identities and the 30 slowest test phases. JUnit XML reports and assignment manifests remain downloadable from the workflow artifacts for seven days, including after a test failure. A cancelled job may not finish/upload its reports; the missing results cannot pass the gate. Dependency installation already uses the pip download cache.

## Advisory timing and assignment contract

`scripts/ci_test_file_costs_v1.json` binds each timing source to its exact run,
attempt, API head, actual checked-out commit and artifact/JUnit SHA256, or a
retained progress-log SHA256. A PR's tested merge is distinct from its branch head.
Completed current main/#2402 file testcase sums are labelled `measured` and
exclude collection, reporting and upload overhead. Older different-head costs
with a 10% variance margin, and 72-case progress-block extrapolations, are
labelled `estimated`. A progress block is not a measured whole-file runtime.
Missing timings remain `missing`. `profiled_tests` is the collection size covered
by a cost; an estimate does not establish completed measurements. Raw reports
and private evidence stay local; the profile contains only aggregate CI costs
and public CI source identities.

Files sort by descending advisory cost, then their normalized repository-relative
path. Each whole file goes to the least-loaded bucket, breaking ties by shard
number. Original pytest collection order is retained after selection. Windows
and POSIX absolute paths normalize against pytest's collection root; profile
paths must be relative. Traversal, portable case collisions, duplicate identities
and invalid/nonfinite/negative timing values fail explicitly.

An unprofiled file costs the greater of 30 seconds and 3.5 seconds per collected
test. Additional tests beyond a profiled file's count add 3.5 seconds each.
The 3.5-second estimate exceeds the observed 224.836/72 = 3.123 seconds per
case scope block; it is a conservative scheduling assumption, not an empirical
upper bound. Unknown files are assigned in full. Timings affect assignment only;
they cannot exclude tests, alter assertions or create scientific/wagering authority.

Every manifest records profile and assignment hashes, all collected node/file
assignments, selected/deselected counts, timing labels, completion identities and
exit status. The aggregate gate requires matching manifests from all five
shards, nonempty selections, an exact disjoint/complete node partition, complete
outcomes and matching successful JUnit identities in original order. Collection-only
reports are useful diagnostics and cannot satisfy the execution gate.

The earlier 17â€“19 minute projection was an estimate. The corrected run's actual
counts and durations must be reported separately. Redistribution does not reduce
the number of tests, total test work or runner cost; added job setup and aggregate
reconciliation may increase cost.

The October 11 revision 2 refresh uses completed current-main/#2416 XML costs
with a labelled 15% advisory variance estimate, and separately labelled
incomplete cancellation observations. The scheduler/schema, fallback, whole-file
order and three-job workflow are unchanged. See
[the exact evidence and phase breakdown](paid-launch/ci-timing-profile-v2.md).
Revision 3 also uses the first refresh's completed same-code shards and explicitly
estimated cancelled inter-outcome intervals; all-node progress is not completed
JUnit evidence. Earlier results remain separate from final-head execution.
Final readiness requires five fresh executed manifests under the same sealed
head/profile; older passed shards under another assignment do not count.

## Local verification

Install the same application dependencies as CI, plus pytest:

```sh
python -m pip install -r requirements.txt pytest
python -m pytest -q
# One selected module:
python -m pytest -q tests/test_locked_picks.py
```

`pytest.ini` sets default discovery to `tests/`, matching CI. Root-level manual
diagnostics and archived experiments are not collected by default. An explicit
file path can still be supplied when intentional. The obsolete `run_tests.py`
dummy and unittest-based `run_all_tests.py` have been removed; neither ran the
maintained pytest suite correctly.

To reproduce the CI partitions:

```sh
python scripts/run_ci_tests.py --shard 1 --shards 5 --assignment-manifest test-results/full-suite-1-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-1.xml
python scripts/run_ci_tests.py --shard 2 --shards 5 --assignment-manifest test-results/full-suite-2-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-2.xml
python scripts/run_ci_tests.py --shard 3 --shards 5 --assignment-manifest test-results/full-suite-3-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-3.xml
python scripts/run_ci_tests.py --shard 4 --shards 5 --assignment-manifest test-results/full-suite-4-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-4.xml
python scripts/run_ci_tests.py --shard 5 --shards 5 --assignment-manifest test-results/full-suite-5-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-5.xml
python scripts/run_ci_tests.py --shards 5 --reconcile-manifests test-results
```

Run local shards sequentially unless they have separate working directories; some application tests use repository-relative paths. GitHub shards have separate runners.

The September 12 baseline ran 1,764 tests in 237 seconds on GitHub; repository checkout took only one second. Local profiling found distributed test cost rather than one long-running test. Warnings remain visible: DataFrame fragmentation and deprecation warnings require separate behavior-preserving fixes, not blanket suppression. Repo history or saved research does not need deletion to obtain this CI improvement.

## Five-shard capacity successor

The owner approved five jobs after the retained #2419 three-shard timeouts and the matched 16-test Ubuntu comparison (run 38336966477). This changes capacity only: the scheduler, whole-file order, advisory file costs, network blocking, production safety, other workflows and 30-minute per-job limit remain unchanged. The existing exact matrix cardinality assertion now requires five entries; no test identity or rejection assertion is removed.

Every one of the five jobs must upload its own JUnit and assignment manifest. The existing full-suite gate rejects failure, cancellation, absent or empty results and mismatched identities/profile/assignments. Old three-shard synthetic regressions continue running. New five-shard regressions exercise the actual CLI and fail-closed reconciliation. Readiness requires all five final-head executed reports and all other required checks; previous-head passes and collection-only previews cannot satisfy it.

The completed metadata correction preserves frozen revision 3 and records successor metadata separately. It fixes an assertion failure and provides no runtime savings. The reviewed capacity estimate was 112.92 aggregate application runner minutes and worst job 26m28s with about 3m32s headroom, using compatible completed historical costs plus a 15% margin and 100s per-job overhead. New regression identities require a fresh assignment preview. These projections are not demonstrated performance; actual final-head test/job/upload times must be retained separately. More jobs do not reduce total test work and introduce extra setup/report overhead. Earlier failures remain preserved.


## NCAAF staged-route capacity follow-up

The current application matrix uses **nine whole-file shards**, each with the
unchanged 30-minute limit. The aggregate requires nine complete XML/assignment
artifacts and exact-once execution of the actual collected inventory. Scheduler,
within-file order, network blocking and substantive rejection assertions stay
unchanged. Historical three/five/six/seven-shard tests retain their old contracts;
only the three explicit current-workflow cardinality assertions move to nine.

43 material upward advisory costs use compatible completed passing-file XML sums
and the existing 15% variance method, with source/checkout/artifact provenance.
Frozen revision 3 metadata and old observations remain retained; successor
metadata is separate. Cancelled spans and profiling samples are not completed
release coverage. Matched diagnostics passed the same 58 identities on both
revisions and did not reproduce the historical slowdown. Its cause is UNKNOWN.

Forecasts use the complete final inventory plus setup/collection/reporting/upload
and cleanup allowances. Runner minutes measure resource consumption, not elapsed
time. Available hosted concurrency and queue delays are not assumed; nine matrix
entries do not establish nine simultaneous runners. All forecasts are estimates.
