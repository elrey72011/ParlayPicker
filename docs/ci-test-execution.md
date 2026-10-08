# CI test execution

The CI workflow keeps `production-safety` as a focused check and `full-suite` as the required gate for all application tests. The complete suite runs in three separate GitHub jobs, each with its own checkout and runner filesystem. `scripts/run_ci_tests.py` partitions whole collected files deterministically using the versioned advisory timing profile and retains collection order within each file. Every collected test belongs to exactly one shard. All three jobs must succeed; a failure, cancellation, missing result or unexpectedly empty shard prevents the existing `full-suite` gate from passing. The 30-minute application-job limit and production-safety selection remain unchanged.

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
exit status. The aggregate gate requires matching manifests from all three
shards, nonempty selections, an exact disjoint/complete node partition, complete
outcomes and matching successful JUnit identities in original order. Collection-only
reports are useful diagnostics and cannot satisfy the execution gate.

The earlier 17–19 minute projection was an estimate. The corrected run's actual
counts and durations must be reported separately. Redistribution does not reduce
the number of tests, total test work or runner cost; added job setup and aggregate
reconciliation may increase cost.

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
python scripts/run_ci_tests.py --shard 1 --shards 3 --assignment-manifest test-results/full-suite-1-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-1.xml
python scripts/run_ci_tests.py --shard 2 --shards 3 --assignment-manifest test-results/full-suite-2-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-2.xml
python scripts/run_ci_tests.py --shard 3 --shards 3 --assignment-manifest test-results/full-suite-3-assignment.json -v tests --durations=30 --junitxml=test-results/full-suite-3.xml
python scripts/run_ci_tests.py --shards 3 --reconcile-manifests test-results
```

Run local shards sequentially unless they have separate working directories; some application tests use repository-relative paths. GitHub shards have separate runners.

The September 12 baseline ran 1,764 tests in 237 seconds on GitHub; repository checkout took only one second. Local profiling found distributed test cost rather than one long-running test. Warnings remain visible: DataFrame fragmentation and deprecation warnings require separate behavior-preserving fixes, not blanket suppression. Repo history or saved research does not need deletion to obtain this CI improvement.
